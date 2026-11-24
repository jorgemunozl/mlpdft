import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from mlpdft import uq_evaluate


def frames(n, runs, rng, scale=1.0):
    out = []
    for i in range(n):
        a = Atoms("LiF" * 2, positions=rng.normal(size=(4, 3)), cell=[5, 5, 5], pbc=True)
        a.arrays["REF_forces"] = rng.normal(size=(4, 3))
        a.info.update({"run": runs[i % len(runs)], "scf_step": 10 * i, "shift_scale": scale})
        out.append(a)
    return out


class EvaluateTests(unittest.TestCase):
    def test_end_to_end_with_mock_predictor(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, run = tmp / "data", tmp / "run"
            data.mkdir()
            run.mkdir()
            sets = {
                "train_pool": ("train", frames(5, ["t"], rng)),
                "cal_replicas": ("calibration", frames(60, ["r1", "r2", "r3"], rng)),
                "test_replicas": ("test", frames(60, ["r4", "r5", "r6"], rng)),
                "shift_far": ("shift_far", frames(60, ["s"], rng, scale=3.0)),
            }
            manifest = {"sets": {}}
            for name, (role, fr) in sets.items():
                write(data / f"{name}.extxyz", fr, format="extxyz")
                manifest["sets"][name] = {"role": role}
            (data / "manifest.json").write_text(json.dumps(manifest))
            arms = {"ens": {"kind": "ensemble", "members": [{"name": "m0", "model_path": "x", "head": None}] * 3},
                    "single": {"kind": "ensemble", "members": [{"name": "m0", "model_path": "x", "head": None}]}}
            (run / "arms_manifest.json").write_text(json.dumps(arms))

            def mock(arm, run_dir, fr, device, dtype):
                ref = np.concatenate([a.arrays["REF_forces"] for a in fr])
                scale = np.concatenate([np.full((len(a), 3), a.info["shift_scale"]) for a in fr])
                sigma = 0.1 * scale  # overconfident by 2x: errors drawn with 0.2 * scale
                mean = ref + rng.normal(0.0, 2 * sigma)
                return (mean, sigma) if len(arm["members"]) > 1 else (mean, None)

            report = uq_evaluate.evaluate(run, data, device="cpu", dtype="float64", block_steps=100,
                                          n_boot=50, target=0.9, predictor=mock)
            ens = report["arms"]["ens"]
            self.assertAlmostEqual(ens["alpha"], 2.0, delta=0.15)
            test_row = ens["sets"]["test_replicas"]
            self.assertEqual(test_row["blocks"], "replica")
            self.assertAlmostEqual(test_row["point"]["conformal_coverage_90"], 0.9, delta=0.05)
            self.assertGreater(ens["sets"]["shift_far"]["shift_auroc_vs_id"], 0.95)
            self.assertIn("accuracy only", report["arms"]["single"]["sets"]["test_replicas"]["note"])
            self.assertTrue((run / "evaluation" / "metrics.json").exists())
            self.assertTrue((run / "evaluation" / "ens" / "shift_far.npz").exists())


class EvaluatorPartsTest(unittest.TestCase):
    def test_batched_forces_match_the_calculator_for_every_head(self):
        import torch
        from mace.calculators.mace import MACECalculator

        from toy_mace import lif_frames, toy_model

        frames = lif_frames(5, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.model"
            torch.save(toy_model(heads=("h0", "h1")), path)
            members = [{"model_path": "toy.model", "head": "h0"}, {"model_path": "toy.model", "head": "h1"}]
            got = uq_evaluate.ensemble_forces(members, Path(tmp), frames, "cpu", "float64", batch_size=2)
            self.assertEqual(got.shape, (2, 5 * 8, 3))
            for index, head in enumerate(("h0", "h1")):
                calc = MACECalculator(model_paths=[str(path)], device="cpu", default_dtype="float64", head=head)
                expected = []
                for atoms in frames:
                    atoms = atoms.copy()
                    atoms.calc = calc
                    expected.append(atoms.get_forces())
                np.testing.assert_allclose(got[index], np.concatenate(expected), atol=1e-10)
            self.assertGreater(np.abs(got[0] - got[1]).max(), 1e-8)

    def test_parallel_bootstrap_matches_in_process(self):
        rng = np.random.default_rng(1)
        tasks = {}
        for name in ("a", "b", "c"):
            frame = np.repeat(np.arange(20), 4)
            sigma = np.abs(rng.normal(0.1, 0.02, size=(80, 3))) + 0.01
            err = rng.normal(size=(80, 3)) * sigma
            tasks[name] = (err, sigma, frame, np.arange(20) // 5, 1.2, 1.5, 0.9, 20)
        serial = uq_evaluate._summaries(tasks, workers=1)
        parallel = uq_evaluate._summaries(tasks, workers=2)
        self.assertEqual(json.dumps(serial, sort_keys=True), json.dumps(parallel, sort_keys=True))

    def test_zero_shot_is_added_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            manifest = {"foundation_model": "/foundation.model",
                        "arms": {"mhc": {"kind": "ensemble", "members": []}}}
            (run / "arms_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(sorted(uq_evaluate.load_arms(run)), ["mhc", "zero-shot"])
            self.assertEqual(sorted(uq_evaluate.load_arms(run, zero_shot=False)), ["mhc"])


if __name__ == "__main__":
    unittest.main()
