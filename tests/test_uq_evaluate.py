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


if __name__ == "__main__":
    unittest.main()
