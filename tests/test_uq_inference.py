import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlpdft import uq_inference as ui

ATOMS = 4


def _set_frames(runs, per_run, step0=0):
    run = np.repeat(np.array(runs), per_run)
    scf = step0 + 10 * np.arange(len(run))
    return run, scf


def write_runs(tmp: Path, n_seeds=3, per_run=60, quality=None, seed=0):
    """Three seed-set runs in the uq_evaluate layout.

    Each frame has a true error scale ``d``. ``quality[arm]`` sets how well the
    arm's sigma follows ``d`` (1 = exactly, 0 = unrelated); ``err_scale[arm]``
    multiplies its errors.
    """
    quality = quality or {"committee": 0.5, "snapshots": 0.95, "mhc": 0.5}
    err_scale = {"committee": 1.0, "snapshots": 1.0, "mhc": 2.0, "frozen-probe": 1.0}
    sets = {"cal_replicas": ("calibration", ["c1", "c2", "c3"]),
            "test_replicas": ("test", ["t1", "t2", "t3"]),
            "test_timeblock": ("test_timeblock", ["tb"]),
            "shift_x": ("shift_chemistry", ["sx"])}
    rng = np.random.default_rng(seed)
    truth = {}
    for name, (_, runs) in sets.items():
        run, scf = _set_frames(runs, per_run if len(runs) > 1 else 3 * per_run)
        scale = 3.0 if name == "shift_x" else 1.0
        truth[name] = (run, scf, scale * rng.lognormal(0, 0.5, len(run)))
    dirs = []
    for s in range(n_seeds):
        r = np.random.default_rng(100 + s)
        ev = tmp / f"seed{s}" / "evaluation"
        report = {"arms": {}}
        for arm in ("committee", "snapshots", "mhc", "frozen-probe"):
            report["arms"][arm] = {"sets": {n: {"role": role} for n, (role, _) in sets.items()}}
            (ev / arm).mkdir(parents=True, exist_ok=True)
            for name, (run, scf, d) in truth.items():
                d_atom = np.repeat(d, ATOMS)[:, None]
                err = err_scale[arm] * d_atom * r.standard_normal((len(d_atom), 3))
                if arm == "frozen-probe":
                    sigma = np.full_like(err, np.nan)
                else:
                    q = quality[arm]
                    noise = r.lognormal(0, 0.5, (len(d), 1)).repeat(ATOMS, 0)
                    sigma = np.broadcast_to((d_atom ** q) * noise ** (1 - q) * 0.5, err.shape).copy()
                frame = np.repeat(np.arange(len(d)), ATOMS)
                np.savez_compressed(ev / arm / f"{name}.npz", err=err, sigma=sigma, frame=frame,
                                    run=run, scf_step=scf)
        (ev / "metrics.json").write_text(json.dumps(report))
        dirs.append(tmp / f"seed{s}")
    return dirs


class IntervalTests(unittest.TestCase):
    def test_two_way_matches_the_formula(self):
        cells = np.array([[1.0, 2.0, 3.0], [2.0, 2.5, 4.0], [0.5, 1.5, 2.0]])
        iv = ui.two_way_interval(cells)
        msd = 3 * np.var(cells.mean(axis=1), ddof=1)
        mss = 3 * np.var(cells.mean(axis=0), ddof=1)
        self.assertAlmostEqual(iv["se"], np.sqrt((msd + mss) / 9))
        self.assertAlmostEqual(iv["df"], (msd + mss) ** 2 / (msd ** 2 / 2 + mss ** 2 / 2))
        self.assertAlmostEqual(iv["mean"], cells.mean())
        self.assertLess(iv["low"], iv["mean"])

    def test_one_unit_is_a_seed_interval(self):
        iv = ui.two_way_interval(np.array([[1.0, 2.0, 3.0]]))
        self.assertTrue(iv["conditional"])
        self.assertAlmostEqual(iv["df"], 2.0)

    def test_holm(self):
        out = ui.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(out["a"]["p_holm"], 0.03)
        self.assertAlmostEqual(out["c"]["p_holm"], 0.06)
        self.assertAlmostEqual(out["b"]["p_holm"], 0.06)
        self.assertEqual([out[k]["reject"] for k in "acb"], [True, False, False])

    def test_seed_inconsistency(self):
        iv = {"low": 0.1, "high": 0.5, "per_seed": [0.3, 0.4, -0.1]}
        self.assertEqual(ui.compare_verdict(iv, True), "inconsistent across seeds")
        self.assertEqual(ui.compare_verdict({"low": -0.5, "high": -0.1, "per_seed": [-0.2, -0.3, -0.1]}, False),
                         "better")


class EndToEndTests(unittest.TestCase):
    def test_verdicts_on_synthetic_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = write_runs(Path(tmp))
            roles, data = ui.load_runs(dirs)
            report = ui.infer(roles, data, {"shift_x": 600})
        test = report["sets"]["test_replicas"]
        self.assertEqual(test["units"], 3)
        self.assertTrue(report["calibrated"])
        snap, mhc = test["arms"]["snapshots"], test["arms"]["mhc"]
        self.assertGreater(snap["delta_rho_frame"]["mean"], 0)
        self.assertEqual(snap["rmse_ratio"]["verdict"], "comparable")
        self.assertEqual(mhc["rmse_ratio"]["verdict"], "worse")
        self.assertEqual(mhc["delta_rho_frame"]["verdict"], "not separated")
        self.assertNotIn("rho_frame", test["arms"]["frozen-probe"])
        self.assertIn("rmse_ratio", test["arms"]["frozen-probe"])
        self.assertEqual(set(report["primary"]), {f"{a}:{m}" for a in ("snapshots", "mhc")
                                                   for m in ("rho_frame", "miscal_cal")})
        self.assertEqual(report["sets"]["shift_x"]["units"], 3)
        self.assertEqual(report["sets"]["shift_x"]["arms"]["committee"]["auroc_vs_id"]["verdict"],
                         "detects the shift")
        self.assertIn("committee", report["leakage"])

    def test_single_trajectory_without_blocks_is_conditional(self):
        with tempfile.TemporaryDirectory() as tmp:
            roles, data = ui.load_runs(write_runs(Path(tmp)))
            report = ui.infer(roles, data)
        shift = report["sets"]["shift_x"]
        self.assertEqual(shift["units"], 1)
        self.assertTrue(shift["arms"]["committee"]["rho_frame"]["conditional"])


if __name__ == "__main__":
    unittest.main()
