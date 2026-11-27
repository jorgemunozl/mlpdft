import json
import tempfile
import unittest
from pathlib import Path

from mlpdft import project_run_time as prt


def line(record):
    return "[timing] " + json.dumps(record)


SMOKE = """LABELS=200
ARM_GROUPS="zero-shot mhc frozen-probe,llpr shallow-nll snapshots committee"
EXTRA_ARGS="--epochs 6 --mhc-heads 4 --mhc-epochs 3 --probe-epochs 3 --shallow-epochs 3 --llpr-max-train-frames 20"
EVAL_ARGS="--n-boot 100 --max-frames-per-set 30"
UQC_CAL_FRAMES=30
"""
TARGET = """# baseline
LABELS=1000
ARM_GROUPS="zero-shot mhc frozen-probe,llpr shallow-nll snapshots committee"
EXTRA_ARGS="--epochs 51 --mhc-heads 4 --mhc-epochs 60 --probe-epochs 15 --shallow-epochs 15 --llpr-max-train-frames 300"
EVAL_ARGS="--n-boot 1000"
UQC_CAL_FRAMES=0
"""


class ProjectionTest(unittest.TestCase):
    def test_scales_the_measured_rates_to_the_target_settings(self):
        log = [line({"kind": "setup", "seconds": 300})]
        for epoch, seconds in enumerate((30.0, 10.0, 10.0)):  # epoch 0 is warm-up and is left out
            log.append(line({"kind": "mace_epoch", "run": "independent_seed_123", "epoch": epoch,
                             "samples": 200, "seconds": seconds}))
            log.append(line({"kind": "mace_epoch", "run": "mhc_frozen_readouts", "epoch": epoch,
                             "samples": 200, "seconds": seconds * 0.8}))
            log.append(line({"kind": "shallow_epoch", "epoch": epoch, "samples": 200, "heads": 4,
                             "seconds": seconds * 4}))
        log.append(line({"kind": "llpr_frames", "phase": "covariance", "frames": 20, "atoms": 1280, "seconds": 12.8}))
        log.append(line({"kind": "eval_member", "frames": 30, "atoms": 1920, "seconds": 1.92}))
        log.append(line({"kind": "step", "step": "train committee", "seconds": 100, "status": 0}))
        log.append(line({"kind": "bootstrap", "arm": "committee", "seconds": 2.0}))
        log.append(line({"kind": "step", "step": "evaluate committee", "seconds": 10, "status": 0}))
        log.append(line({"kind": "group", "group": "committee", "seconds": 130, "status": 0}))
        measured = prt.measure(prt.timing.parse(log))
        self.assertAlmostEqual(measured["per_sample"]["full"], 0.05)
        self.assertAlmostEqual(measured["per_sample"]["probe"], 0.04)  # falls back to the MHC rate
        self.assertAlmostEqual(measured["per_atom"]["llpr_frames"], 0.01)
        self.assertAlmostEqual(measured["overhead"]["committee"]["upload"], 20.0)
        with tempfile.TemporaryDirectory() as tmp:
            smoke_path, target_path = Path(tmp) / "smoke.env", Path(tmp) / "node.env"
            smoke_path.write_text(SMOKE)
            target_path.write_text(TARGET)
            smoke, target = prt.settings(prt.read_env(smoke_path)), prt.settings(prt.read_env(target_path))
        self.assertEqual((target.labels, target.epochs, target.shallow_epochs, target.n_boot), (1000, 51, 15, 1000))
        data = prt.Data(train_atoms=64, monitor_frames=103, monitor_atoms=64, eval_sets=[(300, 64), (100, 18)])
        rows = prt.project(measured, smoke, target, data)
        self.assertAlmostEqual(rows["committee"]["train"], 3 * 51 * 1000 * 0.05)
        self.assertAlmostEqual(rows["snapshots"]["train"], 51 * 1000 * 0.05)
        self.assertAlmostEqual(rows["mhc"]["train"], 60 * 1000 * 0.04)
        self.assertAlmostEqual(rows["shallow-nll"]["train"], 15 * 1000 * 0.2)
        eval_atoms = 300 * 64 + 100 * 18
        self.assertAlmostEqual(rows["frozen-probe,llpr"]["llpr"], (300 * 64 + 103 * 64 + eval_atoms) * 0.01)
        self.assertAlmostEqual(rows["committee"]["evaluate"], 3 * eval_atoms * 0.001)
        text = prt.report(measured, smoke, target, data, target_path, deadline_minutes=630)
        self.assertIn("TOTAL", text)
        self.assertIn("deadline 630 min", text)

        # One discarded burn-in cycle: four 17-epoch cycles instead of three.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "node.env"
            path.write_text(TARGET.replace('--epochs 51 ', '--epochs 51 --snapshot-burn-in-cycles 1 '))
            burn_in = prt.settings(prt.read_env(path))
        self.assertEqual(burn_in.snapshot_burn_in_cycles, 1)
        rows = prt.project(measured, smoke, burn_in, data)
        self.assertAlmostEqual(rows["snapshots"]["train"], 68 * 1000 * 0.05)


if __name__ == "__main__":
    unittest.main()
