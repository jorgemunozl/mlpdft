"""Regression checks for separate calibration and all-head smoke validation."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms

from mlpdft.uq_compare import _evaluate_method, _relative_log_likelihood, _smoke_check_mhc


def frame(reference_force):
    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.info["REF_energy"] = 0.0
    atoms.info["config_type"] = "bulk_LiF"
    atoms.arrays["REF_forces"] = np.full((1, 3), reference_force, dtype=float)
    return atoms


class FakeCalculator:
    def __init__(self, *, model_paths, device, default_dtype, head):
        self.head = head
        self.available_heads = ["head_0", "head_1"]

    def get_potential_energy(self, atoms, force_consistent=False):
        return 1.0 if self.head == "head_0" else 3.0

    def get_forces(self, atoms):
        return np.full((len(atoms), 3), 1.0 if self.head == "head_0" else 3.0)


class InvalidSecondHead(FakeCalculator):
    def get_forces(self, atoms):
        if self.head == "head_1":
            return np.full((len(atoms), 3), np.nan)
        return super().get_forces(atoms)


class EvaluationTests(unittest.TestCase):
    def test_rll_reference_comes_from_calibration_not_test(self):
        members = [("head_0", Path("unused"), "head_0"), ("head_1", Path("unused"), "head_1")]
        opts = argparse.Namespace(device="cpu", dtype="float64")
        with tempfile.TemporaryDirectory() as tmp, patch("mlpdft.uq_compare.MACECalculator", FakeCalculator):
            summary = _evaluate_method("test", members, [frame(6.0)] * 3, opts, Path(tmp),
                                       calibration_frames=[frame(0.0)] * 3)
        self.assertAlmostEqual(summary["calibration"]["baseline_force_rmse_ev_per_angstrom"], 2.0)
        expected = _relative_log_likelihood(np.full(9, -4.0), np.full(9, 2.0), 2.0)
        self.assertAlmostEqual(summary["overall"]["rll_force"], expected)

    def test_smoke_checks_every_head_for_nonfinite_forces(self):
        opts = argparse.Namespace(device="cpu", dtype="float64")
        with patch("mlpdft.uq_compare.MACECalculator", InvalidSecondHead):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                _smoke_check_mhc(Path("unused"), ["head_0", "head_1"], [frame(0.0)], opts)


if __name__ == "__main__":
    unittest.main()
