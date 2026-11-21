import importlib.util
import unittest
from pathlib import Path

import numpy as np

# src/mlpdft/utils.py shadows the utils/ script folder, so load the script by path.
_PATH = Path(__file__).resolve().parents[1] / "src" / "mlpdft" / "utils" / "estimate_qe_correlation.py"
_SPEC = importlib.util.spec_from_file_location("estimate_qe_correlation", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
analyse = _MODULE.analyse
force_autocorrelation = _MODULE.force_autocorrelation
integrated_tau = _MODULE.integrated_tau


def ar1(n: int, phi: float, rng: np.random.Generator, shape: tuple[int, ...] = ()) -> np.ndarray:
    x = np.zeros((n, *shape))
    noise = rng.normal(size=(n, *shape))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


class AssumptionCheckTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.n = 40_000
        self.no_volume = np.full(self.n, np.nan)

    def test_stationary_ar1_passes_and_matches_theory(self):
        # Sokal convention: tau_int = 1/2 + sum_k phi^k = 1/2 + phi / (1 - phi) = 9.5.
        result = analyse(ar1(self.n, 0.9, self.rng), self.no_volume, burn_in=0)
        self.assertAlmostEqual(result["energy"]["tau_detrended_steps"], 9.5, delta=1.5)
        self.assertEqual(result["warnings"], [])
        self.assertTrue(result["stride_reliable"])

    def test_oscillation_is_flagged(self):
        t = np.arange(self.n)
        series = np.sin(2 * np.pi * t / 100) + 0.05 * self.rng.normal(size=self.n)
        result = analyse(series, self.no_volume, burn_in=0)
        self.assertFalse(result["stride_reliable"])
        self.assertTrue(any("oscillatory" in w for w in result["warnings"]))
        self.assertAlmostEqual(result["energy"]["dominant_period_steps"], 100, delta=2)

    def test_drift_is_flagged(self):
        series = ar1(self.n, 0.5, self.rng) + np.linspace(0, 10, self.n)
        result = analyse(series, self.no_volume, burn_in=0)
        self.assertTrue(result["drift_warning"])
        self.assertFalse(result["stride_reliable"])

    def test_force_acf_pools_atoms_and_components(self):
        forces = ar1(20_000, 0.8, self.rng, shape=(4, 3))
        tau = integrated_tau(force_autocorrelation(forces))["tau"]
        self.assertAlmostEqual(tau, 0.5 + 0.8 / 0.2, delta=0.6)


if __name__ == "__main__":
    unittest.main()
