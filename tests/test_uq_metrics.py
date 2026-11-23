import unittest

import numpy as np

from mlpdft import uq_metrics as m


def synthetic(n_frames=200, atoms=8, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.01, 0.1, size=(n_frames * atoms, 3))
    err = rng.normal(0.0, scale * sigma)
    frame = np.repeat(np.arange(n_frames), atoms)
    return err, sigma, frame


class MetricTests(unittest.TestCase):
    def test_calibrated_gaussian(self):
        err, sigma, _ = synthetic(n_frames=2000)
        self.assertAlmostEqual(m.coverage(err, sigma, 1.0), 0.6827, delta=0.01)
        self.assertAlmostEqual(m.coverage(err, sigma, 2.0), 0.9545, delta=0.005)
        self.assertAlmostEqual(m.nll_scale(err, sigma), 1.0, delta=0.02)
        self.assertLess(m.miscalibration_area(err, sigma), 0.01)
        self.assertAlmostEqual(m.conformal_multiplier(err, sigma, 0.9), 1.645, delta=0.03)

    def test_overconfidence_is_recovered_by_alpha(self):
        err, sigma, _ = synthetic(n_frames=2000, scale=2.0, seed=1)
        alpha = m.nll_scale(err, sigma)
        self.assertAlmostEqual(alpha, 2.0, delta=0.05)
        self.assertGreater(m.miscalibration_area(err, sigma), 0.1)
        self.assertAlmostEqual(m.coverage(err, alpha * sigma, 1.0), 0.6827, delta=0.01)

    def test_auroc(self):
        rng = np.random.default_rng(2)
        same = m.auroc(rng.normal(size=4000), rng.normal(size=4000))
        self.assertAlmostEqual(same, 0.5, delta=0.02)
        self.assertEqual(m.auroc(np.zeros(5), np.ones(5)), 1.0)
        self.assertEqual(m.auroc(np.ones(5), np.ones(7)), 0.5)

    def test_per_frame_and_per_atom(self):
        err = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        sigma = np.ones((3, 3))
        frame = np.array([0, 0, 1])
        atom_err, atom_sigma = m.per_atom(err, sigma)
        np.testing.assert_allclose(atom_err, [5.0, 0.0, np.sqrt(3)])
        np.testing.assert_allclose(atom_sigma, np.sqrt(3) * np.ones(3))
        rmse, srms = m.per_frame(err, sigma, frame)
        np.testing.assert_allclose(rmse, [np.sqrt(25 / 6), 1.0])
        np.testing.assert_allclose(srms, [1.0, 1.0])

    def test_ranking_detects_informative_sigma(self):
        rng = np.random.default_rng(3)
        frame = np.repeat(np.arange(300), 8)
        sigma = rng.uniform(0.2, 2.0, 300)[frame][:, None] * np.full((len(frame), 3), 0.05)
        err = rng.normal(0.0, sigma)
        _, s_frame = m.per_frame(err, sigma, frame)
        e_frame, _ = m.per_frame(err, sigma, frame)
        self.assertGreater(m.spearman(e_frame, s_frame), 0.5)
        shuffled = np.random.default_rng(4).permutation(s_frame)
        self.assertLess(abs(m.spearman(e_frame, shuffled)), 0.2)

    def test_summarize_with_calibration_and_blocks(self):
        err, sigma, frame = synthetic(n_frames=120, scale=1.5, seed=5)
        block = np.arange(120) // 20
        cal_err, cal_sigma, _ = synthetic(n_frames=200, scale=1.5, seed=6)
        out = m.summarize(err, sigma, frame, block, alpha=m.nll_scale(cal_err, cal_sigma),
                          conformal_q=m.conformal_multiplier(cal_err, cal_sigma, 0.9), n_boot=200)
        point = out["point"]
        self.assertAlmostEqual(point["coverage_1sigma_cal"], 0.6827, delta=0.03)
        self.assertAlmostEqual(point["conformal_coverage_90"], 0.9, delta=0.03)
        for key, (lo, hi) in out["ci95"].items():
            if np.isfinite(point[key]):
                self.assertLessEqual(lo, point[key] + 1e-9, key)
                self.assertGreaterEqual(hi, point[key] - 1e-9, key)

    def test_shift_auroc(self):
        _, sigma_in, frame_in = synthetic(n_frames=100, seed=7)
        _, sigma_out, frame_out = synthetic(n_frames=100, seed=8)
        self.assertAlmostEqual(m.shift_auroc(sigma_in, frame_in, 3 * sigma_out, frame_out), 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
