"""Tests for the shallow-NLL and LLPR baselines (CPU, toy models, seconds).

Only ``FoundationReconstructionTest`` touches MACE-OMAT-medium: one forward
and one vector-Jacobian product on an 8-atom LiF cell.  Training the real
model is a GPU procedure (assets/research/methods/baselines.md).
"""

import argparse
import csv
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.filterwarnings("ignore")

import numpy as np
import torch

from toy_mace import FOUNDATION_MODEL, lif_frames, toy_model

from mace.calculators.mace import MACECalculator

from mlpdft import llpr
from mlpdft.mace_batches import collate, frames_to_data, load_model
from mlpdft.shallow_ensemble import (
    ShallowConfig,
    ensemble_nll,
    gaussian_nll,
    head_forces,
    per_head_mse,
    train_shallow_ensemble,
)


class LLPRLinearAlgebraTest(unittest.TestCase):
    def test_variance_equals_bayesian_linear_regression_posterior(self):
        """For a linear model the LLPR variance is the exact BLR predictive variance."""
        rng = np.random.default_rng(0)
        d, noise2, prior2 = 5, 0.3, 2.0
        # Training outputs grouped in "frames" of 4 outputs: J is (d, 4) per frame.
        jacs = [rng.normal(size=(d, 4)) for _ in range(6)]
        design = np.concatenate([jac.T for jac in jacs])  # (24, d)
        cov = llpr.covariance(jacs)
        np.testing.assert_allclose(cov, design.T @ design, atol=1e-12)
        x_new = rng.normal(size=(d, 7))
        alpha2, varsigma2 = noise2, noise2 / prior2
        llpr_var = alpha2 * llpr.Eigen.of(cov).quad(x_new, varsigma2)
        posterior = np.linalg.inv(design.T @ design / noise2 + np.eye(d) / prior2)
        blr_var = np.einsum("dc,de,ec->c", x_new, posterior, x_new)
        np.testing.assert_allclose(llpr_var, blr_var, rtol=1e-10)

    def test_energy_term_adds_outer_products(self):
        rng = np.random.default_rng(1)
        jacs = [rng.normal(size=(3, 2))]
        feats = [rng.normal(size=3)]
        cov = llpr.covariance(jacs, feats, energy_weight=0.5)
        np.testing.assert_allclose(cov, jacs[0] @ jacs[0].T + 0.5 * np.outer(feats[0], feats[0]))

    def test_calibration_recovers_alpha_and_alpha_is_nll_optimal(self):
        rng = np.random.default_rng(2)
        d = 6
        # Eigenvalues spread over six decades so varsigma^2 is identifiable
        # (with an isotropic C the NLL is flat in varsigma^2).
        basis, _ = np.linalg.qr(rng.normal(size=(d, d)))
        cov = basis @ np.diag(np.logspace(-4, 2, d)) @ basis.T
        eigen = llpr.Eigen.of(cov)
        true_varsigma2 = 1e-1
        true_alpha2 = 0.04
        jacs = [rng.normal(size=(d, 30)) * rng.uniform(0.1, 3.0, size=(1, 30)) for _ in range(400)]
        errors = [rng.normal(size=30) * np.sqrt(true_alpha2 * eigen.quad(j, true_varsigma2)) for j in jacs]
        fit = llpr.fit_calibration(eigen, jacs, errors, grid=np.array([true_varsigma2]))
        self.assertAlmostEqual(fit["alpha2"] / true_alpha2, 1.0, delta=0.03)
        u = np.concatenate([eigen.quad(j, true_varsigma2) for j in jacs])
        e = np.concatenate(errors)
        for factor in (0.8, 1.25):
            self.assertGreater(llpr.gaussian_nll(e, factor * fit["alpha2"] * u), fit["nll"])
        grid_fit = llpr.fit_calibration(eigen, jacs, errors)
        self.assertLess(abs(np.log10(grid_fit["varsigma2"] / true_varsigma2)), 0.5)
        self.assertFalse(grid_fit["at_grid_edge"])

    def test_jacobian_rows_match_analytic_gradient(self):
        positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        matrix = torch.randn(5, 12, dtype=torch.float64)
        outputs = matrix @ positions.reshape(-1) ** 2
        expected = -(matrix * 2 * positions.reshape(1, -1)).detach()
        for chunk in (1, 3):
            got = llpr._jacobian_rows(outputs, positions, chunk=chunk)
            torch.testing.assert_close(got, expected)


class ToyMaceLLPRTest(unittest.TestCase):
    """Feature decomposition on a 3-head ScaleShiftMACE with ZBL pair term."""

    @classmethod
    def setUpClass(cls):
        cls.model = toy_model()
        for parameter in cls.model.parameters():
            parameter.requires_grad_(False)
        cls.frames = lif_frames(3)

    def test_energy_and_forces_are_linear_in_last_layer_features(self):
        for head in self.model.heads:
            spec = llpr.last_layer(self.model, head)
            self.assertEqual(spec.dim, 6 + 4)  # layer-0 scalars + one head's hidden block
            data = frames_to_data(self.model, self.frames[:1], head)[0]
            feats = llpr.frame_features(self.model, spec, data, check=True)
            errors = llpr.reconstruction_errors(feats, spec)
            self.assertLess(errors["max_abs_force_error"], 1e-10)
            self.assertLess(errors["abs_energy_error"], 1e-10)
            self.assertGreater(errors["max_abs_readout_force"], 1e-4)
            # The single-VJP shortcut equals w^T J from the full Jacobian.
            vjp = llpr.frame_features(self.model, spec, data, jacobian=False)
            np.testing.assert_allclose(vjp["wJ"], (spec.weights @ feats["J"]).reshape(-1, 3), atol=1e-12)

    def test_fit_and_predict_give_finite_positive_sigmas(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "toy.model"
            torch.save(self.model, model_path)
            record = llpr.fit_llpr(model_path, "h1", self.frames[:2], self.frames[2:],
                                   Path(tmp) / "cov.npz", log=lambda *_: None)
            self.assertLess(record["max_abs_force_reconstruction_error"], 1e-10)
            self.assertEqual(record["feature_dim"], 10)
            stored = np.load(Path(tmp) / "cov.npz")
            for key in ("covariance", "weights", "varsigma2", "alpha2", "eigenvalues"):
                self.assertIn(key, stored.files)
            sigmas = llpr.predict_force_sigma(model_path, "h1", Path(tmp) / "cov.npz", self.frames)
            self.assertEqual([s.shape for s in sigmas], [(8, 3)] * 3)
            for sigma in sigmas:
                self.assertTrue(np.all(np.isfinite(sigma)) and np.all(sigma > 0))
            # Point predictions agree with MACECalculator for the same head.
            prediction = llpr.predict(model_path, "h1", Path(tmp) / "cov.npz", self.frames[:1])[0]
            atoms = self.frames[0].copy()
            atoms.calc = MACECalculator(model_paths=[str(model_path)], device="cpu",
                                        default_dtype="float64", head="h1")
            np.testing.assert_allclose(prediction["forces"], atoms.get_forces(), atol=1e-10)
            with self.assertRaises(ValueError):
                llpr.predict_force_sigma(model_path, "missing", Path(tmp) / "cov.npz", self.frames)


class NLLLossTest(unittest.TestCase):
    def test_finite_and_matches_formula(self):
        torch.manual_seed(0)
        pred = torch.randn(4, 10, 3, dtype=torch.float64)
        target = torch.randn(10, 3, dtype=torch.float64)
        loss = ensemble_nll(pred, target, eps=1e-6)
        mean = pred.mean(0)
        var = ((pred - mean) ** 2).mean(0) + 1e-6
        expected = torch.mean(0.5 * (torch.log(var) + (target - mean) ** 2 / var))
        torch.testing.assert_close(loss, expected)
        identical = ensemble_nll(pred[:1].expand(4, -1, -1), target, eps=1e-6)
        self.assertTrue(torch.isfinite(identical))

    def test_gradient_signs(self):
        target = torch.zeros(1, dtype=torch.float64)
        # Mean above the target: loss decreases when the mean moves down.
        offset = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        spread = torch.tensor([-0.1, 0.1], dtype=torch.float64)
        ensemble_nll((offset + spread)[:, None], target).backward()
        self.assertGreater(float(offset.grad), 0.0)
        # Residual^2 (0.25) >> variance (0.01): loss decreases when spread grows.
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        ensemble_nll((0.5 + scale * spread)[:, None], target).backward()
        self.assertLess(float(scale.grad), 0.0)
        # Residual^2 << variance: loss decreases when spread shrinks.
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        ensemble_nll((0.0 + scale * 10 * spread)[:, None], target).backward()
        self.assertGreater(float(scale.grad), 0.0)

    def test_fixed_variance_reduces_to_scaled_mse(self):
        torch.manual_seed(1)
        mean = torch.randn(20, dtype=torch.float64, requires_grad=True)
        target = torch.randn(20, dtype=torch.float64)
        var = torch.full((20,), 0.3, dtype=torch.float64)
        loss = gaussian_nll(mean, var, target)
        mse = torch.mean((mean - target) ** 2)
        torch.testing.assert_close(loss, 0.5 * mse / 0.3 + 0.5 * torch.log(torch.tensor(0.3, dtype=torch.float64)))
        (grad_nll,) = torch.autograd.grad(loss, mean)
        (grad_mse,) = torch.autograd.grad(0.5 * torch.mean((mean - target) ** 2) / 0.3, mean)
        torch.testing.assert_close(grad_nll, grad_mse)
        pred = torch.randn(3, 5, dtype=torch.float64)
        torch.testing.assert_close(per_head_mse(pred, target[:5]), torch.mean((pred - target[:5]) ** 2))


class ToyShallowEnsembleTest(unittest.TestCase):
    """A few optimiser steps on the toy model: plumbing, not learning."""

    def test_joint_training_updates_only_readouts_and_saves_multihead_model(self):
        model = toy_model(heads=("s0", "s1", "s2"))
        frames = lif_frames(4, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            init_path, out_path = Path(tmp) / "init.model", Path(tmp) / "out.model"
            torch.save(model, init_path)
            config = ShallowConfig(epochs=2, warmup_fraction=0.5, batch_size=2, patience=None, seed=5)
            record = train_shallow_ensemble(init_path, frames[:3], frames[3:], out_path, config,
                                            log_path=Path(tmp) / "log.jsonl", log=lambda *_: None)
            log = [json.loads(line) for line in (Path(tmp) / "log.jsonl").read_text().splitlines()]
            self.assertEqual([item["phase"] for item in log], ["mse", "nll"])
            self.assertEqual(record["selected_epoch"], 1)
            self.assertEqual(record["optimizer_steps"], 4)
            before = torch.load(init_path, weights_only=False).state_dict()
            after = torch.load(out_path, weights_only=False).state_dict()
            changed = {name for name in before if not torch.equal(before[name], after[name])}
            self.assertTrue(changed)
            self.assertTrue(all(name.startswith("readouts") for name in changed), changed)
            forces = {}
            for head in ("s0", "s1", "s2"):
                calc = MACECalculator(model_paths=[str(out_path)], device="cpu",
                                      default_dtype="float64", head=head)
                self.assertEqual(calc.head, head)
                atoms = frames[0].copy()
                atoms.calc = calc
                forces[head] = atoms.get_forces()
                self.assertTrue(np.all(np.isfinite(forces[head])))
            self.assertGreater(np.abs(forces["s0"] - forces["s1"]).max(), 1e-8)
            trained = load_model(out_path)
            joint = head_forces(trained, collate(frames_to_data(trained, frames[:1], "s0")), 3, training=False)
            for index, head in enumerate(("s0", "s1", "s2")):
                np.testing.assert_allclose(joint[index].detach().numpy(), forces[head], atol=1e-10)


@unittest.skipUnless(os.path.isfile(FOUNDATION_MODEL), "foundation model not available")
class FoundationReconstructionTest(unittest.TestCase):
    """One forward + one VJP of MACE-OMAT-medium on an 8-atom LiF cell (~2 s)."""

    def test_wT_J_reproduces_foundation_forces(self):
        model = load_model(FOUNDATION_MODEL)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        spec = llpr.last_layer(model, None)
        self.assertEqual(spec.dim, 128 + 16)
        data = frames_to_data(model, lif_frames(1), spec.head)[0]
        errors = llpr.reconstruction_errors(
            llpr.frame_features(model, spec, data, jacobian=False), spec)
        print(f"\n[foundation] d={spec.dim} {errors}")
        self.assertLess(errors["max_abs_force_error"], 1e-6)
        self.assertLess(errors["abs_energy_error"], 1e-6)


class RunnerIntegrationTest(unittest.TestCase):
    def test_arms_manifest_uses_relative_paths(self):
        from mlpdft.uq_compare import _write_arms_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "shallow_nll" / "models" / "shallow_nll.model"
            methods = {
                "shallow_nll": [("shallow_head_0", model, "shallow_head_0"),
                                ("shallow_head_1", model, "shallow_head_1")],
                "frozen_probe": [("frozen_probe", root / "frozen_probe" / "models" / "frozen_probe.model", None)],
            }
            entry = {"model": str(root / "frozen_probe" / "models" / "frozen_probe.model"), "head": None,
                     "covariance_file": str(root / "llpr" / "llpr_covariance.npz")}
            manifest = json.loads(_write_arms_manifest(root, methods, entry).read_text())
        arms = manifest["arms"]
        self.assertEqual(sorted(arms), ["frozen-probe", "llpr", "shallow-nll"])
        self.assertEqual(arms["shallow-nll"]["members"][1],
                         {"name": "shallow_head_1", "model_path": "shallow_nll/models/shallow_nll.model",
                          "head": "shallow_head_1"})
        self.assertEqual(arms["llpr"]["covariance"], "llpr/llpr_covariance.npz")
        self.assertIsNone(arms["frozen-probe"]["members"][0]["head"])

    def test_llpr_evaluation_writes_the_ensemble_formats(self):
        from mlpdft.uq_compare import _evaluate_llpr, _evaluate_method

        test, cal = lif_frames(4, seed=7), lif_frames(3, seed=8)
        rng = np.random.default_rng(0)

        def fake_predict(model_path, head, cov, frames, device, dtype):
            return [{"energy": -20.0 + rng.normal(), "forces": rng.normal(size=(len(a), 3)),
                     "force_sigma": rng.uniform(0.1, 0.5, size=(len(a), 3)),
                     "energy_sigma": float(rng.uniform(0.01, 0.1))} for a in frames]

        class FakeCalculator:
            def __init__(self, **kwargs):
                self.offset = 0.1 if kwargs.get("head") == "b" else 0.0

            def get_potential_energy(self, atoms, force_consistent=False):
                return -20.0 + self.offset

            def get_forces(self, atoms):
                return np.full((len(atoms), 3), self.offset)

        opts = argparse.Namespace(device="cpu", dtype="float64")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch("mlpdft.uq_compare.llpr_lib.predict", fake_predict):
                summary = _evaluate_llpr("llpr", Path("m"), None, Path("c"), test, opts, out, calibration_frames=cal)
            with patch("mlpdft.uq_compare.MACECalculator", FakeCalculator):
                reference = _evaluate_method("ens", [("a", Path("m"), "a"), ("b", Path("m"), "b")],
                                             test, opts, out, calibration_frames=cal)
            for key in ("overall", "by_group", "calibration", "member_accuracy", "atom_level_file"):
                self.assertIn(key, summary)
            self.assertEqual(set(summary["overall"]), set(reference["overall"]))
            self.assertIsNotNone(summary["overall"]["rll_force"])
            self.assertIsNotNone(summary["overall"]["low_uncertainty_risk_coverage_force_rmse"])
            npz, ref_npz = np.load(out / "llpr_atom_level.npz"), np.load(out / "ens_atom_level.npz")
            self.assertEqual(sorted(npz.files), sorted(ref_npz.files))
            with open(out / "llpr_predictions.csv") as a, open(out / "ens_predictions.csv") as b:
                self.assertEqual(next(csv.reader(a)), next(csv.reader(b)))


if __name__ == "__main__":
    unittest.main()
