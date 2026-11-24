"""Last-layer prediction rigidity (LLPR) force uncertainties for MACE.

Bigi, Chong, Ceriotti, Grasselli, "A prediction rigidity formalism for
low-cost uncertainties in trained neural networks"
(https://www.alphaxiv.org/abs/2403.02251); used for foundation-model
fine-tuning in https://www.alphaxiv.org/abs/2607.14486.

For a MACE model the energy of head ``k`` is

    E = sum_i E0_{z_i} + sum_i [ s_k (pair_i + sum_l o_{l,i}) + b_k ]

where ``o_{l,i}`` is readout ``l`` on atom ``i``.  Intermediate readouts are
linear maps of the scalar (0e) node features; the last readout is
Linear -> activation -> Linear.  The *last-layer features* of a frame are the
inputs of those final linear maps summed over atoms,

    f = sum_i [ h^{(0)}_{i,0e}, ..., act(W_1 h^{(L)}_i) ]          (d values)

so that ``E = w^T f + E_rest`` with ``w = s_k [W_0[:, k]; ...; W_2[:, k]]`` and
``E_rest = sum E0 + s_k sum pair + N b_k`` (no trainable readout weight in it).
Force features are ``J = -df/dR`` (d x 3N) so ``F = w^T J + F_rest``.

The LLPR covariance is built from force features of training frames,
``C = sum_frames J J^T`` (optionally ``+ energy_weight * f f^T``), and the
per-component variance of a new prediction is

    sigma_c^2 = alpha^2 j_c^T (C + varsigma^2 I)^{-1} j_c .

``varsigma^2`` is chosen on a log grid and ``alpha^2`` in closed form
(``alpha^2 = mean(e_c^2 / u_c)``) by minimising the Gaussian NLL of the force
errors on a calibration set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase import Atoms

from mlpdft import timing
from mlpdft.mace_batches import (
    collate,
    frames_to_data,
    load_model,
    reference_forces,
    resolve_head,
    with_head,
)

LLPR_FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# Pure linear-algebra core (model-agnostic; exercised by the toy-model test)
# --------------------------------------------------------------------------


def covariance(jacobians: Sequence[np.ndarray], features: Sequence[np.ndarray] | None = None,
               energy_weight: float = 0.0) -> np.ndarray:
    """``C = sum J J^T (+ energy_weight * sum f f^T)``; each J is (d, n_outputs)."""
    d = jacobians[0].shape[0]
    cov = np.zeros((d, d))
    for jac in jacobians:
        cov += jac @ jac.T
    if energy_weight:
        if features is None:
            raise ValueError("energy_weight > 0 needs the energy features")
        for feat in features:
            cov += energy_weight * np.outer(feat, feat)
    return cov


@dataclass
class Eigen:
    values: np.ndarray  # (d,)
    vectors: np.ndarray  # (d, d)

    @classmethod
    def of(cls, cov: np.ndarray) -> "Eigen":
        cov = 0.5 * (cov + cov.T)
        values, vectors = np.linalg.eigh(cov)
        return cls(np.clip(values, 0.0, None), vectors)

    def quad(self, jac: np.ndarray, varsigma2: float) -> np.ndarray:
        """``j_c^T (C + varsigma^2 I)^{-1} j_c`` for every column of ``jac``."""
        projected = self.vectors.T @ jac
        return np.sum(projected**2 / (self.values[:, None] + varsigma2), axis=0)


def gaussian_nll(errors: np.ndarray, variances: np.ndarray) -> float:
    variances = np.maximum(variances, 1e-300)
    return float(np.mean(0.5 * (np.log(2 * np.pi * variances) + errors**2 / variances)))


def fit_calibration(eigen: Eigen, jacobians: Sequence[np.ndarray],
                    errors: Sequence[np.ndarray], grid: np.ndarray | None = None
                    ) -> dict[str, Any]:
    """Fit ``varsigma^2`` on a log grid and ``alpha^2`` in closed form.

    ``errors[i]`` are prediction - reference for the outputs (columns) of
    ``jacobians[i]``.  For fixed ``varsigma^2`` the NLL is minimised by
    ``alpha^2 = mean(e^2 / u)``.  The grid is relative to the largest
    eigenvalue of ``C`` (default ``lambda_max * 10^[-12, 2]``, 85 points).
    """
    e = np.concatenate([np.ravel(err) for err in errors])
    projected = np.concatenate([eigen.vectors.T @ jac for jac in jacobians], axis=1)
    lam_max = float(max(eigen.values.max(), 1e-300))
    if grid is None:
        grid = lam_max * np.logspace(-12, 2, 85)
    nll = np.empty(len(grid))
    alpha2 = np.empty(len(grid))
    for index, varsigma2 in enumerate(grid):
        u = np.sum(projected**2 / (eigen.values[:, None] + varsigma2), axis=0)
        u = np.maximum(u, 1e-300)
        alpha2[index] = float(np.mean(e**2 / u))
        nll[index] = gaussian_nll(e, alpha2[index] * u)
    best = int(np.argmin(nll))
    return {
        "varsigma2": float(grid[best]),
        "alpha2": float(alpha2[best]),
        "nll": float(nll[best]),
        "grid_varsigma2": np.asarray(grid),
        "grid_nll": nll,
        "grid_alpha2": alpha2,
        "at_grid_edge": best in (0, len(grid) - 1),
        "n_components": int(e.size),
    }


# --------------------------------------------------------------------------
# MACE last-layer features
# --------------------------------------------------------------------------


def _linear_matrix(linear: torch.nn.Module, dim_in: int, ref: torch.Tensor) -> torch.Tensor:
    """Dense matrix M with linear(x) = x @ M (e3nn Linear has no bias here)."""
    eye = torch.eye(dim_in, dtype=ref.dtype, device=ref.device)
    with torch.no_grad():
        matrix = linear(eye)
        zero = linear(torch.zeros(1, dim_in, dtype=ref.dtype, device=ref.device))
    if not torch.allclose(zero, torch.zeros_like(zero)):
        raise RuntimeError("Readout linear map has a bias; LLPR features assume none")
    return matrix


def _scalar_indices(irreps) -> list[int]:
    indices: list[int] = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        if ir.l == 0 and ir.p == 1:
            indices.extend(range(offset, offset + width))
        offset += width
    return indices


@dataclass
class LastLayer:
    """Where the last-layer features live in ``model`` and their weights."""

    head: str
    head_index: int
    n_heads: int
    blocks: list[dict[str, Any]]  # per readout: kind, feature indices, dim
    weights: np.ndarray  # (d,) effective weights including the head scale
    scale: float
    shift: float

    @property
    def dim(self) -> int:
        return int(self.weights.size)


def last_layer(model, head: str | None = None) -> LastLayer:
    head, k = resolve_head(model, head)
    n_heads = len(getattr(model, "heads", None) or ["Default"])
    ref = next(model.parameters())
    blocks: list[dict[str, Any]] = []
    weights: list[torch.Tensor] = []
    for index, readout in enumerate(model.readouts):
        if hasattr(readout, "linear_2"):
            linear = readout.linear_2
            dim_in = linear.irreps_in.dim
            matrix = _linear_matrix(linear, dim_in, ref)  # (dim_in, n_heads)
            if dim_in % n_heads:
                raise RuntimeError("Hidden readout width is not a multiple of the heads")
            width = dim_in // n_heads
            # mask_head keeps channels [k*width, (k+1)*width) for head k.
            idx = list(range(k * width, (k + 1) * width)) if n_heads > 1 else list(range(dim_in))
            blocks.append({"readout": index, "kind": "nonlinear_hidden", "indices": idx,
                           "dim_in": dim_in})
            weights.append(matrix[idx, k])
        elif hasattr(readout, "linear"):
            linear = readout.linear
            dim_in = linear.irreps_in.dim
            matrix = _linear_matrix(linear, dim_in, ref)
            idx = _scalar_indices(linear.irreps_in)
            others = sorted(set(range(dim_in)) - set(idx))
            if others and torch.max(torch.abs(matrix[others])) > 0:
                raise RuntimeError("Readout acts on non-scalar features")
            blocks.append({"readout": index, "kind": "linear_scalars", "indices": idx,
                           "dim_in": dim_in})
            weights.append(matrix[idx, k])
        else:
            raise RuntimeError(f"Unknown readout type {type(readout).__name__}")
    scale = float(torch.atleast_1d(model.scale_shift.scale)[k]) if hasattr(model, "scale_shift") else 1.0
    shift = float(torch.atleast_1d(model.scale_shift.shift)[k]) if hasattr(model, "scale_shift") else 0.0
    w = scale * torch.cat(weights).detach().cpu().numpy().astype(float)
    return LastLayer(head, k, n_heads, blocks, w, scale, shift)


JACOBIAN_CHUNK = 1  # rows per backward pass; >1 uses vmap (memory x chunk)
# Batched rows are checked once per process against one-row passes (values and
# speed); None = not checked yet, True = use them, False = one row per pass.
_BATCHED_ROWS_OK: bool | None = None


def _rows(outputs: torch.Tensor, positions: torch.Tensor, chunk: int) -> torch.Tensor:
    d = outputs.shape[0]
    rows = []
    eye = torch.eye(d, dtype=outputs.dtype, device=outputs.device)
    for start in range(0, d, chunk):
        stop = min(start + chunk, d)
        if stop - start == 1:
            rows.append(torch.autograd.grad(outputs, positions, grad_outputs=eye[start],
                                            retain_graph=True)[0][None])
        else:
            rows.append(torch.autograd.grad(outputs, positions, grad_outputs=eye[start:stop],
                                            is_grads_batched=True, retain_graph=True)[0])
    return -torch.cat(rows).reshape(d, -1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _jacobian_rows(outputs: torch.Tensor, positions: torch.Tensor,
                   chunk: int | None = None) -> torch.Tensor:
    """-d outputs / d positions, shape (len(outputs), 3N): one backward per row
    (or per ``chunk`` rows through ``is_grads_batched``).

    The first batched call is checked against one-row passes on the same
    frame; batching stays on only if it gives the same rows and is faster.
    """
    global _BATCHED_ROWS_OK
    chunk = chunk or JACOBIAN_CHUNK
    if chunk <= 1 or _BATCHED_ROWS_OK is False:
        return _rows(outputs, positions, 1)
    if _BATCHED_ROWS_OK:
        return _rows(outputs, positions, chunk)
    device = outputs.device
    _sync(device)
    start = time.perf_counter()
    reference = _rows(outputs, positions, 1)
    _sync(device)
    one_row = time.perf_counter() - start
    try:
        _rows(outputs, positions, chunk)  # warm-up of the vmap path
        _sync(device)
        start = time.perf_counter()
        batched = _rows(outputs, positions, chunk)
        _sync(device)
        batched_time = time.perf_counter() - start
        scale = float(torch.max(torch.abs(reference))) or 1.0
        diff = float(torch.max(torch.abs(batched - reference)))
        tolerance = (1e-4 if outputs.dtype == torch.float32 else 1e-9) * scale
        _BATCHED_ROWS_OK = diff <= tolerance and batched_time < 0.8 * one_row
        detail = (f"max |diff| {diff:.2e} (tolerance {tolerance:.1e}), "
                  f"{batched_time:.3f} s vs {one_row:.3f} s per frame")
    except RuntimeError as exc:  # an operator without a batching rule
        _BATCHED_ROWS_OK = False
        detail = f"failed: {str(exc).splitlines()[0][:200]}"
    print(f"[llpr] batched Jacobian rows, chunk {chunk}: "
          f"{'on' if _BATCHED_ROWS_OK else 'off (one row per backward pass)'}; {detail}", flush=True)
    return reference


def frame_features(model, spec: LastLayer, data, device: str = "cpu",
                   jacobian: bool = True, check: bool = False) -> dict[str, np.ndarray]:
    """Last-layer features of one frame.

    Returns ``f`` (d,), ``J`` (d, 3N) if requested, the model energy and
    forces, and the readout-independent remainder ``E_rest``/``F_rest`` so
    ``E = w.f + E_rest`` and ``F = (w^T J).reshape(N, 3) + F_rest``.
    """
    batch = with_head(collate([data], device), spec.head_index)
    batch["positions"].requires_grad_(True)
    positions = batch["positions"]
    captured: dict[int, torch.Tensor] = {}
    pair: list[torch.Tensor] = []
    hooks = []
    for block in spec.blocks:
        readout = model.readouts[block["readout"]]
        if block["kind"] == "nonlinear_hidden":
            target = readout.linear_2
        else:
            target = readout
        hooks.append(target.register_forward_pre_hook(
            lambda _m, args, i=block["readout"]: captured.__setitem__(i, args[0])))
    if hasattr(model, "pair_repulsion_fn"):
        hooks.append(model.pair_repulsion_fn.register_forward_hook(
            lambda _m, _a, out: pair.append(out)))
    try:
        out = model(batch, training=True, compute_force=False)
    finally:
        for hook in hooks:
            hook.remove()
    phi = torch.cat([captured[b["readout"]][:, b["indices"]] for b in spec.blocks], dim=1)
    if check and spec.n_heads > 1:
        for b in spec.blocks:
            if b["kind"] == "nonlinear_hidden":
                rest = sorted(set(range(b["dim_in"])) - set(b["indices"]))
                if rest and torch.max(torch.abs(captured[b["readout"]][:, rest])) > 0:
                    raise RuntimeError("Hidden channels of other heads are not masked")
    feats = phi.sum(dim=0)
    energy = out["energy"].sum()
    n_atoms = positions.shape[0]
    e0 = model.atomic_energies_fn(batch["node_attrs"])
    e0 = torch.atleast_2d(e0)
    e0 = (e0[:, spec.head_index] if e0.shape[-1] > 1 else e0.reshape(-1)).sum()
    pair_energy = pair[0].sum() if pair else torch.zeros((), dtype=energy.dtype, device=energy.device)
    e_rest = e0 + spec.scale * pair_energy + n_atoms * spec.shift
    (grad_e,) = torch.autograd.grad(energy, positions, retain_graph=True)
    if pair:
        (grad_pair,) = torch.autograd.grad(pair_energy, positions, retain_graph=True)
        f_rest = -spec.scale * grad_pair
    else:
        f_rest = torch.zeros_like(positions)
    result = {
        "f": feats.detach().cpu().numpy().astype(float),
        "energy": float(energy.detach()),
        "forces": (-grad_e).detach().cpu().numpy().astype(float),
        "E_rest": float(e_rest.detach()),
        "F_rest": f_rest.detach().cpu().numpy().astype(float),
    }
    if jacobian:
        result["J"] = _jacobian_rows(feats, positions).detach().cpu().numpy().astype(float)
    else:
        # w^T J by one vector-Jacobian product: -d(w.f)/dR.  Cheap check of
        # the decomposition without forming J (d backward passes).
        w = torch.as_tensor(spec.weights, dtype=feats.dtype, device=feats.device)
        (grad_wf,) = torch.autograd.grad(feats @ w, positions, retain_graph=False)
        result["wJ"] = (-grad_wf).detach().cpu().numpy().astype(float)
    return result


def reconstruction_errors(feats: dict[str, np.ndarray], spec: LastLayer) -> dict[str, float]:
    """max |F - (w^T J + F_rest)| and |E - (w.f + E_rest)| for one frame."""
    wj = feats["wJ"] if "wJ" in feats else (spec.weights @ feats["J"]).reshape(-1, 3)
    return {
        "max_abs_force_error": float(np.max(np.abs(wj + feats["F_rest"] - feats["forces"]))),
        "abs_energy_error": float(abs(spec.weights @ feats["f"] + feats["E_rest"] - feats["energy"])),
        "max_abs_force": float(np.max(np.abs(feats["forces"]))),
        "max_abs_readout_force": float(np.max(np.abs(wj))),
    }


# --------------------------------------------------------------------------
# Fit / save / predict
# --------------------------------------------------------------------------


def fit_llpr(model_path: str | Path, head: str | None, train_frames: Sequence[Atoms],
             calibration_frames: Sequence[Atoms], out_path: str | Path,
             device: str = "cpu", dtype: str = "float64", energy_weight: float = 0.0,
             max_train_frames: int | None = 300, seed: int = 0,
             log=print) -> dict[str, Any]:
    """Build C on training frames, calibrate on ``calibration_frames``, save npz."""
    model = load_model(model_path, device, dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = last_layer(model, head)
    n_total = len(train_frames)
    indices = np.arange(n_total)
    if max_train_frames is not None and n_total > max_train_frames:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_total, size=max_train_frames, replace=False))
    log(f"[llpr] head {spec.head}: d={spec.dim} features, covariance from "
        f"{len(indices)}/{n_total} training frames")
    train_data = frames_to_data(model, [train_frames[i] for i in indices], spec.head)
    cov = np.zeros((spec.dim, spec.dim))
    worst_force = 0.0
    worst_energy = 0.0
    train_selected = [train_frames[i] for i in indices]
    with timing.timed({"kind": "llpr_frames", "phase": "covariance", "frames": len(train_data),
                       "atoms": int(sum(len(a) for a in train_selected))}):
        for count, data in enumerate(train_data):
            feats = frame_features(model, spec, data, device, jacobian=True, check=count == 0)
            cov += feats["J"] @ feats["J"].T
            if energy_weight:
                cov += energy_weight * np.outer(feats["f"], feats["f"])
            force_rec = (spec.weights @ feats["J"]).reshape(-1, 3) + feats["F_rest"]
            worst_force = max(worst_force, float(np.max(np.abs(force_rec - feats["forces"]))))
            worst_energy = max(worst_energy, abs(spec.weights @ feats["f"] + feats["E_rest"] - feats["energy"]))
    log(f"[llpr] reconstruction on training frames: max|F - (w^T J + F_rest)| = "
        f"{worst_force:.3e} eV/A, max|E - (w.f + E_rest)| = {worst_energy:.3e} eV")
    eigen = Eigen.of(cov)
    cal_data = frames_to_data(model, calibration_frames, spec.head)
    jacs, errors = [], []
    with timing.timed({"kind": "llpr_frames", "phase": "calibration", "frames": len(cal_data),
                       "atoms": int(sum(len(a) for a in calibration_frames))}):
        for data, ref in zip(cal_data, reference_forces(calibration_frames)):
            feats = frame_features(model, spec, data, device, jacobian=True)
            jacs.append(feats["J"])
            errors.append((feats["forces"] - ref).ravel())
    fit = fit_calibration(eigen, jacs, errors)
    log(f"[llpr] calibration on {len(calibration_frames)} frames: varsigma^2="
        f"{fit['varsigma2']:.3e}, alpha^2={fit['alpha2']:.3e}, NLL={fit['nll']:.4f}"
        + (" (WARNING: optimum at grid edge)" if fit["at_grid_edge"] else ""))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        format_version=LLPR_FORMAT_VERSION,
        covariance=cov,
        eigenvalues=eigen.values,
        eigenvectors=eigen.vectors,
        weights=spec.weights,
        varsigma2=fit["varsigma2"],
        alpha2=fit["alpha2"],
        calibration_nll=fit["nll"],
        grid_varsigma2=fit["grid_varsigma2"],
        grid_nll=fit["grid_nll"],
        grid_alpha2=fit["grid_alpha2"],
        energy_weight=energy_weight,
        train_frame_indices=indices,
        n_train_frames_total=n_total,
        head=np.asarray(spec.head),
        feature_blocks=np.asarray([f"{b['readout']}:{b['kind']}:{len(b['indices'])}" for b in spec.blocks]),
        model_path=np.asarray(str(model_path)),
        max_abs_force_reconstruction_error=worst_force,
        max_abs_energy_reconstruction_error=worst_energy,
    )
    return {
        "covariance_file": str(out_path),
        "head": spec.head,
        "feature_dim": spec.dim,
        "feature_blocks": [f"readout {b['readout']} {b['kind']} ({len(b['indices'])})" for b in spec.blocks],
        "n_train_frames": int(len(indices)),
        "n_train_frames_total": int(n_total),
        "energy_weight": energy_weight,
        "varsigma2": fit["varsigma2"],
        "alpha2": fit["alpha2"],
        "calibration_nll": fit["nll"],
        "calibration_components": fit["n_components"],
        "varsigma2_at_grid_edge": bool(fit["at_grid_edge"]),
        "max_abs_force_reconstruction_error": worst_force,
        "max_abs_energy_reconstruction_error": worst_energy,
    }


def predict(model_path: str | Path, head: str | None, cov_npz: str | Path,
            frames: Sequence[Atoms], device: str = "cpu", dtype: str = "float64"
            ) -> list[dict[str, np.ndarray]]:
    """Per frame: model energy/forces and LLPR sigmas (forces (N,3), energy)."""
    stored = np.load(cov_npz)
    eigen = Eigen(stored["eigenvalues"], stored["eigenvectors"])
    varsigma2 = float(stored["varsigma2"])
    alpha2 = float(stored["alpha2"])
    model = load_model(model_path, device, dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = last_layer(model, head)
    if not np.allclose(spec.weights, stored["weights"], rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance file was built for different readout weights")
    results = []
    with timing.timed({"kind": "llpr_frames", "phase": "predict", "frames": len(frames),
                       "atoms": int(sum(len(a) for a in frames))}):
        for atoms, data in zip(frames, frames_to_data(model, frames, spec.head)):
            feats = frame_features(model, spec, data, device, jacobian=True)
            u = eigen.quad(feats["J"], varsigma2)
            u_energy = eigen.quad(feats["f"][:, None], varsigma2)[0]
            results.append({
                "energy": feats["energy"],
                "forces": feats["forces"],
                "force_sigma": np.sqrt(alpha2 * u).reshape(len(atoms), 3),
                "energy_sigma": float(np.sqrt(alpha2 * u_energy)),
            })
    return results


def predict_force_sigma(model_path: str | Path, head: str | None, cov_npz: str | Path,
                        frames: Sequence[Atoms], device: str = "cpu",
                        dtype: str = "float64") -> list[np.ndarray]:
    """LLPR force sigmas, one ``(n_atoms, 3)`` array per frame (eV/A)."""
    return [item["force_sigma"] for item in predict(model_path, head, cov_npz, frames, device, dtype)]
