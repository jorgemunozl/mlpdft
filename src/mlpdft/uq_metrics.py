"""Uncertainty metrics for force predictions (NumPy only).

Conventions for one evaluation set:

* ``err``   (n_atoms_total, 3): predicted minus reference force, eV/Å;
* ``sigma`` (n_atoms_total, 3): predicted standard deviation, eV/Å, > 0;
* ``frame`` (n_atoms_total,):  index of the frame each atom belongs to;
* ``block`` (n_frames,):       resampling block of each frame (a replica, or a
  time block of one trajectory) for the bootstrap.

Calibration is fitted on a separate calibration set and applied unchanged to
every test set: a global scale ``alpha`` minimising the Gaussian NLL, and a
split-conformal multiplier ``q`` for a target coverage. Frames, not force
components, are the exchangeable units, so conformal coverage is a
component-level approximation; the block bootstrap carries the dependence.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import erfinv
from scipy.stats import spearmanr


def per_atom(err: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom force-error norm and per-atom uncertainty norm."""
    return np.linalg.norm(err, axis=1), np.sqrt((sigma ** 2).sum(axis=1))


def per_frame(err: np.ndarray, sigma: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame force RMSE and per-frame RMS uncertainty."""
    n = int(frame.max()) + 1
    count = np.bincount(frame, minlength=n) * 3
    rmse = np.sqrt(np.bincount(frame, (err ** 2).sum(axis=1), minlength=n) / count)
    srms = np.sqrt(np.bincount(frame, (sigma ** 2).sum(axis=1), minlength=n) / count)
    return rmse, srms


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def gaussian_nll(err: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood per force component."""
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * (err / sigma) ** 2))


def nll_scale(err: np.ndarray, sigma: np.ndarray) -> float:
    """Global factor alpha minimising the NLL of err under N(0, (alpha sigma)^2)."""
    return float(np.sqrt(np.mean((err / sigma) ** 2)))


def coverage(err: np.ndarray, sigma: np.ndarray, k: float) -> float:
    return float(np.mean(np.abs(err) <= k * sigma))


def miscalibration_area(err: np.ndarray, sigma: np.ndarray, n_levels: int = 99) -> float:
    """Mean |observed - nominal| coverage of central Gaussian intervals over levels in (0, 1)."""
    levels = np.linspace(0.01, 0.99, n_levels)
    z = np.sqrt(2) * erfinv(levels)
    ratio = np.sort(np.abs(err / sigma).ravel())
    observed = np.searchsorted(ratio, z, side="right") / len(ratio)
    return float(np.mean(np.abs(observed - levels)))


def conformal_multiplier(err_cal: np.ndarray, sigma_cal: np.ndarray, target: float = 0.9) -> float:
    """Split-conformal quantile of |err|/sigma on the calibration set."""
    scores = np.sort(np.abs(err_cal / sigma_cal).ravel())
    n = len(scores)
    rank = min(n - 1, int(np.ceil((n + 1) * target)) - 1)
    return float(scores[rank])


def auroc(score_in: np.ndarray, score_out: np.ndarray) -> float:
    """P(score_out > score_in) with ties counted as 1/2 (Mann-Whitney)."""
    scores = np.concatenate([score_in, score_out])
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n_in, n_out = len(score_in), len(score_out)
    u = ranks[n_in:].sum() - n_out * (n_out + 1) / 2
    return float(u / (n_in * n_out))


def block_bootstrap(stat: Callable[[np.ndarray], float], block: np.ndarray, n_boot: int = 1000,
                    seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of ``stat(frame_indices)`` resampling whole blocks with replacement."""
    rng = np.random.default_rng(seed)
    labels = np.unique(block)
    members = {b: np.flatnonzero(block == b) for b in labels}
    values = []
    for _ in range(n_boot):
        pick = rng.choice(labels, size=len(labels), replace=True)
        values.append(stat(np.concatenate([members[b] for b in pick])))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    tail = 100 * (1 - level) / 2
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def _select(err: np.ndarray, sigma: np.ndarray, frame: np.ndarray, frames: np.ndarray):
    """Atoms of the chosen frames, with frames renumbered 0..len(frames)-1 (repeats allowed)."""
    starts = np.searchsorted(frame, np.arange(frame.max() + 2))
    pieces = [np.arange(starts[f], starts[f + 1]) for f in frames]
    idx = np.concatenate(pieces)
    new_frame = np.repeat(np.arange(len(frames)), [len(p) for p in pieces])
    return err[idx], sigma[idx], new_frame


def summarize(err: np.ndarray, sigma: np.ndarray, frame: np.ndarray, block: np.ndarray, *,
              alpha: float | None = None, conformal_q: float | None = None, target: float = 0.9,
              n_boot: int = 1000) -> dict:
    """Accuracy, ranking and calibration of one set, with block-bootstrap intervals.

    ``frame`` must be sorted (atoms grouped by frame), as produced by the evaluator.
    """
    if np.any(np.diff(frame) < 0):
        raise ValueError("atoms must be grouped by frame in increasing order")
    out: dict = {"n_frames": int(frame.max()) + 1, "n_atoms": int(len(frame)),
                 "n_blocks": int(len(np.unique(block)))}

    def metrics(e, s, fr) -> dict:
        ae, an = per_atom(e, s)
        fe, fs = per_frame(e, s, fr)
        row = {"force_rmse": float(np.sqrt(np.mean(e ** 2))), "force_mae": float(np.mean(np.abs(e))),
               "spearman_atom": spearman(ae, an), "spearman_frame": spearman(fe, fs),
               "nll_raw": gaussian_nll(e, s), "miscal_area_raw": miscalibration_area(e, s),
               "coverage_1sigma_raw": coverage(e, s, 1.0), "coverage_2sigma_raw": coverage(e, s, 2.0)}
        if alpha is not None:
            row.update({"nll_cal": gaussian_nll(e, alpha * s), "miscal_area_cal": miscalibration_area(e, alpha * s),
                        "coverage_1sigma_cal": coverage(e, alpha * s, 1.0),
                        "coverage_2sigma_cal": coverage(e, alpha * s, 2.0)})
        if conformal_q is not None:
            row[f"conformal_coverage_{int(round(100 * target))}"] = coverage(e, s, conformal_q)
        return row

    out["point"] = metrics(err, sigma, frame)
    keys = list(out["point"])
    rng = np.random.default_rng(0)
    labels = np.unique(block)
    members = {b: np.flatnonzero(block == b) for b in labels}
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(labels, size=len(labels), replace=True)
        resampled = metrics(*_select(err, sigma, frame, np.concatenate([members[b] for b in pick])))
        for k in keys:
            samples[k].append(resampled[k])
    out["ci95"] = {}
    for k in keys:
        v = np.asarray(samples[k], dtype=float)
        v = v[np.isfinite(v)]
        out["ci95"][k] = ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                          if len(v) else [float("nan"), float("nan")])
    return out


def shift_auroc(sigma_in: np.ndarray, frame_in: np.ndarray, sigma_out: np.ndarray, frame_out: np.ndarray) -> float:
    """AUROC of the per-frame RMS uncertainty for separating a shifted set from the in-distribution set."""
    _, s_in = per_frame(np.zeros_like(sigma_in), sigma_in, frame_in)
    _, s_out = per_frame(np.zeros_like(sigma_out), sigma_out, frame_out)
    return auroc(s_in, s_out)
