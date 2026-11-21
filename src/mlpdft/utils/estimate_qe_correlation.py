#!/usr/bin/env python3
"""Estimate the correlation time of a QE MD run from its ``.out`` file.

The frame stride used when the extxyz files were written is a bookkeeping
choice. This tool works on the origin: every SCF block in the QE output, one
per ionic step, and returns the integrated autocorrelation time of the total
energy (of the cell volume when it varies, and of the atomic forces with
``--forces``).

The integrated autocorrelation time is a decorrelation time only for a
stationary, mixing series that is long compared with it. The tool therefore
checks those assumptions and lists every failure under ``warnings``:

* the energy drifts (non-stationary; tau is not defined);
* the Sokal window hit its n/4 cap (tau is a lower bound);
* the series is oscillatory: one spectral line carries most of the variance,
  so positive and negative ACF lobes cancel and tau_int looks short;
* the run is shorter than 50 tau (Madras-Sokal relative error is reported);
* forces decorrelate more slowly than energy and volume, or keep a coherent
  oscillation long after tau.

``recommended_stride`` is still reported (ADR-0003) but is meaningful only when
``stride_reliable`` is true. See ``assets/research/methods/autocorrelation.md``.

Usage
-----
    PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
        --out dataset/LIWITHF_V3/LIWITHF_V3.out \
        --burn-in 2400 --label F_doped_Li --forces --json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RY_TO_EV = 13.605693122994
BOHR3_TO_A3 = 0.148184711

ENERGY_RE = re.compile(r"=\s*(-?\d+\.\d+)\s+Ry")
VOLUME_RE = re.compile(r"unit-cell volume\s*=\s*(-?\d+\.\d+)\s+\(a\.u\.\)\^3")


def parse_steps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (energy_eV, volume_A3) per converged SCF block."""
    energies: list[float] = []
    volumes: list[float] = []
    energy: float | None = None
    volume = float("nan")
    with path.open(errors="ignore") as handle:
        for line in handle:
            if "!    total energy" in line:
                match = ENERGY_RE.search(line)
                if match:
                    energy = float(match.group(1)) * RY_TO_EV
            elif "unit-cell volume" in line:
                match = VOLUME_RE.search(line)
                if match:
                    volume = float(match.group(1)) * BOHR3_TO_A3
            elif "convergence has been achieved" in line:
                if energy is not None:
                    energies.append(energy)
                    volumes.append(volume)
                energy = None
    if not energies:
        raise SystemExit(f"no SCF blocks found in {path}")
    return np.asarray(energies), np.asarray(volumes)


def parse_forces(path: Path) -> np.ndarray:
    """Return total atomic forces, shape (steps, atoms, 3), one frame per SCF block.

    Only the first table after each ``Forces acting on atoms`` header is read;
    the per-term decompositions printed at high verbosity are skipped.
    """
    frames: list[list[list[float]]] = []
    block: list[list[float]] | None = None
    with path.open(errors="ignore") as handle:
        for line in handle:
            if block is not None:
                if "force =" in line:
                    block.append([float(v) for v in line.split("=", 1)[1].split()[:3]])
                elif block and not line.strip():
                    frames.append(block)
                    block = None
                continue
            if "Forces acting on atoms" in line:
                block = []
    if not frames:
        raise SystemExit(f"no force tables found in {path}")
    return np.asarray(frames)


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized autocovariance of a 1-D series, via FFT."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = float(np.mean(x * x))
    if n < 8 or var <= 1e-30:
        return np.zeros(1)
    length = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(x, n=length)
    acf = np.fft.irfft(fft.conj() * fft, n=length)[:n]
    return acf / (np.arange(n, 0, -1) * var)


def integrated_tau(acf: np.ndarray, c: float = 5.0) -> dict:
    """Sokal automatic-windowing estimate of the integrated autocorrelation time."""
    n = len(acf)
    if n < 8:
        return {"tau": None, "window": None, "capped": True}
    max_window = max(1, n // 4)
    tau = 0.5
    window = 1
    for w in range(1, max_window + 1):
        tau = 0.5 + float(np.sum(acf[1 : w + 1]))
        if w >= c * tau:
            window = w
            break
    else:
        window = max_window
    return {
        "tau": max(tau, 0.5),
        "window": window,
        "capped": window >= max_window,
    }


def decay_lag(acf: np.ndarray, threshold: float) -> int | None:
    below = np.where(acf <= threshold)[0]
    return int(below[0]) if len(below) else None


def envelope_lag(acf: np.ndarray, threshold: float = 0.1) -> int | None:
    """First lag after which |acf| stays below ``threshold`` up to n/4."""
    head = np.abs(acf[: max(2, len(acf) // 4)])
    tail_max = np.maximum.accumulate(head[::-1])[::-1]
    below = np.where(tail_max < threshold)[0]
    return int(below[0]) if len(below) else None


def periodicity(x: np.ndarray, min_cycles: int = 10) -> dict:
    """Dominant period (at least ``min_cycles`` cycles in the series) and the
    share of the variance within +-20% of its frequency."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x))
    total = float(power[1:].sum())
    if len(power) <= min_cycles + 1 or total <= 0:
        return {"period_steps": None, "peak_share": 0.0}
    k = int(np.argmax(power[min_cycles:])) + min_cycles
    band = (freq > 0.8 * freq[k]) & (freq < 1.2 * freq[k])
    return {"period_steps": float(1 / freq[k]), "peak_share": float(power[band].sum() / total)}


def madras_sokal_rel_err(window: int | None, n: int) -> float | None:
    """Relative standard error of tau_int for a window W: sqrt(2(2W+1)/n)."""
    if not window or n <= 0:
        return None
    return float(np.sqrt(2 * (2 * window + 1) / n))


def force_autocorrelation(forces: np.ndarray) -> np.ndarray:
    """ACF of the force field, pooled over atoms and Cartesian components.

    The per-atom mean force, zero in equilibrium, is removed first.
    """
    x = forces - forces.mean(axis=0)
    x = x.reshape(len(x), -1)
    n = len(x)
    length = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(x, n=length, axis=0)
    acf = np.fft.irfft((fft.conj() * fft).sum(axis=1), n=length)[:n]
    acf = acf / np.arange(n, 0, -1)
    return acf / acf[0]


def analyse(
    energies: np.ndarray,
    volumes: np.ndarray,
    burn_in: int,
    forces: np.ndarray | None = None,
) -> dict:
    energies = energies[burn_in:]
    volumes = volumes[burn_in:]
    n = len(energies)
    result: dict = {"steps_after_burn_in": n}
    warnings: list[str] = []

    series = {"energy": energies}
    if np.isfinite(volumes).all() and np.std(volumes) > 1e-6:
        series["volume"] = volumes

    for name, values in series.items():
        acf = autocorrelation(values)
        tau = integrated_tau(acf)
        # A linear drift inflates the ACF of a nonstationary series. Estimate
        # the fluctuation correlation on the detrended copy as well.
        detrended = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
        acf_d = autocorrelation(detrended)
        tau_d = integrated_tau(acf_d)
        period = periodicity(detrended)
        n_over_tau = n / tau_d["tau"] if tau_d["tau"] else None
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "tau_steps": tau["tau"],
            "window": tau["window"],
            "window_capped": tau["capped"],
            "tau_rel_err": madras_sokal_rel_err(tau["window"], n),
            "lag_acf_0.05": decay_lag(acf, 0.05),
            "lag_acf_1_over_e": decay_lag(acf, 1 / np.e),
            "tau_detrended_steps": tau_d["tau"],
            "tau_detrended_window": tau_d["window"],
            "tau_detrended_window_capped": tau_d["capped"],
            "tau_detrended_rel_err": madras_sokal_rel_err(tau_d["window"], n),
            "n_over_tau_detrended": n_over_tau,
            "envelope_lag_0.1_detrended": envelope_lag(acf_d),
            "acf_detrended_at_lag": {
                str(k): round(float(acf_d[k]), 3) for k in (30, 90, 200) if k < len(acf_d)
            },
            "dominant_period_steps": period["period_steps"],
            "dominant_period_variance_share": period["peak_share"],
            "tau_2sigma": (2 * tau["tau"] if tau["tau"] else None),
        }
        if tau_d["capped"]:
            warnings.append(f"{name}: Sokal window capped at n/4; tau is a lower bound")
        if period["peak_share"] > 0.5:
            warnings.append(
                f"{name}: oscillatory ({period['peak_share']:.0%} of the variance near a "
                f"{period['period_steps']:.0f}-step period); tau_int is short because "
                "positive and negative ACF lobes cancel"
            )
        if n_over_tau is not None and n_over_tau < 50:
            warnings.append(
                f"{name}: run is {n_over_tau:.0f} tau long (< 50); relative error on tau "
                f"~{result[name]['tau_detrended_rel_err']:.0%}"
            )

    drift = float(np.polyfit(np.arange(n), energies, 1)[0]) if n > 2 else 0.0
    half = n // 2
    result["drift_ev_per_1000_steps"] = drift * 1000
    result["mean_first_half_ev"] = float(np.mean(energies[:half]))
    result["mean_second_half_ev"] = float(np.mean(energies[half:]))

    slowest = max(
        result[name]["tau_detrended_steps"] for name in series
    )
    result["slowest_detrended_tau_steps"] = slowest
    result["recommended_stride"] = (
        max(1, int(np.ceil(2 * slowest))) if slowest else None
    )
    result["drift_warning"] = abs(drift) * n > 0.5 * float(np.std(energies))
    result["independent_samples"] = (
        n / result["recommended_stride"] if result["recommended_stride"] else None
    )
    if result["drift_warning"]:
        warnings.insert(0, "energy drifts: the series is not stationary, so tau is not defined")

    if forces is not None:
        frames = forces[burn_in:]
        acf_f = force_autocorrelation(frames)
        tau_f = integrated_tau(acf_f)
        envelope = envelope_lag(acf_f)
        result["forces"] = {
            "frames": len(frames),
            "tau_steps": tau_f["tau"],
            "window": tau_f["window"],
            "window_capped": tau_f["capped"],
            "tau_rel_err": madras_sokal_rel_err(tau_f["window"], len(frames)),
            "envelope_lag_0.1": envelope,
            "acf_at_lag": {
                str(k): round(float(acf_f[k]), 3) for k in (10, 30, 100, 300) if k < len(acf_f)
            },
        }
        if tau_f["capped"]:
            warnings.append("forces: Sokal window capped at n/4; tau is a lower bound")
        if slowest and tau_f["tau"] > slowest:
            warnings.append(
                f"forces: tau ({tau_f['tau']:.0f} steps) exceeds the energy/volume tau "
                f"({slowest:.0f}); the energy proxy understates the correlation"
            )
        if envelope is None:
            warnings.append("forces: |ACF| does not stay below 0.1 within n/4")
        elif envelope > 10 * tau_f["tau"]:
            warnings.append(
                f"forces: |ACF| stays above 0.1 until lag {envelope} (> 10 tau); "
                "a coherent oscillation persists"
            )

    result["warnings"] = warnings
    result["stride_reliable"] = not warnings
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--burn-in", type=int, default=0,
                        help="MD steps to drop before estimating correlation")
    parser.add_argument("--label", default=None)
    parser.add_argument("--forces", action="store_true",
                        help="also estimate the correlation of the atomic forces")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    energies, volumes = parse_steps(args.out)
    forces = None
    if args.forces:
        forces = parse_forces(args.out)
        # A run cut between an SCF block and its force table leaves one extra energy.
        n = min(len(energies), len(forces))
        energies, volumes, forces = energies[:n], volumes[:n], forces[:n]
    result = analyse(energies, volumes, args.burn_in, forces)
    result["out"] = str(args.out)
    result["label"] = args.label or args.out.stem
    result["burn_in_steps"] = args.burn_in
    result["total_steps"] = len(energies)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{result['label']}: {result['total_steps']} steps, burn-in "
            f"{args.burn_in}, tau={result['energy']['tau_steps']:.1f} steps, "
            f"recommended stride={result['recommended_stride']}, "
            f"independent samples~{result['independent_samples']:.0f}, "
            f"stride reliable={result['stride_reliable']}"
        )
        for warning in result["warnings"]:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
