#!/usr/bin/env python3
"""Estimate the correlation time of a QE MD run from its ``.out`` file.

The frame stride used when the extxyz files were written is a bookkeeping
choice. This tool works on the origin: every SCF block in the QE output, one
per ionic step, and returns the integrated autocorrelation time of the total
energy (and of the cell volume when it varies). The recommended stride is the
smallest spacing at which consecutive selected frames are approximately
independent.

Usage
-----
    python3 src/mlpdft/utils/estimate_qe_correlation.py \
        --out dataset/LIWITHF_V3/LIWITHF_V3.out \
        --burn-in 2400 --label F_doped_Li --json
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


def analyse(
    energies: np.ndarray, volumes: np.ndarray, burn_in: int
) -> dict:
    energies = energies[burn_in:]
    volumes = volumes[burn_in:]
    n = len(energies)
    result: dict = {"steps_after_burn_in": n}

    series = {"energy": energies}
    if np.isfinite(volumes).all() and np.std(volumes) > 1e-6:
        series["volume"] = volumes

    for name, values in series.items():
        acf = autocorrelation(values)
        tau = integrated_tau(acf)
        # A linear drift inflates the ACF of a nonstationary series. Estimate
        # the fluctuation correlation on the detrended copy as well.
        detrended = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
        tau_d = integrated_tau(autocorrelation(detrended))
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "tau_steps": tau["tau"],
            "window": tau["window"],
            "window_capped": tau["capped"],
            "lag_acf_0.05": decay_lag(acf, 0.05),
            "lag_acf_1_over_e": decay_lag(acf, 1 / np.e),
            "tau_detrended_steps": tau_d["tau"],
            "tau_detrended_window_capped": tau_d["capped"],
            "tau_2sigma": (2 * tau["tau"] if tau["tau"] else None),
        }

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
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--burn-in", type=int, default=0,
                        help="MD steps to drop before estimating correlation")
    parser.add_argument("--label", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    energies, volumes = parse_steps(args.out)
    result = analyse(energies, volumes, args.burn_in)
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
            f"independent samples~{result['independent_samples']:.0f}"
        )


if __name__ == "__main__":
    main()
