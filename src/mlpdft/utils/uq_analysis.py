"""Environment-resolved UQ analysis with paired block bootstrap.

This module consumes the per-frame prediction CSVs written by
``mlpdft.uq_compare`` and answers the questions the per-method summary JSON
cannot answer on its own:

* Is the uncertainty--error ranking positive *within* each environment, and
  how wide is that correlation once temporal correlation is accounted for?
* Do two methods differ by more than the shared sampling noise on the same
  frames?  A paired bootstrap answers this; comparing two marginal confidence
  intervals does not.
* What does the risk--coverage curve look like inside one environment, rather
  than pooled over environments of different difficulty?
* Is the ensemble spread calibrated in the coverage sense, at one and two
  sigma?

The statistical unit is the contiguous time block, not the frame.  Frames from
one AIMD trajectory are strongly autocorrelated, so an i.i.d. bootstrap over
frames understates every interval.  Groups with fewer frames than one block are
reported without an interval instead of being given a misleading one.

Usage
-----
::

    python src/mlpdft/utils/uq_analysis.py \
        --prediction committee=path/to/independent_committee_predictions.csv \
        --prediction snapshot=path/to/warm_snapshots_predictions.csv \
        --output-dir assets/research/PHASE-2/analysis \
        --block 20 --n-boot 2000

The output is ``uq_analysis.json`` plus a human-readable
``uq_analysis.md``.  Prediction files must share the same test frames; the
loader asserts that the frame sets agree before any paired comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = (
    "frame_index",
    "group",
    "n_atoms",
    "reference_energy_ev",
    "predicted_energy_mean_ev",
    "energy_abs_error_mev_per_atom",
    "force_rmse_ev_per_angstrom",
    "force_uncertainty_ev_per_angstrom",
)


@dataclass(frozen=True)
class Predictions:
    """Per-frame predictions for one method on one locked test set."""

    name: str
    path: Path
    frame_index: np.ndarray
    group: np.ndarray
    n_atoms: np.ndarray
    energy_error_mev_per_atom: np.ndarray
    force_rmse: np.ndarray
    force_uncertainty: np.ndarray

    def __len__(self) -> int:
        return int(self.frame_index.size)


def load_predictions(name: str, path: Path) -> Predictions:
    """Read one ``*_predictions.csv`` file and validate its columns."""

    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")

    frame_index = np.array([int(r["frame_index"]) for r in rows], dtype=int)
    n_atoms = np.array([int(r["n_atoms"]) for r in rows], dtype=float)
    force_rmse = np.array(
        [float(r["force_rmse_ev_per_angstrom"]) for r in rows], dtype=float
    )
    force_unc = np.array(
        [float(r["force_uncertainty_ev_per_angstrom"]) for r in rows], dtype=float
    )
    # Signed energy error: the CSV stores the absolute value, but the reference
    # and mean prediction are both present.
    reference = np.array([float(r["reference_energy_ev"]) for r in rows], dtype=float)
    predicted = np.array(
        [float(r["predicted_energy_mean_ev"]) for r in rows], dtype=float
    )
    energy_error = (predicted - reference) / n_atoms * 1000.0

    if not np.isfinite(force_rmse).all() or not np.isfinite(force_unc).all():
        raise ValueError(f"{path} contains non-finite force values")
    if np.any(n_atoms <= 0):
        raise ValueError(f"{path} contains a non-positive atom count")

    return Predictions(
        name=name,
        path=path,
        frame_index=frame_index,
        group=np.array([str(r["group"]) for r in rows], dtype=object),
        n_atoms=n_atoms,
        energy_error_mev_per_atom=energy_error,
        force_rmse=force_rmse,
        force_uncertainty=force_unc,
    )


def assert_same_frames(reference: Predictions, other: Predictions) -> None:
    """Require two prediction sets to cover exactly the same frames."""

    left = set(zip(reference.group.tolist(), reference.frame_index.tolist()))
    right = set(zip(other.group.tolist(), other.frame_index.tolist()))
    if left != right:
        only_left = sorted(left - right)[:5]
        only_right = sorted(right - left)[:5]
        raise ValueError(
            f"{other.name} does not cover the same frames as {reference.name}: "
            f"only in {reference.name}={only_left}, only in {other.name}={only_right}"
        )


# ---------------------------------------------------------------------------
# Point statistics
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved to their average rank (no scipy dependency)."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((x * y).sum() / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN when either vector is constant."""

    if x.size < 3:
        return float("nan")
    return pearson(_average_ranks(x), _average_ranks(y))


def atom_weighted_rmse(frame_rmse: np.ndarray, n_atoms: np.ndarray) -> float:
    """Combine per-frame force RMSEs into the atom-weighted global RMSE."""

    weights = n_atoms / n_atoms.sum()
    return float(math.sqrt(float((weights * frame_rmse**2).sum())))


def atom_weighted_mean(values: np.ndarray, n_atoms: np.ndarray) -> float:
    return float((values * n_atoms).sum() / n_atoms.sum())


def energy_rmse(errors_mev_per_atom: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(errors_mev_per_atom**2))))


def empirical_coverage(
    errors: np.ndarray, sigma: np.ndarray, level: float
) -> float | None:
    """Fraction of frames with ``|error| <= level * sigma``.

    The force error is a per-frame RMS over atoms and ``sigma`` is a per-atom
    standard deviation, so this is a Gaussian-style reference rather than a
    strict predictive interval.  It is reported for comparability, not as a
    formal calibration test.
    """

    if sigma.size == 0 or np.any(sigma <= 0):
        return None
    return float(np.mean(np.abs(errors) <= level * sigma))


def _error_over_sigma(
    errors: np.ndarray, sigma: np.ndarray
) -> float | None:
    """Median ratio of frame error to predicted spread; >1 means under-scaled."""

    mask = sigma > 0
    if not np.any(mask):
        return None
    return float(np.median(errors[mask] / sigma[mask]))


def risk_coverage(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    n_atoms: np.ndarray,
    fractions: Sequence[float],
) -> dict[str, float]:
    """Mean error of the most-certain ``fraction`` of frames."""

    order = np.argsort(uncertainty, kind="mergesort")
    out: dict[str, float] = {}
    for fraction in fractions:
        count = max(1, int(round(fraction * len(order))))
        selected = order[:count]
        out[f"{fraction:.2f}"] = atom_weighted_mean(
            errors[selected], n_atoms[selected]
        )
    return out


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap sample of ``n`` in-range indices."""

    block = min(block, n)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _group_order(
    group: np.ndarray, frame_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Indices of each group, sorted by trajectory time index."""

    by_group: dict[str, list[int]] = defaultdict(list)
    for position, label in enumerate(group):
        by_group[str(label)].append(position)
    return {
        label: np.array(sorted(positions, key=lambda i: frame_index[i]))
        for label, positions in by_group.items()
    }


def _within_group_spearman_mean(
    x: np.ndarray, y: np.ndarray, by_group: dict[str, np.ndarray]
) -> float:
    values = [spearman(x[idx], y[idx]) for idx in by_group.values()]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_within_group(
    predictions: Predictions,
    block: int,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap the within-group mean Spearman by resampling blocks."""

    by_group = _group_order(predictions.group, predictions.frame_index)
    point = _within_group_spearman_mean(
        predictions.force_uncertainty, predictions.force_rmse, by_group
    )

    eligible = [idx for idx in by_group.values() if idx.size > block]
    if not eligible:
        return {
            "estimate": point,
            "ci_low": None,
            "ci_high": None,
            "n_boot": 0,
            "note": f"no group has more than {block} frames",
        }

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        per_group = []
        for label, idx in by_group.items():
            if idx.size <= block:
                value = spearman(
                    predictions.force_uncertainty[idx],
                    predictions.force_rmse[idx],
                )
            else:
                resampled = idx[block_indices(idx.size, block, rng)]
                value = spearman(
                    predictions.force_uncertainty[resampled],
                    predictions.force_rmse[resampled],
                )
            if np.isfinite(value):
                per_group.append(value)
        samples[i] = float(np.mean(per_group)) if per_group else np.nan

    samples = samples[np.isfinite(samples)]
    if samples.size < 0.9 * n_boot:
        return {
            "estimate": point,
            "ci_low": None,
            "ci_high": None,
            "n_boot": int(samples.size),
            "note": "too many undefined bootstrap replicates",
        }
    return {
        "estimate": point,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "n_boot": int(samples.size),
    }


def paired_bootstrap(
    reference: Predictions,
    other: Predictions,
    block: int,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Paired block bootstrap of the within-group mean Spearman difference.

    The same resampled blocks are used for both methods, so the interval is on
    the difference and not on two independently resampled correlations.  Groups
    with no more frames than one block are excluded from the interval: they
    cannot be resampled, and including them as a fixed offset would understate
    the interval width.
    """

    assert_same_frames(reference, other)
    by_group = _group_order(reference.group, reference.frame_index)
    eligible = {
        label: idx for label, idx in by_group.items() if idx.size > block
    }
    excluded = sorted(set(by_group) - set(eligible))

    per_group: dict[str, float | None] = {}
    for label, idx in by_group.items():
        a = spearman(reference.force_uncertainty[idx], reference.force_rmse[idx])
        b = spearman(other.force_uncertainty[idx], other.force_rmse[idx])
        per_group[label] = float(a - b) if np.isfinite(a) and np.isfinite(b) else None

    def delta_for(index_map: dict[str, np.ndarray]) -> float:
        deltas = []
        for label, idx in index_map.items():
            a = spearman(
                reference.force_uncertainty[idx], reference.force_rmse[idx]
            )
            b = spearman(other.force_uncertainty[idx], other.force_rmse[idx])
            if np.isfinite(a) and np.isfinite(b):
                deltas.append(a - b)
        return float(np.mean(deltas)) if deltas else float("nan")

    if not eligible:
        return {
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "n_boot": 0,
            "groups_used": [],
            "groups_excluded": excluded,
            "by_group": per_group,
            "note": f"no group has more than {block} frames",
        }

    point = delta_for(eligible)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        resampled = {
            label: idx[block_indices(idx.size, block, rng)]
            for label, idx in eligible.items()
        }
        samples[i] = delta_for(resampled)
    samples = samples[np.isfinite(samples)]
    if samples.size < 0.9 * n_boot:
        return {
            "delta": point,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "n_boot": int(samples.size),
            "groups_used": sorted(eligible),
            "groups_excluded": excluded,
            "by_group": per_group,
            "note": "too many undefined bootstrap replicates",
        }

    # Two-sided p-value: twice the smaller tail mass relative to zero.
    lower = float(np.mean(samples <= 0.0))
    upper = float(np.mean(samples >= 0.0))
    p_value = min(1.0, 2.0 * min(lower, upper))
    return {
        "delta": point,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "p_value": p_value,
        "n_boot": int(samples.size),
        "groups_used": sorted(eligible),
        "groups_excluded": excluded,
        "by_group": per_group,
    }


# ---------------------------------------------------------------------------
# Per-method summary
# ---------------------------------------------------------------------------

RISK_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def summarise(
    predictions: Predictions, block: int, n_boot: int, seed: int
) -> dict[str, object]:
    by_group = _group_order(predictions.group, predictions.frame_index)
    groups: dict[str, object] = {}

    for label, idx in by_group.items():
        uncertainty = predictions.force_uncertainty[idx]
        force_rmse = predictions.force_rmse[idx]
        n_atoms = predictions.n_atoms[idx]
        rho = spearman(uncertainty, force_rmse)

        if idx.size > block:
            rng = np.random.default_rng(seed)
            samples = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                resampled = idx[block_indices(idx.size, block, rng)]
                samples[i] = spearman(
                    predictions.force_uncertainty[resampled],
                    predictions.force_rmse[resampled],
                )
            samples = samples[np.isfinite(samples)]
            ci_low = float(np.percentile(samples, 2.5)) if samples.size else None
            ci_high = float(np.percentile(samples, 97.5)) if samples.size else None
        else:
            ci_low = ci_high = None

        groups[label] = {
            "n_frames": int(idx.size),
            "force_rmse_atom_weighted": atom_weighted_rmse(force_rmse, n_atoms),
            "force_rmse_frame_weighted": float(np.sqrt(np.mean(force_rmse**2))),
            "energy_rmse_mev_per_atom": energy_rmse(
                predictions.energy_error_mev_per_atom[idx]
            ),
            "spearman": rho,
            "spearman_ci_low": ci_low,
            "spearman_ci_high": ci_high,
            "coverage_1sigma": empirical_coverage(
                force_rmse, uncertainty, 1.0
            ),
            "coverage_2sigma": empirical_coverage(
                force_rmse, uncertainty, 2.0
            ),
            "median_error_over_sigma": _error_over_sigma(force_rmse, uncertainty),
            "risk_coverage": risk_coverage(
                force_rmse, uncertainty, n_atoms, RISK_FRACTIONS
            ),
        }

    bootstrap = bootstrap_within_group(predictions, block, n_boot, seed)
    group_rhos = [
        float(g["spearman"])  # type: ignore[arg-type]
        for g in groups.values()
        if np.isfinite(float(g["spearman"]))  # type: ignore[arg-type]
    ]

    return {
        "prediction_file": str(predictions.path),
        "n_frames": len(predictions),
        "overall": {
            "force_rmse_atom_weighted": atom_weighted_rmse(
                predictions.force_rmse, predictions.n_atoms
            ),
            "force_rmse_frame_weighted": float(
                np.sqrt(np.mean(predictions.force_rmse**2))
            ),
            "energy_rmse_mev_per_atom": energy_rmse(
                predictions.energy_error_mev_per_atom
            ),
            "spearman_pooled": spearman(
                predictions.force_uncertainty, predictions.force_rmse
            ),
            "spearman_within_group_mean": bootstrap["estimate"],
            "coverage_1sigma": empirical_coverage(
                predictions.force_rmse, predictions.force_uncertainty, 1.0
            ),
            "coverage_2sigma": empirical_coverage(
                predictions.force_rmse, predictions.force_uncertainty, 2.0
            ),
            "median_error_over_sigma": _error_over_sigma(
                predictions.force_rmse, predictions.force_uncertainty
            ),
            "n_groups": len(groups),
            "n_positive_groups": int(sum(r > 0 for r in group_rhos)),
        },
        "groups": groups,
        "bootstrap": bootstrap,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "predictions must be given as NAME=PATH"
        )
    name, path = value.split("=", 1)
    return name, Path(path)


def _format_interval(low: object, high: object) -> str:
    if low is None or high is None:
        return "n/a"
    return f"[{float(low):+.2f}, {float(high):+.2f}]"  # type: ignore[arg-type]


def write_markdown(report: dict[str, object], path: Path) -> None:
    meta = report["meta"]
    lines = [
        "# UQ analysis",
        "",
        f"Block bootstrap: {meta['block']} frames, "  # type: ignore[index]
        f"{meta['n_boot']} replicates, seed "  # type: ignore[index]
        f"{meta['seed']}.",  # type: ignore[index]
        "",
    ]
    for name, method in report["methods"].items():  # type: ignore[union-attr]
        overall = method["overall"]
        bootstrap = method["bootstrap"]
        mean_interval = _format_interval(
            bootstrap["ci_low"], bootstrap["ci_high"]
        )
        lines += [
            f"## {name}",
            "",
            f"- atom-weighted force RMSE: "
            f"{overall['force_rmse_atom_weighted']:.4f} eV/Å",
            f"- energy RMSE: {overall['energy_rmse_mev_per_atom']:.3f} meV/atom",
            f"- pooled Spearman: {_signed(overall['spearman_pooled'])}; "
            f"within-group mean {_signed(overall['spearman_within_group_mean'])} "
            f"{mean_interval}",
            f"- groups with positive rank correlation: "
            f"{overall['n_positive_groups']}/{overall['n_groups']}",
            "",
            "| group | n | force RMSE | Spearman | 95% CI | e/σ |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for label, group in method["groups"].items():
            lines.append(
                f"| {label} | {group['n_frames']} | "
                f"{group['force_rmse_atom_weighted']:.4f} | "
                f"{_signed(group['spearman'])} | "
                f"{_format_interval(group['spearman_ci_low'], group['spearman_ci_high'])} | "
                f"{_ratio(group['median_error_over_sigma'])} |"
            )
        lines.append("")

    if report.get("paired"):
        lines += [
            "## Paired differences",
            "",
            "Δ is the within-group mean Spearman of the first method minus the "
            "second, bootstrapped on the same blocks.",
            "",
            "| comparison | Δ | 95% CI | p |",
            "|---|---:|---|---:|",
        ]
        for key, paired in report["paired"].items():  # type: ignore[union-attr]
            lines.append(
                f"| {key} | {_signed(paired['delta'])} | "
                f"{_format_interval(paired['ci_low'], paired['ci_high'])} | "
                f"{_p_value(paired['p_value'])} |"
            )
        excluded = sorted(
            {
                group
                for paired in report["paired"].values()  # type: ignore[union-attr]
                for group in paired["groups_excluded"]
            }
        )
        if excluded:
            lines.append(
                "Groups excluded from the paired interval (no more frames than "
                "one block): " + ", ".join(excluded) + "."
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _signed(value: object) -> str:
    if value is None:
        return "n/a"
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.2f}"


def _ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}x"  # type: ignore[arg-type]


def _coverage(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"  # type: ignore[arg-type]


def _p_value(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"  # type: ignore[arg-type]


def _jsonable(value: object) -> object:
    """Replace NaN/inf with null so the report is valid strict JSON."""

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        type=_parse_prediction,
        metavar="NAME=PATH",
        help="prediction CSV; repeat for each method",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    methods = {}
    for name, path in args.prediction:
        methods[name] = load_predictions(name, path)

    reference_name, reference = next(iter(methods.items()))
    for other in methods.values():
        assert_same_frames(reference, other)

    report: dict[str, object] = {
        "meta": {
            "block": args.block,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "reference_method": reference_name,
        },
        "methods": {
            name: summarise(method, args.block, args.n_boot, args.seed)
            for name, method in methods.items()
        },
        "paired": {},
    }

    names = list(methods)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            report["paired"][f"{left} - {right}"] = paired_bootstrap(  # type: ignore[index]
                methods[left], methods[right], args.block, args.n_boot, args.seed
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "uq_analysis.json"
    md_path = args.output_dir / "uq_analysis.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
