"""Figures for the environment-resolved UQ analysis and the AL loop.

Reads ``uq_analysis.json`` (written by ``uq_analysis.py``) and, optionally, an
``al_manifest.json`` (written by ``uq_active_learning.py``).  The figures are
deliberately small and vector-first: one risk--coverage panel per environment,
one calibration-scale panel, and the label-efficiency curves once the loop has
run.

Usage
-----
::

    python src/mlpdft/utils/make_uq_figures.py \
        --analysis assets/research/PHASE-2/analysis/uq_analysis.json \
        --outdir assets/research/PHASE-2/analysis/figures

    # after the active-learning run
    python src/mlpdft/utils/make_uq_figures.py \
        --analysis .../uq_analysis.json \
        --al-manifest outputs/.../al_manifest.json \
        --outdir .../figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

# Okabe--Ito palette, colour-blind safe.
COLOURS = {
    "committee": "#0072B2",
    "snapshot": "#D55E00",
    "multihead": "#009E73",
    "random": "#999999",
}
LABELS = {
    "committee": "Independent committee",
    "snapshot": "Warm-restart snapshots",
    "multihead": "Multi-head committee",
    "random": "Random",
}
GROUP_LABELS = {
    "LIF64_KJPAW_FINAL": "Bulk LiF",
    "LIWITHF_KJPAW_FINAL": "F-doped Li",
    "LIFINTERFACE_KJPAW_FINAL": "Interface",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {path.with_suffix('.pdf')}")


def _ordered_methods(methods: dict[str, object]) -> list[str]:
    order = ["committee", "snapshot", "multihead"]
    return [name for name in order if name in methods]


def plot_risk_coverage(report: dict, outdir: Path) -> None:
    methods = _ordered_methods(report["methods"])
    groups = list(next(iter(report["methods"].values()))["groups"])
    fig, axes = plt.subplots(1, len(groups), figsize=(6.6, 2.0), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        for name in methods:
            curve = report["methods"][name]["groups"][group]["risk_coverage"]
            xs = [float(k) for k in curve]
            ys = [curve[k] for k in curve]
            ax.plot(xs, ys, marker="o", ms=2.5, color=COLOURS.get(name), label=LABELS.get(name, name))
        ax.set_title(GROUP_LABELS.get(group, group))
        ax.set_xlabel("Coverage")
        ax.set_xlim(0.05, 1.02)
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("Force RMSE (eV/Å)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, outdir / "fig_risk_coverage")


def plot_scale(report: dict, outdir: Path) -> None:
    methods = _ordered_methods(report["methods"])
    groups = list(next(iter(report["methods"].values()))["groups"])
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for index, name in enumerate(methods):
        values = [
            report["methods"][name]["groups"][group]["median_error_over_sigma"]
            or np.nan
            for group in groups
        ]
        ax.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            color=COLOURS.get(name),
            label=LABELS.get(name, name),
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [GROUP_LABELS.get(g, g) for g in groups])
    ax.set_ylabel("Median error / spread")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    _save(fig, outdir / "fig_error_over_sigma")


def plot_al_curves(manifest_path: Path, outdir: Path) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    results = manifest.get("results", {})
    if not results:
        raise ValueError(f"{manifest_path} contains no results")

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for name, payload in results.items():
        records = payload.get("records", [])
        if not records:
            continue
        labels = [record["labeled_count"] for record in records]
        rmse = [record["test_force_rmse_ev_per_angstrom"] for record in records]
        ax.plot(
            labels,
            rmse,
            marker="o",
            ms=2.5,
            color=COLOURS.get(name),
            label=LABELS.get(name, name),
        )
    target = manifest.get("target_rmse")
    if target is not None:
        ax.axhline(target, color="black", linewidth=0.6, linestyle="--")
        ax.annotate(
            "1.05 × full-pool RMSE",
            xy=(0.02, target),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=6,
        )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Acquired labels")
    ax.set_ylabel("Test force RMSE (eV/Å)")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, outdir / "fig_al_curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--al-manifest", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    plot_risk_coverage(report, args.outdir)
    plot_scale(report, args.outdir)
    if args.al_manifest is not None:
        plot_al_curves(args.al_manifest, args.outdir)


if __name__ == "__main__":
    main()
