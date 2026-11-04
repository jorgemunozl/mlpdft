"""Build the paper's main figures from frozen artifacts.

Four figures, fixed by the figure plan:

1. ``fig1_testbed``      three environments and the pool/split composition
2. ``fig2_discrimination`` within-environment Spearman with block-bootstrap CIs
3. ``fig3_calibration``  risk--coverage per environment and error/spread scale
4. ``fig4_efficiency``   test error against labels and against GPU-hours

Every number is read from a file: the split manifest, ``uq_analysis.json``
(written by ``uq_analysis.py``) and ``al_manifest.json`` (written by
``uq_active_learning.py``). Nothing is hard-coded. Figures are built at the
final printed width (5.5 in, the text width of the single-column Typst paper)
and written as PDF (vector, for the paper) plus SVG and PNG (for preview).

Usage
-----
::

    python src/mlpdft/utils/make_paper_figures.py \
        --split-manifest dataset/xyz_files/uq_splits/split_manifest.json \
        --train-file dataset/xyz_files/uq_splits/uq_train.extxyz \
        --analysis assets/research/PHASE-2/analysis/uq_analysis.json \
        --outdir assets/research/PHASE-2/analysis/figures

    # once the loop has run
    python src/mlpdft/utils/make_paper_figures.py ... \
        --al-manifest outputs/.../al_manifest.json

Use ``--only`` to build a subset while iterating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from figstyle import (  # vendored beside this script
    PALETTE,
    TEXT,
    figure,
    figure_grid,
    panel_labels,
    save,
    use_style,
)

# Environment colours are fixed across every figure, so a reader who learns
# them in Fig. 1 can carry them to Figs. 2--4.
ENV_COLOURS = {
    "bulk_LiF": PALETTE["blue"],
    "F_doped_Li": PALETTE["orange"],
    "interface": PALETTE["green"],
}
ENV_LABELS = {
    "bulk_LiF": "Bulk LiF",
    "F_doped_Li": "F-doped Li",
    "interface": "Interface",
}
ENV_ORDER = ["bulk_LiF", "F_doped_Li", "interface"]

ARM_COLOURS = {
    "committee": PALETTE["blue"],
    "snapshot": PALETTE["orange"],
    "multihead": PALETTE["green"],
}
ARM_LABELS = {
    "committee": "Independent committee",
    "snapshot": "Warm-restart snapshots",
    "multihead": "Multi-head committee",
}
ARM_ORDER = ["committee", "snapshot", "multihead"]

GROUP_LABELS = {
    "LIF64": "Bulk LiF",
    "LIWITHF": "F-doped Li",
    "LIFINTERFACE": "Interface",
}

ELEMENT_COLOURS = {"Li": PALETTE["purple"], "F": PALETTE["green"]}


def group_label(name: str) -> str:
    for key, label in GROUP_LABELS.items():
        if key in name:
            return label
    return name
ELEMENT_SIZES = {"Li": 120, "F": 70}


# ---------------------------------------------------------------------------
# Figure 1 -- testbed
# ---------------------------------------------------------------------------


def _rotation(elev: float, azim: float) -> np.ndarray:
    """Rotation matrix for a simple orthographic view."""

    elev, azim = np.deg2rad(elev), np.deg2rad(azim)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(elev), -np.sin(elev)], [0, np.sin(elev), np.cos(elev)]]
    )
    rz = np.array(
        [[np.cos(azim), -np.sin(azim), 0], [np.sin(azim), np.cos(azim), 0], [0, 0, 1]]
    )
    return rz @ rx


def _draw_structure(ax, atoms, elev: float = 12.0, azim: float = -65.0) -> None:
    """Orthographic ball-and-stick projection, no axes."""

    positions = atoms.positions @ _rotation(elev, azim).T
    species = np.array(atoms.get_chemical_symbols())
    depth = positions[:, 2]
    span = max(depth.max() - depth.min(), 1e-6)
    shade = (depth - depth.min()) / span

    # Li--F contacts only; Li--Li and F--F would turn the bulk cell into a mesh.
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if {species[i], species[j]} != {"Li", "F"}:
                continue
            if np.linalg.norm(atoms.positions[i] - atoms.positions[j]) > 2.45:
                continue
            z = 0.5 * (shade[i] + shade[j])
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color="#9A9A9A",
                linewidth=0.25 + 0.55 * z,
                solid_capstyle="round",
                zorder=1 + z,
            )

    order = np.argsort(depth)
    for index in order:
        element = species[index]
        ax.scatter(
            positions[index, 0],
            positions[index, 1],
            s=ELEMENT_SIZES[element] * (0.75 + 0.35 * shade[index]),
            color=ELEMENT_COLOURS[element],
            edgecolors="#303030",
            linewidths=0.25,
            zorder=2 + shade[index],
        )
    ax.set_aspect("equal")
    ax.axis("off")
    margin = 1.0
    ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)


def _first_frame_per_environment(path: Path):
    """Read only until one frame of each environment has been seen."""

    from ase.io import iread

    wanted = set(ENV_ORDER)
    found = {}
    for atoms in iread(path, format="extxyz"):
        label = str(atoms.info.get("config_type", ""))
        if label in wanted and label not in found:
            found[label] = atoms
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{path} has no frames for {sorted(missing)}")
    return found


def build_testbed(split_manifest: Path, train_file: Path, outdir: Path) -> None:
    counts = json.loads(split_manifest.read_text(encoding="utf-8"))
    splits = {item["split"]: item for item in counts["splits"]}
    frames = _first_frame_per_environment(train_file)

    fig = plt_figure((TEXT, 3.15))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.55, 1.0), hspace=0.05, wspace=0.02)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    views = [(15, 0), (15, 0), (85, 0)]
    for ax, environment, view in zip(axes, ENV_ORDER, views):
        _draw_structure(ax, frames[environment], elev=view[0], azim=view[1])
        ax.text(
            0.5,
            -0.02,
            ENV_LABELS[environment],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
    panel_labels(axes, ["(a)"])

    ax = fig.add_subplot(grid[1, :])
    order = ["train_pool", "calibration", "test", "ood_test"]
    display = ["Pool", "Calibration", "Test", "OOD"]
    y = np.arange(len(order))[::-1]
    total_by_split = []
    for row, (split, label) in enumerate(zip(order, display)):
        item = splits[split]
        total = item["frames"]
        total_by_split.append(total)
        left = 0.0
        for environment in ENV_ORDER:
            value = item["environments"].get(environment, 0)
            fraction = value / total
            ax.barh(
                y[row],
                fraction,
                left=left,
                height=0.62,
                color=ENV_COLOURS[environment],
                edgecolor="white",
                linewidth=0.5,
            )
            if fraction > 0.12:
                ax.text(
                    left + fraction / 2,
                    y[row],
                    f"{100 * fraction:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
            left += fraction
        ax.text(1.01, y[row], f"{total:,}", ha="left", va="center", fontsize=6)
    ax.set_yticks(y, display)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, len(order) - 1 + 1.15)
    ax.set_xlabel("Fraction of split")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.01, y[0] + 0.42, "Frames", ha="left", va="bottom", fontsize=6)
    panel_labels([ax], ["(b)"])

    handles = [
        plt_patch(ENV_COLOURS[env], ENV_LABELS[env]) for env in ENV_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2,
    )
    save(fig, str(outdir / "fig1_testbed"), formats=("pdf", "svg", "png"))


# ---------------------------------------------------------------------------
# Figure 2 -- within-environment discrimination
# ---------------------------------------------------------------------------


def _group_sort_key(label: str) -> int:
    for index, environment in enumerate(ENV_ORDER):
        if environment in label:
            return index
    return 99


def build_discrimination(analysis: Path, outdir: Path) -> None:
    report = json.loads(analysis.read_text(encoding="utf-8"))
    methods = [name for name in ARM_ORDER if name in report["methods"]]
    groups = list(next(iter(report["methods"].values()))["groups"])
    groups = sorted(groups, key=_group_sort_key)

    fig, ax = figure(width=TEXT, ratio=0.58)
    base = np.arange(len(groups) + 1)[::-1]  # pooled at the bottom
    offsets = np.linspace(-0.22, 0.22, len(methods))

    for arm_index, method in enumerate(methods):
        colour = ARM_COLOURS[method]
        for group_index, group in enumerate(groups):
            stats = report["methods"][method]["groups"][group]
            y = base[group_index] + offsets[arm_index]
            rho = stats["spearman"]
            low = stats["spearman_ci_low"]
            high = stats["spearman_ci_high"]
            if low is None or high is None:
                ax.scatter([rho], [y], s=18, color=colour, zorder=4)
            else:
                ax.errorbar(
                    [rho],
                    [y],
                    xerr=[[rho - low], [high - rho]],
                    fmt="o",
                    ms=3.6,
                    color=colour,
                    ecolor=colour,
                    elinewidth=0.9,
                    capsize=1.6,
                    zorder=4,
                )
        pooled = report["methods"][method]["overall"]["spearman_pooled"]
        ax.scatter(
            [pooled],
            [base[-1] + offsets[arm_index]],
            s=22,
            facecolors="none",
            edgecolors=colour,
            linewidths=1.0,
            zorder=4,
        )

    ax.axvline(0.0, color="#7F7F7F", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(base[-1] + 0.5, color="#DDDDDD", linewidth=0.6, zorder=0)

    labels = []
    for group in groups:
        n = report["methods"][methods[0]]["groups"][group]["n_frames"]
        labels.append(f"{group_label(group)} ({n})")
    labels.append("Pooled")
    ax.set_yticks(base, labels)
    ax.set_xlim(-0.35, 1.0)
    ax.set_xlabel("Spearman ρ (uncertainty vs force error)")
    ax.grid(axis="x", visible=False)

    handles = [
        plt_line(ARM_COLOURS[method], ARM_LABELS[method], marker="o")
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    save(fig, str(outdir / "fig2_discrimination"), formats=("pdf", "svg", "png"))


# ---------------------------------------------------------------------------
# Figure 3 -- calibration and scale
# ---------------------------------------------------------------------------


def build_calibration(analysis: Path, outdir: Path) -> None:
    report = json.loads(analysis.read_text(encoding="utf-8"))
    methods = [name for name in ARM_ORDER if name in report["methods"]]
    groups = list(next(iter(report["methods"].values()))["groups"])
    groups = sorted(groups, key=_group_sort_key)

    fig = plt_figure((TEXT, 3.5))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0), hspace=0.07, wspace=0.10)
    top = [fig.add_subplot(grid[0, 0])]
    top += [fig.add_subplot(grid[0, i], sharey=top[0]) for i in (1, 2)]
    for ax, group in zip(top, groups):
        for method in methods:
            curve = report["methods"][method]["groups"][group]["risk_coverage"]
            xs = [float(key) for key in curve]
            ys = [curve[key] for key in curve]
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=2.4,
                color=ARM_COLOURS[method],
                label=ARM_LABELS[method],
            )
        ax.set_xlim(0.05, 1.02)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Coverage")
        ax.grid(axis="x", visible=False)
        ax.text(
            0.5,
            1.03,
            group_label(group),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    top[0].set_ylabel("Force RMSE (eV/Å)")
    for ax in top[1:]:
        ax.tick_params(labelleft=False)
    panel_labels(top, ["(a)", "", ""])

    bottom = fig.add_subplot(grid[1, :])
    x = np.arange(len(groups))
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        values = []
        for group in groups:
            value = report["methods"][method]["groups"][group][
                "median_error_over_sigma"
            ]
            values.append(np.nan if value is None else value)
        bottom.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            color=ARM_COLOURS[method],
            label=ARM_LABELS[method],
        )
    bottom.set_yscale("log")
    bottom.set_xticks(x, [group_label(group) for group in groups])
    bottom.set_ylabel("Median error / spread")
    bottom.axhline(1.0, color="#7F7F7F", linewidth=0.6, linestyle="--")
    bottom.grid(axis="x", visible=False)
    panel_labels([bottom], ["(b)"])

    handles = [
        plt_line(ARM_COLOURS[method], ARM_LABELS[method]) for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    save(fig, str(outdir / "fig3_calibration"), formats=("pdf", "svg", "png"))


# ---------------------------------------------------------------------------
# Figure 4 -- label and compute efficiency
# ---------------------------------------------------------------------------


def build_efficiency(manifest_path: Path, outdir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = manifest.get("results", {})
    strategies = [name for name in (*ARM_ORDER, "random") if name in results]
    if not strategies:
        raise ValueError(f"{manifest_path} has no strategy results")

    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.44, sharey=True)
    for name in strategies:
        records = results[name].get("records", [])
        if not records:
            continue
        labels = [record["labeled_count"] for record in records]
        rmse = [record["test_force_rmse_ev_per_angstrom"] for record in records]
        seconds = [record.get("cumulative_seconds") for record in records]
        colour = ARM_COLOURS.get(name, "#7F7F7F")
        label = ARM_LABELS.get(name, name.title())
        axes[0].plot(labels, rmse, marker="o", ms=2.6, color=colour, label=label)
        if all(value is not None and value > 0 for value in seconds):
            axes[1].plot(
                [value / 3600.0 for value in seconds],
                rmse,
                marker="o",
                ms=2.6,
                color=colour,
                label=label,
            )

    target = manifest.get("target_rmse")
    if target is not None:
        for ax in axes:
            ax.axhline(target, color="#7F7F7F", linewidth=0.6, linestyle="--")

    axes[0].set_xlabel("Acquired labels")
    axes[0].set_ylabel("Test force RMSE (eV/Å)")
    axes[1].set_xlabel("Cumulative GPU-hours")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    panel_labels(axes, ["(a)", "(b)"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(4, len(labels)),
        frameon=False,
    )
    save(fig, str(outdir / "fig4_efficiency"), formats=("pdf", "svg", "png"))


# ---------------------------------------------------------------------------
# Small helpers that keep matplotlib imports local to the plotting path
# ---------------------------------------------------------------------------


def plt_figure(size):
    import matplotlib.pyplot as plt

    return plt.figure(figsize=size, layout="constrained")


def plt_patch(colour: str, label: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=colour, edgecolor="white", label=label)


def plt_line(colour: str, label: str, marker=None):
    from matplotlib.lines import Line2D

    return Line2D(
        [0], [0], color=colour, label=label, marker=marker, markersize=3.6
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--analysis", type=Path, default=None)
    parser.add_argument("--al-manifest", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--only",
        default="1,2,3,4",
        help="comma-separated figure numbers to build",
    )
    args = parser.parse_args()
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    args.outdir.mkdir(parents=True, exist_ok=True)

    use_style()
    if "1" in wanted:
        if args.split_manifest is None or args.train_file is None:
            raise SystemExit("figure 1 needs --split-manifest and --train-file")
        build_testbed(args.split_manifest, args.train_file, args.outdir)
    if "2" in wanted:
        if args.analysis is None:
            raise SystemExit("figure 2 needs --analysis")
        build_discrimination(args.analysis, args.outdir)
    if "3" in wanted:
        if args.analysis is None:
            raise SystemExit("figure 3 needs --analysis")
        build_calibration(args.analysis, args.outdir)
    if "4" in wanted:
        if args.al_manifest is None:
            raise SystemExit("figure 4 needs --al-manifest")
        build_efficiency(args.al_manifest, args.outdir)


if __name__ == "__main__":
    main()
