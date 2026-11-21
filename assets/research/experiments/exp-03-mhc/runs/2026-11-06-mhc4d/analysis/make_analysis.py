#!/usr/bin/env python3
"""Build the MHC-4D analysis tables and figures from the archived run.

Run from this directory:
    uv run --no-project --with matplotlib --with numpy python make_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))

from orx_figstyle import (  # noqa: E402
    BASELINE,
    PALETTE,
    TEXT,
    figure,
    figure_grid,
    panel_labels,
    save,
    use_style,
)

DATA = HERE / "data"
FIGS = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SUMMARY = BASE / "full" / "evaluation" / "uq_summary.json"
PREDICTIONS = BASE / "full" / "evaluation" / "frozen_mhc_predictions.csv"
TRAIN_LOG = BASE / "full" / "mhc_frozen_readouts" / "results" / "mhc_frozen_readouts_run-123_train.txt"

METHODS = ("frozen_mhc", "previous_frozen_mhc", "zero_shot")
SCOPES = ("overall", "bulk_LiF", "F_doped_Li")


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def frame_rows() -> list[dict]:
    with PREDICTIONS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def training_rows() -> list[dict]:
    records = [
        json.loads(line)
        for line in TRAIN_LOG.read_text().splitlines()
        if line.strip().startswith("{")
    ]
    rows = []
    for rec in records:
        if rec.get("mode") != "eval" or not isinstance(rec.get("epoch"), int):
            continue
        if not rec.get("head"):
            continue
        rows.append(
            {
                "epoch": rec["epoch"],
                "head": rec["head"],
                "rmse_f_mev_a": 1000.0 * float(rec["rmse_f"]),
                "loss": float(rec["loss"]),
            }
        )
    return rows


def write_metrics(summary: dict) -> None:
    rows = []
    for method in METHODS:
        block = summary[method]
        alpha = (block.get("calibration") or {}).get("alpha_force")
        for scope in SCOPES:
            data = block["overall"] if scope == "overall" else block["by_group"][scope]
            rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "n_frames": data.get("n_frames", ""),
                    "force_rmse_mev_a": 1000.0 * data["force_rmse_ev_per_angstrom"],
                    "force_mae_mev_a": 1000.0 * data["force_mae_ev_per_angstrom"],
                    "spearman": data.get("spearman_uncertainty_vs_force_error", ""),
                    "pearson": data.get("pearson_uncertainty_vs_force_error", ""),
                    "coverage_1sigma_force": data.get("coverage_1sigma_force", ""),
                    "coverage_2sigma_force": data.get("coverage_2sigma_force", ""),
                    "rll_force": data.get("rll_force", ""),
                    "alpha_force": "" if alpha is None else alpha,
                }
            )
    with (DATA / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_training(rows: list[dict]) -> None:
    with (DATA / "training_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "head", "rmse_f_mev_a", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def risk_curve(errors: np.ndarray, sigmas: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(sigmas)
    values = []
    for fraction in fractions:
        keep = max(1, int(round(fraction * len(order))))
        values.append(float(np.sqrt(np.mean(errors[order[:keep]] ** 2))))
    return fractions, np.asarray(values)


def write_risk(rows: list[dict]) -> dict:
    fractions = np.linspace(0.05, 1.0, 20)
    out = []
    curves = {}
    for label, subset in (
        ("overall", rows),
        ("bulk_LiF", [r for r in rows if r["group"] == "bulk_LiF"]),
        ("F_doped_Li", [r for r in rows if r["group"] == "F_doped_Li"]),
    ):
        errors = np.asarray([float(r["force_rmse_ev_per_angstrom"]) for r in subset])
        sigmas = np.asarray([float(r["force_uncertainty_ev_per_angstrom"]) for r in subset])
        x, y = risk_curve(errors, sigmas, fractions)
        curves[label] = (x, y)
        for coverage, rmse in zip(x, y):
            out.append(
                {
                    "group": label,
                    "coverage": coverage,
                    "rmse_mev_a": 1000.0 * rmse,
                    "n_frames_retained": max(1, int(round(coverage * len(subset)))),
                }
            )
    with (DATA / "risk_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return curves


def plot_training(rows: list[dict]) -> None:
    use_style()
    fig, ax = figure(width=TEXT, ratio=0.56)
    heads = sorted({row["head"] for row in rows})
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
    for head, color in zip(heads, colors):
        sub = sorted((row for row in rows if row["head"] == head), key=lambda r: r["epoch"])
        ax.plot(
            [row["epoch"] for row in sub],
            [row["rmse_f_mev_a"] for row in sub],
            color=color,
            lw=0.9,
            alpha=0.85,
            label=head.replace("mhc_head_", "head "),
        )
    epochs = sorted({row["epoch"] for row in rows})
    mean = [
        float(np.mean([row["rmse_f_mev_a"] for row in rows if row["epoch"] == epoch]))
        for epoch in epochs
    ]
    ax.plot(epochs, mean, color=PALETTE["black"], lw=1.8, label="mean of 4 heads")
    ax.set_yscale("log")
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(25, 500)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation force RMSE (meV Å$^{-1}$)")
    ax.grid(axis="y", which="major", color="#E6E6E6", lw=0.6)
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.6)
    ax.annotate(
        "plateau ≈ 33 meV Å$^{-1}$",
        xy=(max(epochs), mean[-1]),
        xytext=(52, 58),
        fontsize=7,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
    )
    save(fig, str(FIGS / "training-curves"))


def plot_uncertainty(rows: list[dict], summary: dict) -> None:
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.46)
    xticks = {"bulk_LiF": [4, 6, 8], "F_doped_Li": [3.0, 3.2, 3.4]}
    yticks = {"bulk_LiF": [30, 40, 50], "F_doped_Li": [22, 23, 24]}
    for ax, group, label, color in zip(
        axes,
        ("bulk_LiF", "F_doped_Li"),
        ("Bulk LiF", "F-doped Li"),
        (PALETTE["blue"], PALETTE["orange"]),
    ):
        subset = [row for row in rows if row["group"] == group]
        x = np.asarray([1000.0 * float(row["force_uncertainty_ev_per_angstrom"]) for row in subset])
        y = np.asarray([1000.0 * float(row["force_rmse_ev_per_angstrom"]) for row in subset])
        keep = (x > 0) & (y > 0)
        ax.scatter(x[keep], y[keep], s=14, color=color, alpha=0.65, linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Committee σ (meV Å$^{-1}$)")
        ax.set_ylabel("Frame force RMSE (meV Å$^{-1}$)")
        rho = summary["frozen_mhc"]["by_group"][group]["spearman_uncertainty_vs_force_error"]
        ax.text(0.04, 0.94, label, transform=ax.transAxes, fontweight="bold")
        ax.text(0.04, 0.86, f"ρ = {rho:.2f}", transform=ax.transAxes)
        ax.set_xticks(xticks[group])
        ax.set_xticklabels([f"{value:g}" for value in xticks[group]])
        ax.set_yticks(yticks[group])
        ax.set_yticklabels([f"{value:g}" for value in yticks[group]])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, which="major", color="#E8E8E8", lw=0.5)
    panel_labels(axes)
    save(fig, str(FIGS / "uncertainty-vs-error"))


def plot_risk(curves: dict) -> None:
    use_style()
    fig, ax = figure(width=TEXT, ratio=0.58)
    styles = (
        ("overall", "Overall", PALETTE["black"]),
        ("bulk_LiF", "Bulk LiF", PALETTE["blue"]),
        ("F_doped_Li", "F-doped Li", PALETTE["orange"]),
    )
    for key, label, color in styles:
        x, y = curves[key]
        ax.plot(x, 1000.0 * y, color=color, lw=1.6, label=label)
        ax.text(
            x[-1],
            1000.0 * y[-1] + 0.9,
            label,
            color=color,
            fontsize=7,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8),
        )
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(20, 43)
    ax.set_xlabel("Coverage (lowest-σ fraction retained)")
    ax.set_ylabel("Force RMSE on retained frames (meV Å$^{-1}$)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.annotate(
        "F-doped: flat",
        xy=(0.62, 23.3),
        xytext=(0.48, 21.2),
        fontsize=7,
        color="#555555",
        arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
    )
    save(fig, str(FIGS / "risk-coverage"))


def main() -> None:
    summary = load_summary()
    rows = frame_rows()
    training = training_rows()
    write_metrics(summary)
    write_training(training)
    curves = write_risk(rows)
    plot_training(training)
    plot_uncertainty(rows, summary)
    plot_risk(curves)
    print("Wrote:", *(str(path) for path in sorted(DATA.glob("*.csv"))))
    print("Wrote:", *(str(path) for path in sorted(FIGS.glob("*.svg"))))


if __name__ == "__main__":
    main()
