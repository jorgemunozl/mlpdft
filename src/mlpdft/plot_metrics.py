#!/usr/bin/env python3
"""Generate publication-quality bar charts from MACE and FitSNAP metrics JSON.

Reads a single metrics JSON file (produced by evaluate_mace_metrics.py) or
the older per-group .txt files, and produces PDF plots for presentation slides.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mlpdft.constants import PLOT_DIR

# ---------------------------------------------------------------------------
# Short labels for presentation slides
# ---------------------------------------------------------------------------
SHORT_LABELS: dict[str, str] = {
    "LIFINTERFACE_KJPAW_V1": "KJPAW_V1",
    "LIFINTERFACE_KJPAW_NPT_V2": "KJPAW_NPT_V2",
    "LIFINTERFACE_KJPAW_NPT": "KJPAW_NPT",
    "LIWITHF_NPT_FINAL": "NPT_FINAL",
    "LIWITHF_ISOLATED": "ISOLATED",
    "LIF64_KJPAW_V2": "LIF64_V2",
    "LIWITHF_V3": "V3",
    "LIF64_ISOLATED": "LIF64_ISOL",
}

ATOM_COUNTS: dict[str, int] = {
    "LIFINTERFACE_KJPAW_V1": 122,
    "LIFINTERFACE_KJPAW_NPT_V2": 122,
    "LIFINTERFACE_KJPAW_NPT": 122,
    "LIWITHF_NPT_FINAL": 54,
    "LIWITHF_ISOLATED": 54,
    "LIF64_KJPAW_V2": 64,
    "LIWITHF_V3": 54,
    "LIF64_ISOLATED": 64,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class GroupMetrics:
    group: str
    energy_rmse_per_atom_mev: float
    force_rmse_mev_ang: float


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
class MetricsReader:
    """Read metrics from a JSON file or a directory of .txt files."""

    @staticmethod
    def from_json(json_path: Path) -> list[GroupMetrics]:
        """Parse the single JSON produced by evaluate_mace_metrics.py."""
        data: dict[str, Any] = json.loads(json_path.read_text())
        results: list[GroupMetrics] = []
        for entry in data["results"]:
            results.append(
                GroupMetrics(
                    group=entry["group"],
                    energy_rmse_per_atom_mev=entry["energy_rmse_per_atom"] * 1000,
                    force_rmse_mev_ang=entry["force_rmse_all"] * 1000,
                )
            )
        results.sort(key=lambda m: m.energy_rmse_per_atom_mev)
        return results

    @staticmethod
    def from_txt_dir(directory: Path, *, style: str = "mace") -> list[GroupMetrics]:
        """Legacy: parse per-group .txt files."""
        results: list[GroupMetrics] = []
        for txt_path in sorted(directory.glob("*.txt")):
            try:
                if style == "mace":
                    results.append(MetricsReader._parse_mace_txt(txt_path))
                else:
                    results.append(MetricsReader._parse_fitsnap_txt(txt_path))
            except (ValueError, KeyError) as exc:
                print(f"  [SKIP] {txt_path.name}: {exc}")
        results.sort(key=lambda m: m.energy_rmse_per_atom_mev)
        return results

    @staticmethod
    def _extract(pattern: str, text: str) -> float:
        m = re.search(pattern, text, re.DOTALL)
        if m is None:
            raise ValueError(f"Could not find pattern {pattern!r}")
        return float(m.group(1))

    @classmethod
    def _parse_mace_txt(cls, path: Path) -> GroupMetrics:
        text = path.read_text()
        group = path.stem.rsplit("_", 2)[0]
        e_rmse_mev = cls._extract(r"Energy.*per atom.*\n.*RMSE\s*=\s*([\d.]+)", text)
        f_rmse_ev = cls._extract(r"Forces.*\n.*RMSE\s*\(all\)\s*=\s*([\d.]+)", text)
        return GroupMetrics(
            group=group,
            energy_rmse_per_atom_mev=e_rmse_mev,
            force_rmse_mev_ang=f_rmse_ev * 1000,
        )

    @classmethod
    def _parse_fitsnap_txt(cls, path: Path) -> GroupMetrics:
        text = path.read_text()
        group = path.stem.rsplit("_", 2)[0]
        e_rmse_total_ev = cls._extract(r"Energy.*\n.*RMSE\s*=\s*([\d.]+)", text)
        f_rmse_ev = cls._extract(r"Forces.*\n.*RMSE\s*\(all\)\s*=\s*([\d.]+)", text)
        natoms = ATOM_COUNTS.get(group, 1)
        return GroupMetrics(
            group=group,
            energy_rmse_per_atom_mev=(e_rmse_total_ev / natoms) * 1000,
            force_rmse_mev_ang=f_rmse_ev * 1000,
        )


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------
class BarPlotter:
    """Generate grouped / single bar charts for presentation slides."""

    BAR_WIDTH = 0.35
    COLOR_MACE = "#2171b5"
    COLOR_FITSNAP = "#e6550d"
    FIG_SIZE = (6.5, 4.0)
    DPI = 150

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Energy bar chart (single model)
    # ------------------------------------------------------------------
    def plot_energy(self, metrics: list[GroupMetrics], title_suffix: str = "") -> Path:
        """Single bar chart: Energy RMSE per atom (meV/atom)."""
        labels = [SHORT_LABELS.get(m.group, m.group) for m in metrics]
        values = np.array([m.energy_rmse_per_atom_mev for m in metrics])
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=self.FIG_SIZE, dpi=self.DPI)
        bars = ax.bar(
            x,
            values,
            self.BAR_WIDTH * 2,
            color=self.COLOR_MACE,
            edgecolor="white",
            linewidth=0.5,
        )

        y_room = max(values) * 0.03 if max(values) > 0 else 1
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_room,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("RMSE (meV / atom)", fontsize=11)
        ax.set_title(
            f"Energy RMSE per atom{title_suffix}", fontsize=13, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / "energy_rmse_per_group.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Force grouped bar chart (MACE vs FitSNAP)
    # ------------------------------------------------------------------
    def plot_force_grouped(
        self,
        mace_metrics: list[GroupMetrics],
        fitsnap_metrics: list[GroupMetrics],
        title_suffix: str = "",
    ) -> Path:
        """Grouped bar chart: Force RMSE (meV/Å) — MACE vs FitSNAP per group."""
        fitsnap_map = {m.group: m for m in fitsnap_metrics}
        aligned: list[tuple[str, float, float | None]] = []
        for m in mace_metrics:
            fs = fitsnap_map.get(m.group)
            aligned.append(
                (m.group, m.force_rmse_mev_ang, fs.force_rmse_mev_ang if fs else None)
            )

        labels = [SHORT_LABELS.get(g, g) for g, _, _ in aligned]
        mace_vals = np.array([v for _, v, _ in aligned])
        fitsnap_vals = [v for _, _, v in aligned]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=self.FIG_SIZE, dpi=self.DPI)
        bars_mace = ax.bar(
            x - self.BAR_WIDTH / 2,
            mace_vals,
            self.BAR_WIDTH,
            color=self.COLOR_MACE,
            edgecolor="white",
            linewidth=0.5,
            label="MACE",
        )

        fs_mask = np.array([v is not None for v in fitsnap_vals])
        xs_fs = x[fs_mask] + self.BAR_WIDTH / 2
        vals_fs = [v for v in fitsnap_vals if v is not None]
        bars_fs: list = []
        if vals_fs:
            bars_fs = ax.bar(
                xs_fs,
                vals_fs,
                self.BAR_WIDTH,
                color=self.COLOR_FITSNAP,
                edgecolor="white",
                linewidth=0.5,
                label="FitSNAP",
                alpha=0.85,
            )

        all_vals = list(mace_vals) + [v for v in fitsnap_vals if v is not None]
        y_room = max(all_vals) * 0.03 if all_vals else 1
        for bar, val in zip(bars_mace, mace_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_room,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=self.COLOR_MACE,
            )
        for bar, val in zip(bars_fs, vals_fs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_room,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=self.COLOR_FITSNAP,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("RMSE (meV / Å)", fontsize=11)
        ax.set_title(f"Force RMSE{title_suffix}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, frameon=True, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / "force_rmse_per_group.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    metrics_dir = script_dir / "metrics"

    # ── MACE (mock_2_test): prefer JSON, fall back to .txt ──
    mace_json = metrics_dir / "mock_2_test" / "mock_2_test.json"
    if mace_json.exists():
        print(f"Reading MACE metrics from {mace_json}")
        mace_metrics = MetricsReader.from_json(mace_json)
    else:
        print("JSON not found, falling back to .txt files")
        mace_metrics = MetricsReader.from_txt_dir(
            metrics_dir / "mock_2_test", style="mace"
        )
    for m in mace_metrics:
        print(
            f"  {m.group:35s}  E={m.energy_rmse_per_atom_mev:8.1f} meV/at  F={m.force_rmse_mev_ang:8.1f} meV/Å"
        )

    # ── FitSNAP (legacy .txt only) ──
    print("\nParsing FitSNAP metrics …")
    fitsnap_metrics = MetricsReader.from_txt_dir(
        metrics_dir / "fitsnap", style="fitsnap"
    )
    for m in fitsnap_metrics:
        print(f"  {m.group:35s}  F={m.force_rmse_mev_ang:8.1f} meV/Å")

    # ── Plot ──
    plotter = BarPlotter(output_dir=PLOT_DIR)

    out_e = plotter.plot_energy(mace_metrics, title_suffix=" — MACE")
    print(f"\n✓ Energy plot saved: {out_e}")

    out_f = plotter.plot_force_grouped(
        mace_metrics, fitsnap_metrics, title_suffix=" — MACE vs FitSNAP"
    )
    print(f"✓ Force plot saved:  {out_f}")


if __name__ == "__main__":
    main()
