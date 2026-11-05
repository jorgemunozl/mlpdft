"""Experiment 3 gate and C/D/A attribution report.

Reads the merged ``uq_summary.json`` produced by ``uq_compare`` and answers
the three questions the pre-registered protocol asks about the frozen-body
multi-head committee (MHC):

* Does it clear the accuracy gate (force RMSE <= 1.25x the independent
  committee) overall and in every environment?
* Does its within-environment rank correlation stay within 0.10 of the
  committee's?
* What do the two controls say?  ``C - D`` is the cost of splitting the labels
  across four heads; ``D - A`` is the cost of freezing the body.

The report is deliberately independent of the training logs: it uses the same
per-frame summaries that feed the paper, so its numbers are the paper's numbers.

Usage
-----
::

    python src/mlpdft/utils/experiment3_report.py \
        --summary /workspace/runs/lif_uq_mhc_002/evaluation/uq_summary.json \
        --outdir  /workspace/runs/lif_uq_mhc_002/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GATE_RATIO = 1.25
GATE_RHO_SLACK = 0.10


def _overall(summary: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in summary:
        raise KeyError(f"{key!r} is not in the summary: {sorted(summary)}")
    return summary[key]["overall"]


def _groups(summary: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return summary[key]["by_group"]


def _rho_mean(summary: dict[str, Any], key: str) -> float | None:
    values = [
        float(group["spearman_uncertainty_vs_force_error"])
        for group in _groups(summary, key).values()
        if group.get("spearman_uncertainty_vs_force_error") is not None
    ]
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(
    summary: dict[str, Any],
    committee_key: str,
    mhc_key: str,
    probe_key: str,
    mhft_key: str | None,
    gate_ratio: float,
    gate_rho_slack: float,
) -> dict[str, Any]:
    arms = [
        key
        for key in (committee_key, mhc_key, probe_key, mhft_key)
        if key and key in summary
    ]
    table = {
        key: {
            "force_rmse_ev_per_angstrom": _overall(summary, key)[
                "force_rmse_ev_per_angstrom"
            ],
            "energy_rmse_mev_per_atom": _overall(summary, key)[
                "energy_rmse_mev_per_atom"
            ],
            "spearman_pooled": _overall(summary, key)[
                "spearman_uncertainty_vs_force_error"
            ],
            "spearman_within_group_mean": _rho_mean(summary, key),
            "n_frames": _overall(summary, key)["n_frames"],
        }
        for key in arms
    }

    committee_groups = _groups(summary, committee_key)
    mhc_groups = _groups(summary, mhc_key)
    probe_groups = _groups(summary, probe_key)

    per_group = {}
    for group in sorted(committee_groups):
        committee = committee_groups[group]["force_rmse_ev_per_angstrom"]
        mhc = mhc_groups[group]["force_rmse_ev_per_angstrom"]
        probe = probe_groups[group]["force_rmse_ev_per_angstrom"]
        per_group[group] = {
            "committee": committee,
            "mhc": mhc,
            "probe": probe,
            "mhc_over_committee": mhc / committee,
            "mhc_minus_committee_mev": (mhc - committee) * 1000.0,
            "mhc_minus_probe_mev": (mhc - probe) * 1000.0,
            "probe_minus_committee_mev": (probe - committee) * 1000.0,
        }

    overall_ratio = (
        table[mhc_key]["force_rmse_ev_per_angstrom"]
        / table[committee_key]["force_rmse_ev_per_angstrom"]
    )
    rho_mhc = table[mhc_key]["spearman_within_group_mean"]
    rho_committee = table[committee_key]["spearman_within_group_mean"]
    rho_ok = (
        rho_mhc is not None
        and rho_committee is not None
        and rho_mhc >= rho_committee - gate_rho_slack
    )
    group_ok = all(
        item["mhc_over_committee"] <= gate_ratio for item in per_group.values()
    )
    passed = bool(overall_ratio <= gate_ratio and group_ok and rho_ok)

    return {
        "arms": table,
        "gate": {
            "overall_ratio": overall_ratio,
            "max_group_ratio": max(
                item["mhc_over_committee"] for item in per_group.values()
            ),
            "ratio_limit": gate_ratio,
            "rho_mhc": rho_mhc,
            "rho_committee": rho_committee,
            "rho_slack": gate_rho_slack,
            "rho_ok": rho_ok,
            "passed": passed,
        },
        "per_group": per_group,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    arms = report["arms"]
    gate = report["gate"]
    lines = [
        "# Experiment 3 — frozen-body MHC gate",
        "",
        "## Arms",
        "",
        "| method | force RMSE (meV/Å) | energy RMSE (meV/atom) | pooled ρ | within-group ρ |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, arm in arms.items():
        lines.append(
            f"| {key} | {1000 * arm['force_rmse_ev_per_angstrom']:.1f} | "
            f"{arm['energy_rmse_mev_per_atom']:.3f} | "
            f"{_fmt(arm['spearman_pooled'], 2)} | "
            f"{_fmt(arm['spearman_within_group_mean'], 2)} |"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- Overall MHC / committee force ratio: **{gate['overall_ratio']:.2f}** "
        f"(limit {gate['ratio_limit']:.2f})",
        f"- Worst environment ratio: **{gate['max_group_ratio']:.2f}**",
        f"- Within-group ρ: MHC {_fmt(gate['rho_mhc'], 2)} vs committee "
        f"{_fmt(gate['rho_committee'], 2)} "
        f"(allowed slack {gate['rho_slack']:.2f})",
        f"- **Verdict: {'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "## Per environment",
        "",
        "| group | committee | MHC | probe | MHC/committee | MHC−probe | probe−committee |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in report["per_group"].items():
        lines.append(
            f"| {group} | {1000 * item['committee']:.1f} | "
            f"{1000 * item['mhc']:.1f} | {1000 * item['probe']:.1f} | "
            f"{item['mhc_over_committee']:.2f}× | "
            f"{item['mhc_minus_probe_mev']:+.1f} | "
            f"{item['probe_minus_committee_mev']:+.1f} |"
        )
    lines += [
        "",
        "All force columns in meV/Å. `MHC − probe` estimates the cost of splitting "
        "the labels across four heads; `probe − committee` estimates the cost of "
        "freezing the body.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--committee-key", default="independent_committee")
    parser.add_argument("--mhc-key", default="frozen_mhc")
    parser.add_argument("--probe-key", default="frozen_probe")
    parser.add_argument(
        "--mhft-key", default="multihead_bootstrap", help="Set empty to omit"
    )
    parser.add_argument("--gate-ratio", type=float, default=GATE_RATIO)
    parser.add_argument("--gate-rho-slack", type=float, default=GATE_RHO_SLACK)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = build_report(
        summary,
        args.committee_key,
        args.mhc_key,
        args.probe_key,
        args.mhft_key or None,
        args.gate_ratio,
        args.gate_rho_slack,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "experiment3_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, args.outdir / "experiment3_report.md")
    print(f"Wrote {args.outdir / 'experiment3_report.md'}")
    print(f"Gate: {'PASS' if report['gate']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
