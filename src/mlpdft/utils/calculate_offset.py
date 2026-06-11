#!/usr/bin/env python3
"""
Count atoms and elemental composition for each frame in every dataset's extxyz.
Prints a summary and saves to outputs/metrics/atom_counts.txt.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import ase.io

from mlpdft.constants import DATA_DIR, OUTPUTS_DIR, XYZ_DIR

GROUPS = [
    "LIF64_ISOLATED",
    "LIF64_KJPAW_V2",
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_NPT",
    "LIFINTERFACE_KJPAW_NPT_V2",
    "LIF_KJPAW",
    "LIWITHF_V3",
    "LIWITHF_ISOLATED",
    "LIWITHF_NPT_FINAL",
    "BLI_V2",
    "LIBF4_V4",
]

FRAME_STRIDE = 5
MAX_FRAMES = 100


def count_atoms_in_group(group: str) -> list[dict]:
    """Return per-frame atom counts for a group."""
    extxyz_path = (
        DATA_DIR / group / XYZ_DIR / f"{group}_{FRAME_STRIDE}_{MAX_FRAMES}.extxyz"
    )
    if not extxyz_path.exists():
        print(f"  [SKIP] {extxyz_path} not found")
        return []

    atoms_list = ase.io.read(str(extxyz_path), index=":")
    results = []
    for atoms in atoms_list:
        symbols = atoms.get_chemical_symbols()
        total = len(symbols)
        composition = dict(Counter(symbols).most_common())
        results.append({"total": total, "composition": composition})
    return results


def main() -> None:
    output_path = OUTPUTS_DIR / "metrics" / "atom_counts.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("ATOM COUNTS PER DATASET")
    lines.append("=" * 60)

    all_summaries = []

    for group in GROUPS:
        frames = count_atoms_in_group(group)
        if not frames:
            lines.append(f"\n{group}: [no data]")
            continue

        # Summary stats
        totals = [f["total"] for f in frames]
        min_n, max_n = min(totals), max(totals)
        consistent = all(t == totals[0] for t in totals)

        # Composition: check if consistent across frames
        comp_keys = set()
        for f in frames:
            comp_keys.update(f["composition"].keys())

        # Check if composition is the same in every frame
        comps_seen = set()
        comp_str = ""
        for f in frames:
            c = tuple(sorted(f["composition"].items()))
            comps_seen.add(c)

        if len(comps_seen) == 1:
            single_comp = dict(sorted(frames[0]["composition"].items()))
            comp_parts = [f"{elem}: {n}" for elem, n in single_comp.items()]
            comp_str = ", ".join(comp_parts)
            comp_consistent = True
        else:
            comp_consistent = False
            # Show first frame composition as example
            ex = dict(sorted(frames[0]["composition"].items()))
            comp_parts = [f"{elem}: {n}" for elem, n in ex.items()]
            comp_str = ", ".join(comp_parts) + " (varies!)"

        if consistent and comp_consistent:
            status = "✓ uniform"
        else:
            status = "⚠ varies across frames"

        summary = {
            "group": group,
            "n_frames": len(frames),
            "total_atoms": totals[0] if consistent else f"{min_n}–{max_n}",
            "composition": comp_str,
            "status": status,
        }
        all_summaries.append(summary)

    # Print summary table
    lines.append("")
    header = (
        f"{'Group':<35} {'Frames':>6} {'Atoms':>8}  {'Composition':<35} {'Status':<20}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for s in all_summaries:
        lines.append(
            f"{s['group']:<35} {s['n_frames']:>6} {str(s['total_atoms']):>8}  "
            f"{s['composition']:<35} {s['status']:<20}"
        )

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Saved to: {output_path}")
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)

    output_path.write_text(output, encoding="utf-8")
    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
