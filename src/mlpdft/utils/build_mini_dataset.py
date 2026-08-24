#!/usr/bin/env python3
"""
Build a mini dataset by sampling a few frames from different groups.

Picks evenly-spaced frames from a representative extxyz file in each group,
tags each frame with its group, and writes them to a single extxyz.

Usage:
    python build_mini_dataset.py [--groups G1 G2 ...] [--per-group N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from mlpdft.constants import DATA_DIR, OUTPUTS_DIR, XYZ_DIR

# Groups known to have extxyz files locally, spanning Li-F interface, Li-rich,
# bulk, B-Li, and salt chemistries.
MINI_GROUPS = [
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_NPT",
    "LIWITHF_ISOLATED",
    "LIWITHF_NPT_FINAL",
    "LIF64_ISOLATED",
    "BLI_V2",
    "BLI_NPT",
    "LIBF4_NPT",
    "LIBF4_NPT_FINAL",
    "LIBF4_V4",
    "LIF64_KJPAW_V2",
    "LIF64_KJPAW_NPT",
    "BLI_INTERFACE_FINAL",
    "BLI_INTERFACE_NPT_FINAL",
]


def representative_file(group: str) -> Path | None:
    """Return the extxyz file to sample from for a group, preferring *_5_100."""
    files = sorted((DATA_DIR / group / XYZ_DIR).glob("*.extxyz"))
    if not files:
        return None
    for path in files:
        if "_5_100" in path.name:
            return path
    return max(files, key=lambda p: p.stat().st_size)


def sample_group(group: str, per_group: int) -> list[Atoms]:
    """Sample ``per_group`` evenly-spaced frames from a group."""
    path = representative_file(group)
    if path is None:
        print(f"  [skip] {group}: no extxyz found")
        return []

    frames = read(str(path), index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    if per_group == 1:
        idx = [len(frames) - 1]
    elif per_group >= len(frames):
        idx = range(len(frames))
    else:
        idx = np.linspace(0, len(frames) - 1, per_group, dtype=int)

    sampled = []
    for i in idx:
        atoms = frames[int(i)].copy()
        atoms.info["group"] = group
        atoms.info["source_file"] = path.name
        sampled.append(atoms)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mini cross-group dataset.")
    parser.add_argument("--groups", nargs="+", default=MINI_GROUPS, help="Group names")
    parser.add_argument(
        "--per-group", type=int, default=1, help="Frames to sample per group"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "mini_dataset" / "mini_dataset_v2.extxyz",
        help="Output extxyz path",
    )
    args = parser.parse_args()

    frames: list[Atoms] = []
    for group in args.groups:
        frames.extend(sample_group(group, args.per_group))

    if not frames:
        parser.error("No frames were sampled; check the --groups list.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, frames)
    print(
        print(
            f"Wrote {len(frames)} frames from {len({f.info['group'] for f in frames})} groups"
        )
    )
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
