#!/usr/bin/env python3
"""Materialize the Phase 1 frozen splits as extended XYZ files.

Reads ``assets/research/PHASE-1/audit/frames.csv``, which records the source
file and frame index for every unique frame and its split assignment, and writes
one extxyz file per split:

    train_pool  -> uq_train.extxyz
    calibration -> uq_valid.extxyz
    test        -> uq_test.extxyz
    ood_test    -> uq_ood.extxyz

Each output frame keeps ``REF_energy`` and ``REF_forces`` and gains
``config_type`` / ``environment`` / ``split`` / ``source_file`` /
``source_frame`` so every row remains traceable to the audit.

Pure standard library, no ASE dependency.

Usage
-----
    python3 src/mlpdft/utils/build_phase1_splits.py \
        --data-root /path/to/dataset \
        --frames assets/research/PHASE-1/audit/frames.csv \
        --output-dir /path/to/dataset/xyz_files/uq_splits

Use ``--limit-per-split 20`` for a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from phase1_audit import parse_energy, parse_lattice, read_frames
except ImportError:  # pragma: no cover - package import
    from mlpdft.utils.phase1_audit import parse_energy, parse_lattice, read_frames

SPLIT_FILES = {
    "train_pool": "uq_train.extxyz",
    "calibration": "uq_valid.extxyz",
    "test": "uq_test.extxyz",
    "ood_test": "uq_ood.extxyz",
}

LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def write_frame(
    handle,
    natoms: int,
    comment: str,
    species: list[str],
    positions: list[tuple[float, float, float]],
    forces: list[tuple[float, float, float]],
    row: dict,
) -> None:
    lattice_match = LATTICE_RE.search(comment)
    if not lattice_match:
        raise ValueError(f"frame {row['file']}:{row['frame_index']} has no Lattice")
    energy = parse_energy(comment)
    if energy is None:
        raise ValueError(f"frame {row['file']}:{row['frame_index']} has no REF_energy")
    if not forces:
        raise ValueError(f"frame {row['file']}:{row['frame_index']} has no REF_forces")

    properties = "species:S:1:pos:R:3:REF_forces:R:3"
    new_comment = (
        f'Lattice="{lattice_match.group(1)}" '
        f"Properties={properties} "
        f"REF_energy={energy!r} "
        f"config_type={row['environment']} "
        f"environment={row['environment']} "
        f"split={row['split']} "
        f"source_file={row['file']} "
        f"source_frame={row['frame_index']}"
    )
    handle.write(f"{natoms}\n{new_comment}\n")
    for symbol, (x, y, z), (fx, fy, fz) in zip(species, positions, forces):
        handle.write(
            f"{symbol} {x:.8f} {y:.8f} {z:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n"
        )


def build_split(
    split: str,
    rows: list[dict],
    data_root: Path,
    output_path: Path,
    limit: int | None,
) -> dict:
    selected = rows[:limit] if limit else rows
    by_file: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in selected:
        by_file[row["file"]][int(row["frame_index"])] = row

    written = 0
    environment_counts: Counter = Counter()
    with output_path.open("w") as handle:
        for file_name, wanted in sorted(by_file.items()):
            # All rows for a file come from one group directory.
            group = next(iter(wanted.values()))["group"]
            path = data_root / group / "xyz_files" / file_name
            if not path.is_file():
                raise FileNotFoundError(path)
            remaining = set(wanted)
            for index, (natoms, comment, species, positions, forces) in enumerate(
                read_frames(path)
            ):
                if index not in wanted:
                    continue
                row = wanted[index]
                write_frame(
                    handle, natoms, comment, species, positions, forces, row
                )
                written += 1
                environment_counts[row["environment"]] += 1
                remaining.discard(index)
                if not remaining:
                    break
            if remaining:
                raise ValueError(
                    f"{path} ended before frames {sorted(remaining)[:5]} were read"
                )

    if written != len(selected):
        raise ValueError(
            f"{split}: wrote {written} frames but selected {len(selected)}"
        )
    return {
        "split": split,
        "file": output_path.name,
        "frames": written,
        "environments": dict(environment_counts),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("assets/research/PHASE-1/audit/frames.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Write only the first N rows per split (smoke test).",
    )
    args = parser.parse_args()

    with args.frames.open() as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["split"] in SPLIT_FILES]
    excluded_counts = Counter(
        row["split"] for row in all_rows if row["split"] not in SPLIT_FILES
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    for split_rows in by_split.values():
        split_rows.sort(
            key=lambda row: (
                row["environment"],
                row["group"],
                row["file"],
                int(row["frame_index"]),
            )
        )

    results = []
    for split, filename in SPLIT_FILES.items():
        split_rows = by_split.get(split, [])
        if not split_rows:
            continue
        result = build_split(
            split,
            split_rows,
            args.data_root,
            args.output_dir / filename,
            args.limit_per_split,
        )
        results.append(result)
        print(
            f"{filename:20s} frames={result['frames']:6d} "
            f"environments={result['environments']}"
        )

    # Disjointness check on the source coordinates.
    seen: set[tuple[str, int]] = set()
    duplicates = 0
    for row in all_rows:
        key = (row["file"], int(row["frame_index"]))
        if key in seen:
            duplicates += 1
        seen.add(key)
    if duplicates:
        raise ValueError(f"{duplicates} duplicate source frames in frames.csv")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames_csv": str(args.frames),
        "data_root": str(args.data_root),
        "limit_per_split": args.limit_per_split,
        "splits": results,
        "source_rows": len(rows),
        "excluded_rows": dict(excluded_counts),
        "duplicate_source_frames": duplicates,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"\nWrote {len(results)} split files to {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
