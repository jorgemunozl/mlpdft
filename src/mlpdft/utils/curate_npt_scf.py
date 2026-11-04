#!/usr/bin/env python3
"""Curate an NPT trajectory by per-frame SCF quality and density regime.

The Phase 1 audit used a group-level SCF warning rate, which is conservative:
it drops a whole trajectory if many SCF steps failed. This script maps each
extxyz frame to its SCF block in the QE ``.out`` log (frame ``i`` corresponds to
block ``i * stride``), marks frames with zero ``eigenvalues not converged``
warnings, applies the same block-equilibration rule, and writes the surviving
frames as a clean extxyz.

For the F-doped NPT run the volume overshoots and settles at a denser state; use
``--transition-time`` to keep only the post-transition (dense) regime.

Pure standard library; reuses the Phase 1 reader.

Usage
-----
    python3 src/mlpdft/utils/curate_npt_scf.py \
        --group LIWITHF_NPT_FINAL \
        --log  /path/LIWITHF_NPT_FINAL.out \
        --xyz  /path/LIWITHF_NPT_FINAL_3_5394.extxyz \
        --stride 3 \
        --environment F_doped_Li \
        --transition-time 6180 --keep-regime dense \
        --tag LIWITHF_NPT_dense \
        --output-dir dataset/xyz_files/curated_npt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from phase1_audit import equilibration_cutoff, parse_energy, read_frames

RY_TO_EV = 13.605693122994
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
STRIDE_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_scf_blocks(log_path: Path) -> list[tuple[float, int]]:
    """Return (energy_eV, warning_count) for each SCF block."""
    blocks: list[tuple[float, int]] = []
    energy: float | None = None
    warnings = 0
    with log_path.open(errors="ignore") as handle:
        for line in handle:
            if "!    total energy" in line:
                match = re.search(r"=\s*(-?\d+\.\d+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * RY_TO_EV
            elif "eigenvalues not converged" in line:
                warnings += 1
            elif "convergence has been achieved" in line:
                if energy is not None:
                    blocks.append((energy, warnings))
                energy = None
                warnings = 0
    return blocks


def scan_frames(xyz_path: Path, stride: int) -> list[dict]:
    """First pass: metadata only, no positions."""
    records: list[dict] = []
    index = 0
    with xyz_path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            natoms = int(line)
            comment = handle.readline()
            lattice = [float(x) for x in LATTICE_RE.search(comment).group(1).split()]
            records.append(
                {
                    "frame_index": index,
                    "time_index": index * stride,
                    "natoms": natoms,
                    "cell_a": lattice[0],
                    "cell_b": lattice[4],
                    "cell_c": lattice[8],
                    "energy": parse_energy(comment),
                }
            )
            for _ in range(natoms):
                handle.readline()
            index += 1
    return records


def write_selected(
    xyz_path: Path,
    output_path: Path,
    selected: dict[int, dict],
    args: argparse.Namespace,
) -> int:
    written = 0
    with xyz_path.open() as handle, output_path.open("w") as out:
        index = 0
        while True:
            line = handle.readline()
            if not line:
                break
            natoms = int(line)
            comment = handle.readline()
            lattice = LATTICE_RE.search(comment).group(1)
            species: list[str] = []
            positions: list[tuple[float, float, float]] = []
            forces: list[tuple[float, float, float]] = []
            has_forces = "REF_forces" in comment
            for _ in range(natoms):
                parts = handle.readline().split()
                species.append(parts[0])
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if has_forces:
                    forces.append((float(parts[4]), float(parts[5]), float(parts[6])))
            record = selected.get(index)
            if record is not None:
                if not has_forces:
                    raise ValueError(f"frame {index} has no REF_forces")
                properties = "species:S:1:pos:R:3:REF_forces:R:3"
                new_comment = (
                    f'Lattice="{lattice}" Properties={properties} '
                    f"REF_energy={record['energy']!r} "
                    f"config_type={args.tag} "
                    f"environment={args.environment} "
                    f"density_regime={record['regime']} "
                    f"scf_clean=True "
                    f"source_group={args.group} "
                    f"source_file={Path(args.xyz).name} "
                    f"source_frame={index} "
                    f"scf_block={record['scf_block']} "
                    f"scf_warnings={record['scf_warnings']}"
                )
                out.write(f"{natoms}\n{new_comment}\n")
                for symbol, (x, y, z), (fx, fy, fz) in zip(
                    species, positions, forces
                ):
                    out.write(
                        f"{symbol} {x:.8f} {y:.8f} {z:.8f} "
                        f"{fx:.8f} {fy:.8f} {fz:.8f}\n"
                    )
                written += 1
            index += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--transition-time",
        type=int,
        default=None,
        help="Keep only frames at or after this time index (dense regime).",
    )
    parser.add_argument(
        "--keep-regime",
        choices=("dense", "expanded"),
        default=None,
        help="Which regime to keep when --transition-time is set.",
    )
    parser.add_argument("--energy-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    stride = args.stride
    if stride is None:
        match = STRIDE_RE.search(args.xyz.stem)
        if not match:
            raise ValueError("Cannot infer stride; pass --stride")
        stride = int(match.group(1))

    blocks = parse_scf_blocks(args.log)
    records = scan_frames(args.xyz, stride)

    if len(records) * stride > len(blocks):
        raise ValueError(
            f"{len(records)} frames x stride {stride} exceeds {len(blocks)} SCF blocks"
        )

    # Energy offset validation: the extxyz energy should track the SCF block
    # energy up to a constant (the log and the extxyz use different references).
    offsets = []
    for record in records:
        block = record["frame_index"] * stride
        record["scf_block"] = block
        block_energy, warnings = blocks[block]
        record["scf_warnings"] = warnings
        record["scf_clean"] = warnings == 0
        if record["energy"] is not None:
            offsets.append(record["energy"] - block_energy)
    mean_offset = sum(offsets) / len(offsets)
    spread = max(offsets) - min(offsets)
    if spread > args.energy_tolerance:
        raise ValueError(
            f"Energy offset spread {spread:.4f} eV exceeds tolerance "
            f"{args.energy_tolerance}; check the stride or log"
        )

    cutoff = equilibration_cutoff(records)
    for record in records:
        record["post_equilibration"] = record["time_index"] >= cutoff
        if args.transition_time is not None:
            record["regime"] = (
                "dense" if record["time_index"] >= args.transition_time else "expanded"
            )
        else:
            record["regime"] = "npt"
        volume_per_atom = (
            record["cell_a"] * record["cell_b"] * record["cell_c"] / record["natoms"]
        )
        record["volume_per_atom"] = volume_per_atom

    selected: dict[int, dict] = {}
    for record in records:
        if not record["scf_clean"] or not record["post_equilibration"]:
            continue
        if args.keep_regime and record["regime"] != args.keep_regime:
            continue
        selected[record["frame_index"]] = record

    if not selected:
        raise ValueError("No frames survived curation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.tag}.extxyz"
    written = write_selected(args.xyz, output_path, selected, args)

    csv_path = args.output_dir / f"{args.tag}_frames.csv"
    with csv_path.open("w", newline="") as handle:
        fields = [
            "frame_index",
            "time_index",
            "scf_block",
            "scf_warnings",
            "scf_clean",
            "post_equilibration",
            "regime",
            "energy",
            "volume_per_atom",
            "selected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["selected"] = record["frame_index"] in selected
            writer.writerow({key: row.get(key) for key in fields})

    manifest = {
        "group": args.group,
        "log": str(args.log),
        "xyz": str(args.xyz),
        "stride": stride,
        "scf_blocks": len(blocks),
        "frames": len(records),
        "energy_offset_mean_eV": mean_offset,
        "energy_offset_spread_eV": spread,
        "equilibration_cutoff_time": cutoff,
        "transition_time": args.transition_time,
        "keep_regime": args.keep_regime,
        "scf_clean_frames": sum(1 for record in records if record["scf_clean"]),
        "post_equilibration_frames": sum(
            1 for record in records if record["post_equilibration"]
        ),
        "selected_frames": written,
        "selected_by_regime": dict(
            Counter(record["regime"] for record in selected.values())
        ),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / f"{args.tag}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
