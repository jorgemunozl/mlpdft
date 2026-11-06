#!/usr/bin/env python3
"""Build the 2,900-frame interim UQ dataset from the Phase 1 audit.

Reuse build_phase1_splits.build_split for extxyz materialization. The held-out
sets are time blocks from the same trajectories, not independent replicas.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from build_phase1_splits import build_split
from curate_npt_scf import parse_scf_blocks
from phase1_audit import frame_hash, parse_energy, parse_lattice, read_frames

QUOTAS = {"bulk_LiF": 800, "F_doped_Li": 800, "interface": 400}
FILES = {"train_pool": "uq_train.extxyz", "validation": "uq_valid.extxyz", "calibration": "uq_calibration.extxyz", "test": "uq_test.extxyz"}
GUARD = 16


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: (int(r["time_index"]), r["group"], r["file"], int(r["frame_index"])))
    if len(rows) < count:
        raise ValueError(f"need {count} frames; only {len(rows)} eligible")
    # Integer arithmetic selects endpoints and avoids platform-dependent randomness.
    return [rows[(i * (len(rows) - 1)) // (count - 1)] for i in range(count)]


def select(all_rows: list[dict]) -> list[dict]:
    chosen = []
    for env, ntrain in QUOTAS.items():
        grouped = {split: [r for r in all_rows if r["environment"] == env and r["split"] == split] for split in ("train_pool", "calibration", "test")}
        train_end = max(int(r["time_index"]) for r in grouped["train_pool"])
        cal_start = min(int(r["time_index"]) for r in grouped["calibration"])
        cal_end = max(int(r["time_index"]) for r in grouped["calibration"])
        test_start = min(int(r["time_index"]) for r in grouped["test"])
        middle = (cal_start + cal_end) // 2
        pools = {
            "train_pool": ([r for r in grouped["train_pool"] if int(r["time_index"]) <= train_end - GUARD], ntrain),
            "validation": ([r for r in grouped["calibration"] if cal_start + GUARD <= int(r["time_index"]) <= middle - GUARD], 100),
            "calibration": ([r for r in grouped["calibration"] if middle + GUARD + 1 <= int(r["time_index"]) <= cal_end - GUARD], 100),
            "test": ([r for r in grouped["test"] if int(r["time_index"]) >= test_start + GUARD], 100),
        }
        for split, (pool, count) in pools.items():
            for row in evenly_spaced(pool, count):
                chosen.append({**row, "audit_split": row["split"], "split": split})
    return chosen


def verify(output: Path, old_test: Path | None) -> dict:
    seen: dict[str, str] = {}
    counts: dict[str, dict] = {}
    frame_digests = set()
    for split, name in FILES.items():
        env_counts: Counter = Counter()
        max_force = 0.0
        high_force = 0
        for natoms, comment, species, positions, forces in read_frames(output / name):
            energy = parse_energy(comment)
            lattice = parse_lattice(comment)
            values = [energy, *lattice, *(x for p in positions for x in p), *(x for f in forces for x in f)]
            if energy is None or len(species) != natoms or len(forces) != natoms or any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f"invalid energy, force, cell or positions in {name}")
            digest = frame_hash(species, positions, [lattice[i] for i in (0, 4, 8)])
            if digest in seen:
                raise ValueError(f"duplicate structure: {split} and {seen[digest]}")
            seen[digest] = split
            frame_digests.add(digest)
            force = max(math.sqrt(sum(c*c for c in f)) for f in forces)
            max_force = max(max_force, force)
            high_force += force > 5.0
            env = next(field.split('=', 1)[1] for field in comment.split() if field.startswith('environment='))
            env_counts[env] += 1
        expected = sum(QUOTAS.values()) if split == "train_pool" else 300
        if sum(env_counts.values()) != expected:
            raise ValueError(f"wrong count in {name}: {env_counts}")
        counts[split] = {"frames": expected, "environments": dict(env_counts), "max_force_ev_per_angstrom": max_force, "frames_force_above_5_ev_per_angstrom": high_force}
    previous = Counter()
    if old_test and old_test.exists():
        for _, comment, species, pos, _ in read_frames(old_test):
            lattice = parse_lattice(comment)
            h = frame_hash(species, pos, [lattice[i] for i in (0, 4, 8)])
            if h in seen:
                previous[seen[h]] += 1
    return {"splits": counts, "duplicate_hashes": 0, "hash_definition": "species, positions and cell diagonal rounded to 0.001 angstrom", "prior_pilot_test_exact_hash_overlap": dict(previous)}


def verify_scf(rows: list[dict], root: Path) -> dict:
    """Check the existing curator's frame-to-SCF alignment for each source."""
    result = {}
    for group in sorted({r["group"] for r in rows}):
        blocks = parse_scf_blocks(root / group / f"{group}.out")
        offsets = []
        warnings = 0
        relevant = [r for r in rows if r["group"] == group]
        for row in relevant:
            index = int(row["time_index"])
            if index >= len(blocks):
                raise ValueError(f"{group}: missing SCF block {index}")
            energy, n_warnings = blocks[index]
            offsets.append(float(row["energy"]) - energy)
            warnings += int(n_warnings > 0)
        spread = max(offsets) - min(offsets)
        if warnings or spread > 0.05:
            raise ValueError(f"{group}: {warnings} warned frames; energy offset spread {spread} eV")
        result[group] = {"selected_frames": len(relevant), "scf_blocks": len(blocks), "warned_selected_frames": warnings, "energy_offset_spread_ev": spread}
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--frames", type=Path, default=Path("assets/research/PHASE-1/audit/frames.csv"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--previous-test", type=Path)
    args = p.parse_args()
    with args.frames.open() as f:
        rows = list(csv.DictReader(f))
    chosen = select(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frame_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)
    results = []
    for split, name in FILES.items():
        results.append(build_split(split, [r for r in chosen if r["split"] == split], args.data_root, args.output_dir / name, None))
    qc = verify(args.output_dir, args.previous_test)
    qc["scf_alignment"] = verify_scf(chosen, args.data_root)
    sources = Counter((r["environment"], r["group"], r["file"], r["stride"], r["split"]) for r in chosen)
    summary = {"recipe": "time-blocked subsampling of Phase 1 audit; original files retain their source stride", "time_buffer_steps_each_side": GUARD, "selection": "deterministic evenly spaced indices within each eligible block; output cadence is irregular", "thermalization_cutoff_steps": {"bulk_LiF": 3, "F_doped_Li": 0, "interface": 586}, "outputs": results, "qc": qc, "sources": [{"environment": k[0], "group": k[1], "file": k[2], "source_stride": int(k[3]), "split": k[4], "frames": n} for k, n in sorted(sources.items())], "audit_frames_sha256": hashlib.sha256(args.frames.read_bytes()).hexdigest()}
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
