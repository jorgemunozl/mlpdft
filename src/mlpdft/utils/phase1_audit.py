#!/usr/bin/env python3
"""Phase 1 audit for the UQ + AL Li-F testbed.

Pure-stdlib reader for extended XYZ. It does not need ASE, MACE, or torch, so it
can run on a read-only data mount and write only the audit artifacts.

What it does
------------
1. Parses every ``*.extxyz`` in the selected groups.
2. Records per-file provenance: stride, frame count, composition, cell,
   ensemble (NVT/NPT inferred from cell drift), energy range, force range.
3. De-duplicates frames by a rounded-position hash, so different strides or
   copies of the same trajectory do not inflate the pool.
4. Groups unique frames into trajectory families by first-frame hash and
   ensemble.
5. Assigns frozen splits per environment: the largest family is split
   contiguously into train/pool and calibration; every other family is locked
   test. Isolated reference cells are excluded from the primary pool.
6. Writes ``provenance.csv``, ``frames.csv``, and ``manifest.json``.

Usage
-----
    python3 src/mlpdft/utils/phase1_audit.py \
        --data-root /path/to/dataset \
        --output-dir assets/research/PHASE-1/audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CORE_GROUPS = [
    "LIF64_KJPAW_V2",
    "LIF64_KJPAW_FINAL",
    "LIF64_KJPAW_NPT",
    "LIF64_KJPAW_NPT_V2",
    "LIF64_KJPAW_NPT_V3",
    "LIF64_KJPAW_NPT_FINAL",
    "LIF64_ISOLATED",
    "LIWITHF_V3",
    "LIWITHF_KJPAW_FINAL",
    "LIWITHF_NPT_FINAL",
    "LIWITHF_ISOLATED",
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_V2",
    "LIFINTERFACE_KJPAW_NPT",
    "LIFINTERFACE_KJPAW_NPT_V2",
    "LIFINTERFACE_KJPAW_FINAL",
    "LIFINTERFACE_KJPAW_NPT_FINAL",
    "BCC_54_KJPAW_FINAL",
    "BCC_54_KJPAW_NPT_FINAL",
]

ENV_BY_COMP = {
    (32, 32): "bulk_LiF",
    (53, 1): "F_doped_Li",
    (86, 36): "interface",
}

LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
ENERGY_RE = re.compile(r"REF_energy=(\S+)")
STRIDE_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_lattice(comment: str) -> list[float]:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError("frame has no Lattice field")
    return [float(value) for value in match.group(1).split()]


def parse_energy(comment: str) -> float | None:
    match = ENERGY_RE.search(comment)
    return float(match.group(1)) if match else None


def read_frames(path: Path):
    """Yield (natoms, comment, species, positions, forces) for each frame."""
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            natoms = int(line)
            comment = handle.readline().strip()
            species: list[str] = []
            positions: list[tuple[float, float, float]] = []
            forces: list[tuple[float, float, float]] = []
            has_forces = "REF_forces" in comment
            for _ in range(natoms):
                parts = handle.readline().split()
                species.append(parts[0])
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if has_forces:
                    forces.append(
                        (float(parts[4]), float(parts[5]), float(parts[6]))
                    )
            yield natoms, comment, species, positions, forces


def frame_hash(species: list[str], positions: list[tuple[float, float, float]], cell: list[float]) -> str:
    payload = ";".join(
        f"{symbol}:{x:.3f},{y:.3f},{z:.3f}"
        for symbol, (x, y, z) in zip(species, positions)
    )
    payload += "|" + ",".join(f"{value:.3f}" for value in cell)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def minimum_image(delta: float, length: float) -> float:
    return delta - length * round(delta / length)


def li_f_coordination(species, positions, cell, cutoff: float = 2.6) -> float:
    li = [i for i, symbol in enumerate(species) if symbol == "Li"]
    f_index = [i for i, symbol in enumerate(species) if symbol == "F"]
    if not li or not f_index:
        return 0.0
    total = 0
    for i in li:
        count = 0
        xi, yi, zi = positions[i]
        for j in f_index:
            xj, yj, zj = positions[j]
            dx = minimum_image(xj - xi, cell[0])
            dy = minimum_image(yj - yi, cell[1])
            dz = minimum_image(zj - zi, cell[2])
            if dx * dx + dy * dy + dz * dz <= cutoff * cutoff:
                count += 1
        total += count
    return total / len(li)


def li_li_nearest_median(species, positions, cell) -> float | None:
    li = [i for i, symbol in enumerate(species) if symbol == "Li"]
    if len(li) < 2:
        return None
    nearest: list[float] = []
    for i in li:
        best = math.inf
        xi, yi, zi = positions[i]
        for j in li:
            if i == j:
                continue
            xj, yj, zj = positions[j]
            dx = minimum_image(xj - xi, cell[0])
            dy = minimum_image(yj - yi, cell[1])
            dz = minimum_image(zj - zi, cell[2])
            distance = dx * dx + dy * dy + dz * dz
            if distance < best:
                best = distance
        nearest.append(math.sqrt(best))
    return statistics.median(nearest)


def scf_warning_rate(path: Path) -> float | None:
    """Fraction of SCF steps with a non-converged diagonalization.

    Streams the log in byte chunks so a 500 MB QE output is cheap to scan.
    Returns None when the log has no SCF markers.
    """
    if not path.is_file():
        return None
    achieved = 0
    not_converged = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            achieved += chunk.count(b"convergence has been achieved")
            not_converged += chunk.count(b"eigenvalues not converged")
    if achieved == 0:
        return None
    return not_converged / achieved


def force_stats(forces) -> tuple[float | None, float | None]:
    if not forces:
        return None, None
    norms = [math.sqrt(fx * fx + fy * fy + fz * fz) for fx, fy, fz in forces]
    return max(norms), sum(norms) / len(norms)


def environment_of(composition: Counter, cell: list[float]) -> str:
    if max(cell) > 20.0:
        return "isolated_reference"
    key = (composition.get("Li", 0), composition.get("F", 0))
    if key in ENV_BY_COMP:
        return ENV_BY_COMP[key]
    if composition.get("F", 0) == 0 and composition.get("Li", 0) > 0:
        return "Li_metal"
    return "other"


def stride_of(name: str) -> int | None:
    match = STRIDE_RE.search(name.rsplit(".", 1)[0])
    return int(match.group(1)) if match else None


def audit_file(path: Path, group: str):
    records = []
    family = None
    cell_values: list[list[float]] = []
    energies = []
    fmax_values = []
    fmean_values = []
    natoms_seen = None
    composition = None
    has_forces = False

    for index, (natoms, comment, species, positions, forces) in enumerate(
        read_frames(path)
    ):
        cell = [parse_lattice(comment)[i] for i in (0, 4, 8)]
        cell_values.append(cell)
        digest = frame_hash(species, positions, cell)
        if family is None:
            family = digest
        energy = parse_energy(comment)
        if energy is not None:
            energies.append(energy)
        fmax, fmean = force_stats(forces)
        if fmax is not None:
            has_forces = True
            fmax_values.append(fmax)
            fmean_values.append(fmean)
        if composition is None:
            composition = Counter(species)
            natoms_seen = natoms
        records.append(
            {
                "group": group,
                "file": path.name,
                "frame_index": index,
                "stride": stride_of(path.name) or 1,
                "time_index": index * (stride_of(path.name) or 1),
                "family": family,
                "natoms": natoms,
                "cell_a": round(cell[0], 4),
                "cell_b": round(cell[1], 4),
                "cell_c": round(cell[2], 4),
                "energy": energy,
                "fmax": fmax,
                "fmean": fmean,
                "frame_hash": digest,
                "species": species,
                "positions": positions,
                "composition": composition,
            }
        )

    cell_drift = 0.0
    if len(cell_values) > 1:
        for axis in range(3):
            values = [cell[axis] for cell in cell_values]
            cell_drift = max(cell_drift, max(values) - min(values))
    ensemble = "NPT" if cell_drift > 0.02 else "NVT"
    for record in records:
        record["ensemble"] = ensemble

    summary = {
        "group": group,
        "file": path.name,
        "stride": stride_of(path.name),
        "ensemble": ensemble,
        "cell_drift_A": round(cell_drift, 4),
        "n_frames": len(records),
        "natoms": natoms_seen,
        "composition": dict(composition) if composition else {},
        "cell_a": round(cell_values[0][0], 4) if cell_values else None,
        "cell_b": round(cell_values[0][1], 4) if cell_values else None,
        "cell_c": round(cell_values[0][2], 4) if cell_values else None,
        "has_forces": has_forces,
        "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
        "fmax_max": max(fmax_values) if fmax_values else None,
        "fmean_mean": (
            sum(fmean_values) / len(fmean_values) if fmean_values else None
        ),
    }
    return summary, records


def equilibration_cutoff(
    records: list[dict],
    n_blocks: int = 16,
    energy_tol: float = 0.02,
    volume_tol: float = 0.005,
) -> int:
    """Return the first time index that is consistent with the final blocks.

    Energy per atom and volume per atom are compared block-wise against the
    median of the last four blocks. A drift larger than the tolerance marks the
    preceding blocks as non-equilibrated. Returns 0 when no cutoff is needed.
    """
    if len(records) < n_blocks * 4:
        return 0
    ordered = sorted(records, key=lambda record: record["time_index"])
    n = len(ordered)
    energies: list[float] = []
    volumes: list[float] = []
    for block in range(n_blocks):
        chunk = ordered[block * n // n_blocks : (block + 1) * n // n_blocks]
        energies.append(
            statistics.mean(record["energy"] / record["natoms"] for record in chunk)
        )
        volumes.append(
            statistics.mean(
                record["cell_a"]
                * record["cell_b"]
                * record["cell_c"]
                / record["natoms"]
                for record in chunk
            )
        )
    reference_energy = statistics.median(energies[-4:])
    reference_volume = statistics.median(volumes[-4:])
    for block in range(n_blocks):
        energy_ok = abs(energies[block] - reference_energy) <= energy_tol
        volume_ok = (
            abs(volumes[block] - reference_volume) / reference_volume <= volume_tol
        )
        if energy_ok and volume_ok:
            return ordered[block * n // n_blocks]["time_index"]
    return 0


def family_quality(rows: list[dict]) -> int:
    """Rank a family by electronic-structure cleanliness: 2 clean, 1 unknown, 0 dirty."""
    rates = [record.get("scf_warning_rate") for record in rows]
    if any(rate is None for rate in rates):
        return 1
    return 2 if all(rate == 0 for rate in rates) else 0


def assign_splits(records: list[dict], cal_frac: float = 0.15) -> dict[str, int]:
    by_environment: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_environment[record["environment"]].append(record)

    cutoffs: dict[str, int] = {}
    for environment, rows in by_environment.items():
        for record in rows:
            record["family_key"] = f"{record['family']}:{record['ensemble']}"
            rate = record.get("scf_warning_rate")
            record["high_scf"] = rate is not None and rate > 0.1
        if environment in {"isolated_reference", "other"}:
            for record in rows:
                record["split"] = "excluded"
            cutoffs[environment] = 0
            continue

        families: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in rows:
            families[(record["family"], record["ensemble"])].append(record)

        ranked = sorted(
            families.items(),
            key=lambda item: (family_quality(item[1]), len(item[1])),
            reverse=True,
        )
        primary_key, primary = ranked[0]
        primary_label = f"{primary_key[0]}:{primary_key[1]}"
        for record in rows:
            record["is_primary_family"] = record["family_key"] == primary_label
        primary.sort(
            key=lambda record: (
                record["time_index"],
                record["file"],
                record["frame_index"],
            )
        )

        # Frames from trajectories with frequent SCF failures are excluded from
        # the training/calibration/test blocks; they carry too much label noise.
        clean_primary = [record for record in primary if not record["high_scf"]]
        for record in primary:
            if record["high_scf"]:
                record["split"] = "excluded_scf"
        if not clean_primary:
            clean_primary = primary
            for record in primary:
                record["split"] = None

        cutoff = equilibration_cutoff(clean_primary)
        analysis = [
            record for record in clean_primary if record["time_index"] >= cutoff
        ]
        if not analysis:
            cutoff = 0
            analysis = clean_primary
        cutoffs[environment] = cutoff
        for record in clean_primary:
            record["equilibration_cutoff"] = cutoff
            if record["time_index"] < cutoff:
                record["split"] = "excluded_equilibration"

        n = len(analysis)
        n_test = max(1, int(round(n * cal_frac))) if n >= 20 else 0
        n_cal = max(1, int(round(n * cal_frac))) if n >= 20 else 0
        n_train = max(0, n - n_cal - n_test)
        for position, record in enumerate(analysis):
            if position < n_train:
                record["split"] = "train_pool"
            elif position < n_train + n_cal:
                record["split"] = "calibration"
            else:
                record["split"] = "test"

        # One independent trajectory is reserved as an OOD test; it is capped
        # by even subsampling so it cannot dominate the locked test.
        for rank, (_, rows_other) in enumerate(ranked[1:]):
            rows_other.sort(
                key=lambda record: (
                    record["time_index"],
                    record["file"],
                    record["frame_index"],
                )
            )
            if rank == 0:
                ood_cutoff = equilibration_cutoff(rows_other)
                ood_analysis = [
                    record
                    for record in rows_other
                    if record["time_index"] >= ood_cutoff
                ]
                if not ood_analysis:
                    ood_cutoff = 0
                    ood_analysis = rows_other
                for record in rows_other:
                    record["equilibration_cutoff"] = ood_cutoff
                    if record["time_index"] < ood_cutoff:
                        record["split"] = "excluded_equilibration"
                stride = max(1, math.ceil(len(ood_analysis) / 1000))
                for position, record in enumerate(ood_analysis):
                    record["split"] = (
                        "ood_test" if position % stride == 0 else "excluded_extra"
                    )
            else:
                for record in rows_other:
                    record["equilibration_cutoff"] = 0
                    record["split"] = "excluded_extra"

    return cutoffs



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root containing <group>/xyz_files/*.extxyz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/research/PHASE-1/audit"),
    )
    parser.add_argument("--groups", nargs="+", default=CORE_GROUPS)
    parser.add_argument("--cal-frac", type=float, default=0.15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_summaries: list[dict] = []
    all_records: list[dict] = []
    missing: list[str] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    group_scf_rates: dict[str, float | None] = {}

    for group in args.groups:
        xyz_dir = args.data_root / group / "xyz_files"
        files = sorted(xyz_dir.glob("*.extxyz")) if xyz_dir.is_dir() else []
        if not files:
            missing.append(group)
            continue
        out_file = next(iter(sorted((args.data_root / group).glob("*.out"))), None)
        scf_rate = scf_warning_rate(out_file) if out_file else None
        group_scf_rates[group] = scf_rate
        # Largest first so the most complete file claims its frames.
        files.sort(key=lambda path: path.stat().st_size, reverse=True)
        for path in files:
            summary, records = audit_file(path, group)
            summary["scf_warning_rate"] = scf_rate
            summary["n_unique_kept"] = 0
            summary["n_duplicates"] = 0
            for record in records:
                digest = record["frame_hash"]
                if digest in seen_hashes:
                    summary["n_duplicates"] += 1
                    duplicate_count += 1
                    continue
                seen_hashes.add(digest)
                summary["n_unique_kept"] += 1
                record["scf_warning_rate"] = scf_rate
                record["environment"] = environment_of(
                    record["composition"], [record["cell_a"], record["cell_b"], record["cell_c"]]
                )
                record["li_f_coord"] = li_f_coordination(
                    record["species"],
                    record["positions"],
                    [record["cell_a"], record["cell_b"], record["cell_c"]],
                )
                record["li_li_nn"] = li_li_nearest_median(
                    record["species"],
                    record["positions"],
                    [record["cell_a"], record["cell_b"], record["cell_c"]],
                )
                del record["species"]
                del record["positions"]
                all_records.append(record)
            file_summaries.append(summary)

    equilibration_cutoffs = assign_splits(all_records, cal_frac=args.cal_frac)

    provenance_path = args.output_dir / "provenance.csv"
    with provenance_path.open("w", newline="") as handle:
        fields = [
            "group",
            "file",
            "stride",
            "ensemble",
            "cell_drift_A",
            "n_frames",
            "n_unique_kept",
            "n_duplicates",
            "natoms",
            "composition",
            "cell_a",
            "cell_b",
            "cell_c",
            "has_forces",
            "energy_min",
            "energy_max",
            "fmax_max",
            "fmean_mean",
            "scf_warning_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in file_summaries:
            row = dict(summary)
            row["composition"] = json.dumps(row["composition"], sort_keys=True)
            writer.writerow({key: row.get(key) for key in fields})

    frames_path = args.output_dir / "frames.csv"
    with frames_path.open("w", newline="") as handle:
        fields = [
            "environment",
            "split",
            "family_key",
            "is_primary_family",
            "group",
            "file",
            "frame_index",
            "stride",
            "time_index",
            "natoms",
            "cell_a",
            "cell_b",
            "cell_c",
            "energy",
            "fmax",
            "fmean",
            "li_f_coord",
            "li_li_nn",
            "scf_warning_rate",
            "high_scf",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            all_records,
            key=lambda item: (
                item["environment"],
                item["split"],
                item["file"],
                item["frame_index"],
            ),
        ):
            writer.writerow({key: record.get(key) for key in fields})

    environment_summary: dict[str, dict] = {}
    for environment in sorted({record["environment"] for record in all_records}):
        rows = [record for record in all_records if record["environment"] == environment]
        families: dict[str, int] = Counter(record["family_key"] for record in rows)
        split_counts: dict[str, int] = Counter(record["split"] for record in rows)
        environment_summary[environment] = {
            "unique_frames": len(rows),
            "families": dict(sorted(families.items(), key=lambda item: -item[1])),
            "split_counts": dict(split_counts),
            "compositions": dict(Counter(
                json.dumps(
                    {
                        "Li": record["composition"].get("Li", 0),
                        "F": record["composition"].get("F", 0),
                    },
                    sort_keys=True,
                )
                for record in rows
            )),
        }

    split_counts: dict[str, int] = Counter(record["split"] for record in all_records)
    family_counts: dict[str, dict] = defaultdict(dict)
    for record in all_records:
        family_counts[record["environment"]].setdefault(
            record["family_key"], {"frames": 0, "splits": Counter()}
        )
        family_counts[record["environment"]][record["family_key"]]["frames"] += 1
        family_counts[record["environment"]][record["family_key"]]["splits"][
            record["split"]
        ] += 1

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "groups_requested": args.groups,
        "groups_missing": missing,
        "n_files": len(file_summaries),
        "frames_raw": sum(summary["n_frames"] for summary in file_summaries),
        "frames_unique": len(all_records),
        "duplicates_removed": duplicate_count,
        "split_counts": dict(split_counts),
        "equilibration_cutoffs": equilibration_cutoffs,
        "group_scf_warning_rates": group_scf_rates,
        "environments": environment_summary,
        "families": {
            environment: {
                key: {"frames": value["frames"], "splits": dict(value["splits"])}
                for key, value in families.items()
            }
            for environment, families in family_counts.items()
        },
        "split_rule": (
            "Energy per atom and volume per atom are checked in 16 blocks "
            "against the last four blocks; earlier blocks outside tolerance "
            "are marked excluded_equilibration. Families are ranked by SCF "
            "cleanliness first, then size; frames from groups with more than "
            "0.1 diagonalization warnings per SCF step are excluded_scf from "
            "the primary blocks. Within each environment, the "
            "largest (family, ensemble) group is then sorted by timestep and "
            "split contiguously into train/pool, calibration, and in-domain "
            f"test ({args.cal_frac} each). The largest non-primary family is "
            "reserved as an independent OOD test, evenly subsampled to at most "
            "1000 frames; remaining families are excluded from the primary "
            "claim. Isolated cells are excluded."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )

    print(json.dumps(
        {
            "files": len(file_summaries),
            "frames_raw": manifest["frames_raw"],
            "frames_unique": manifest["frames_unique"],
            "duplicates_removed": duplicate_count,
            "split_counts": dict(split_counts),
            "missing": missing,
            "output_dir": str(args.output_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
