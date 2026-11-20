#!/usr/bin/env python3
"""Curate the three-environment UQ dataset: bulk LiF, F-doped Li, interface.

Supersedes ``build_interim_two_environment.py`` for the interface-inclusive
design. It selects from the Phase 1 audit, applies per-environment burn-ins and
a 60-step guard at every split boundary, and reports *achieved versus target*
counts. Deficits are recorded instead of shrinking quotas silently, so the
manifest states exactly how many more frames the final dataset needs.

Usage
-----
    python3 src/mlpdft/utils/build_three_environment_uq_dataset.py \
        --data-root /path/to/dataset \
        --frames assets/research/PHASE-1/audit/frames.csv \
        --output-dir outputs/uq_dataset_three_env

No independence is implied by time-blocked splits or by the provisional
burn-ins; the manifest says so explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

try:  # package import during tests, script import during a build
    from mlpdft.utils.build_interim_uq_dataset import FILES, verify_scf
    from mlpdft.utils.build_phase1_splits import build_split
    from mlpdft.utils.phase1_audit import (
        frame_hash,
        parse_energy,
        parse_lattice,
        read_frames,
    )
except ImportError:  # pragma: no cover
    from build_interim_uq_dataset import FILES, verify_scf
    from build_phase1_splits import build_split
    from phase1_audit import frame_hash, parse_energy, parse_lattice, read_frames

# One source run per environment: mixing runs would confound thermostats and
# QE settings that the audit cannot verify from extxyz alone.
SOURCES = {
    "bulk_LiF": {"group": "LIF64_KJPAW_NPT_V3", "stride": 3, "burn_in": 100},
    "F_doped_Li": {"group": "LIWITHF_V3", "stride": 3, "burn_in": 2400},
    "interface": {"group": "LIFINTERFACE_KJPAW_V2", "stride": 1, "burn_in": 586},
}

GUARD = 60  # MD steps kept clear on each side of every split boundary

# Final design target; achieved counts may be lower until more MD is generated.
TARGET_TRAIN = {"bulk_LiF": 800, "F_doped_Li": 800, "interface": 400}
TARGET_HOLDOUT = 100

DATASET_NAME = "three-environment Li-F UQ dataset"


def _parse_quota(values: list[str]) -> dict[str, int]:
    quota = dict(TARGET_TRAIN)
    for value in values:
        env, _, count = value.partition("=")
        if env not in SOURCES:
            raise SystemExit(f"unknown environment in --train-quota: {env}")
        quota[env] = int(count)
    return quota


def _take_evenly(rows: list[dict], count: int) -> list[dict]:
    """Deterministic evenly spaced subset; returns fewer rows when short."""
    ordered = sorted(
        rows,
        key=lambda r: (int(r["time_index"]), r["file"], int(r["frame_index"])),
    )
    if count <= 0 or not ordered:
        return []
    if len(ordered) <= count:
        return ordered
    return [ordered[(i * (len(ordered) - 1)) // (count - 1)] for i in range(count)]


def _audit_blocks(
    subset: list[dict], cfg: dict
) -> dict[str, list[dict]]:
    """Use the Phase 1 audit's original time blocks."""
    by_split = {
        split: [r for r in subset if r["split"] == split]
        for split in ("train_pool", "calibration", "test")
    }
    missing = [name for name, group in by_split.items() if not group]
    if missing:
        raise SystemExit(f"{cfg['group']}: no audit rows for split(s) {missing}")

    train = by_split["train_pool"]
    train_end = max(int(r["time_index"]) for r in train)
    cal = by_split["calibration"]
    cal_lo = min(int(r["time_index"]) for r in cal)
    cal_hi = max(int(r["time_index"]) for r in cal)
    middle = (cal_lo + cal_hi) // 2
    test = by_split["test"]
    test_start = min(int(r["time_index"]) for r in test)

    return {
        "train_pool": [
            r for r in train if cfg["burn_in"] <= int(r["time_index"]) <= train_end - GUARD
        ],
        "validation": [
            r for r in cal if cal_lo + GUARD <= int(r["time_index"]) <= middle - GUARD
        ],
        "calibration": [
            r for r in cal if middle + GUARD + 1 <= int(r["time_index"]) <= cal_hi - GUARD
        ],
        "test": [r for r in test if int(r["time_index"]) >= test_start + GUARD],
    }


def _need_blocks(
    subset: list[dict], cfg: dict, train_target: int, holdout_target: int
) -> dict[str, list[dict]]:
    """Re-block the usable trajectory to fit the targets when it is long enough.

    Each block reserves ``GUARD`` steps on both sides, so adjacent blocks are
    separated by 2*GUARD. When the trajectory is shorter than the total demand,
    every block shrinks proportionally and the shortfall is reported.
    """
    times = sorted({int(r["time_index"]) for r in subset})
    lo = max(cfg["burn_in"], times[0])
    hi = times[-1]
    span = hi - lo + 1
    stride = cfg["stride"]
    targets = [train_target, holdout_target, holdout_target, holdout_target]
    # Widths are in MD steps: one frame costs ``stride`` steps, and each block
    # keeps GUARD steps clear at both ends.
    needed = [t * stride + 2 * GUARD for t in targets]
    total = sum(needed)
    if span < total:
        scale = span / total
        widths = [
            max((2 * GUARD + 1) * stride, int(round(width * scale))) for width in needed
        ]
    else:
        widths = needed

    order = ["train_pool", "validation", "calibration", "test"]
    chosen: dict[str, list[dict]] = {}
    cursor = lo
    for split, width, target in zip(order, widths, targets):
        block_lo, block_hi = cursor, cursor + width - 1
        pool = [
            r for r in subset if block_lo + GUARD <= int(r["time_index"]) <= block_hi - GUARD
        ]
        achievable = max(0, (width - 2 * GUARD) // stride)
        chosen[split] = _take_evenly(pool, min(target, achievable))
        cursor = block_hi + 1
    return chosen


def eligible_blocks(
    rows: list[dict],
    train_quota: dict[str, int],
    holdout_quota: int,
    block_mode: str,
) -> dict[str, dict[str, list[dict]]]:
    """Return the eligible frame pool per environment and split."""
    blocks: dict[str, dict[str, list[dict]]] = {}
    for env, cfg in SOURCES.items():
        subset = [
            r
            for r in rows
            if r["environment"] == env
            and r["group"] == cfg["group"]
            and int(r["stride"]) == cfg["stride"]
        ]
        if not subset:
            raise SystemExit(f"{env}: no audit rows for {cfg['group']}")
        if block_mode == "audit":
            blocks[env] = _audit_blocks(subset, cfg)
        elif block_mode == "need":
            blocks[env] = _need_blocks(
                subset, cfg, train_quota[env], holdout_quota
            )
        else:  # pragma: no cover - argparse constrained
            raise SystemExit(f"unknown block mode: {block_mode}")
    return blocks


def select(
    rows: list[dict],
    train_quota: dict[str, int],
    holdout_quota: int,
    block_mode: str = "audit",
) -> tuple[list[dict], dict]:
    """Choose frames and return (chosen, achieved-vs-target report)."""
    blocks = eligible_blocks(rows, train_quota, holdout_quota, block_mode)
    chosen: list[dict] = []
    report: dict[str, dict] = {}
    targets = {
        "train_pool": train_quota,
        "validation": {env: holdout_quota for env in SOURCES},
        "calibration": {env: holdout_quota for env in SOURCES},
        "test": {env: holdout_quota for env in SOURCES},
    }
    for env in SOURCES:
        report[env] = {}
        for split, target_by_env in targets.items():
            target = target_by_env[env]
            pool = blocks[env][split]
            taken = _take_evenly(pool, target)
            for row in taken:
                chosen.append({**row, "audit_split": row["split"], "split": split})
            report[env][split] = {
                "target": target,
                "eligible": len(pool),
                "achieved": len(taken),
                "deficit": max(0, target - len(taken)),
            }
    return chosen, report


def autocorrelation(values: np.ndarray) -> dict | None:
    """Normalized ACF with first-nonpositive truncation, on an energy proxy.

    Selected frames are irregularly spaced, so this is a triage diagnostic,
    not a correlation time for the source trajectory.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(axis=0, keepdims=True)
    n = len(x)
    var = float(np.mean(x * x))
    if n < 8 or var <= 1e-30:
        return None
    length = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(x, n=length, axis=0)
    sums = np.fft.irfft(fft.conj() * fft, n=length, axis=0)[:n].mean(axis=1)
    acf = sums / (np.arange(n, 0, -1) * var)
    max_lag = n // 4
    stop = next((i for i in range(1, max_lag + 1) if acf[i] <= 0), max_lag + 1)
    tau = max(1.0, float(1 + 2 * np.sum(acf[1:stop])))
    return {
        "lag1": float(acf[1]),
        "tau_frames": tau,
        "tau_md_steps": SOURCES["interface"]["stride"] * tau,
        "ess": n / tau,
        "truncation_lag": stop - 1,
        "reached_window_limit": stop == max_lag + 1,
    }


def temporal_diagnostics(chosen: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for env in SOURCES:
        result[env] = {}
        stride = SOURCES[env]["stride"]
        for split in FILES:
            rows = [r for r in chosen if r["environment"] == env and r["split"] == split]
            if not rows:
                continue
            rows.sort(key=lambda r: int(r["time_index"]))
            energy = np.array(
                [float(r["energy"]) / int(r["natoms"]) for r in rows], dtype=float
            )
            acf = autocorrelation(energy)
            steps = np.diff([int(r["time_index"]) for r in rows])
            result[env][split] = {
                "frames": len(rows),
                "source_stride_steps": stride,
                "selected_step_gap_median": float(np.median(steps)) if len(steps) else None,
                "energy_acf": acf,
                "note": "selected-frame ACF; irregular spacing; triage only",
            }
    return result


def qc(chosen: list[dict], output: Path, achieved: dict) -> dict:
    identities: set[tuple[str, str, int]] = set()
    for row in chosen:
        ident = (row["group"], row["file"], int(row["frame_index"]))
        if ident in identities:
            raise ValueError(f"reused source frame: {ident}")
        identities.add(ident)

    seen: dict[str, str] = {}
    results: dict[str, dict] = {}
    for split, name in FILES.items():
        counts: Counter = Counter()
        max_force = 0.0
        force_over = 0
        for natoms, comment, species, positions, forces in read_frames(output / name):
            lattice = parse_lattice(comment)
            energy = parse_energy(comment)
            values = [
                energy,
                *lattice,
                *(v for point in positions for v in point),
                *(v for force in forces for v in force),
            ]
            if (
                energy is None
                or len(forces) != natoms
                or len(species) != natoms
                or any(v is None or not math.isfinite(v) for v in values)
            ):
                raise ValueError(f"invalid labels in {name}")
            if np.linalg.det(np.array(lattice).reshape(3, 3)) <= 0:
                raise ValueError(f"invalid cell in {name}")
            digest = frame_hash(species, positions, [lattice[i] for i in (0, 4, 8)])
            if digest in seen:
                raise ValueError(f"duplicate structure: {split} / {seen[digest]}")
            seen[digest] = split
            norm = max(math.sqrt(sum(c * c for c in force)) for force in forces)
            max_force = max(max_force, norm)
            force_over += norm > 5.0
            env = next(
                field.split("=", 1)[1]
                for field in comment.split()
                if field.startswith("environment=")
            )
            counts[env] += 1
        expected = {
            env: achieved[env][split]["achieved"] for env in SOURCES
        }
        if dict(counts) != expected:
            raise ValueError(f"{name}: environment counts {dict(counts)} != {expected}")
        results[split] = {
            "frames": sum(counts.values()),
            "environments": dict(counts),
            "maximum_force_ev_per_angstrom": max_force,
            "frames_over_5_ev_per_angstrom": force_over,
        }
    return {
        "splits": results,
        "duplicate_source_frames": 0,
        "duplicate_rounded_geometry": 0,
        "hash_precision_angstrom": 0.001,
    }


def gate_report(
    achieved: dict, qc_result: dict, scf: dict, temporal: dict, train_quota: dict
) -> list[dict]:
    gates: list[dict] = []

    shortfalls = [
        f"{env}/{split}: {entry['achieved']}/{entry['target']}"
        for env in achieved
        for split, entry in achieved[env].items()
        if entry["deficit"] > 0
    ]
    gates.append(
        {
            "gate": "target frame counts",
            "pass": not shortfalls,
            "detail": "all targets met" if not shortfalls else "; ".join(shortfalls),
        }
    )

    scf_warned = sum(v["warned_selected_frames"] for v in scf.values())
    scf_spread = max(v["energy_offset_spread_ev"] for v in scf.values())
    gates.append(
        {
            "gate": "SCF warnings zero and offset spread < 0.05 eV",
            "pass": scf_warned == 0 and scf_spread < 0.05,
            "detail": f"warned={scf_warned}, max offset spread={scf_spread:.2e} eV",
        }
    )

    over = sum(v["frames_over_5_ev_per_angstrom"] for v in qc_result["splits"].values())
    gates.append(
        {
            "gate": "no frame above 5 eV/angstrom",
            "pass": over == 0,
            "detail": f"{over} frames above 5 eV/angstrom",
        }
    )

    lag_limited = [
        f"{env}/{split}"
        for env in temporal
        for split, entry in temporal[env].items()
        if entry.get("energy_acf") and entry["energy_acf"]["reached_window_limit"]
    ]
    recorded = all(
        temporal.get(env, {}).get(split, {}).get("energy_acf")
        for env in temporal
        for split in FILES
    )
    gates.append(
        {
            "gate": "temporal dependence characterised",
            "pass": recorded,
            "detail": (
                "ACF recorded for every split"
                if not lag_limited
                else "ACF recorded; lag limit reached in " + ", ".join(lag_limited)
            ),
        }
    )

    gates.append(
        {
            "gate": "burn-in recorded per environment",
            "pass": True,
            "detail": ", ".join(
                f"{env}={cfg['burn_in']}" for env, cfg in SOURCES.items()
            ),
        }
    )
    gates.append(
        {
            "gate": "calibration separate from validation and test",
            "pass": True,
            "detail": "disjoint time blocks with a 60-step guard",
        }
    )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("assets/research/PHASE-1/audit/frames.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-quota",
        nargs="+",
        default=[f"{env}={n}" for env, n in TARGET_TRAIN.items()],
        help="Per-environment training target, e.g. bulk_LiF=800",
    )
    parser.add_argument("--holdout-quota", type=int, default=TARGET_HOLDOUT)
    parser.add_argument(
        "--block-mode",
        choices=("audit", "need"),
        default="audit",
        help=(
            "audit: keep the Phase 1 audit's time blocks; need: re-block the "
            "usable trajectory so every split can meet its target"
        ),
    )
    args = parser.parse_args()

    train_quota = _parse_quota(args.train_quota)
    rows = list(csv.DictReader(args.frames.open()))
    chosen, achieved = select(rows, train_quota, args.holdout_quota, args.block_mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frame_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)

    outputs = [
        build_split(
            split,
            [row for row in chosen if row["split"] == split],
            args.data_root,
            args.output_dir / name,
            None,
        )
        for split, name in FILES.items()
    ]
    temporal = temporal_diagnostics(chosen)
    checks = qc(chosen, args.output_dir, achieved)
    checks["scf_alignment"] = verify_scf(chosen, args.data_root)
    gates = gate_report(
        achieved, checks, checks["scf_alignment"], temporal, train_quota
    )

    summary = {
        "dataset": DATASET_NAME,
        "status": "provisional until every gate passes",
        "block_mode": args.block_mode,
        "sources": SOURCES,
        "target_train_frames": train_quota,
        "target_holdout_frames": args.holdout_quota,
        "achieved": achieved,
        "shortfall_frames": {
            env: sum(entry["deficit"] for entry in achieved[env].values())
            for env in achieved
        },
        "guard_steps_each_side": GUARD,
        "independence_claim": False,
        "outputs": outputs,
        "qc": checks,
        "gates": gates,
        "audit_sha256": hashlib.sha256(args.frames.read_bytes()).hexdigest(),
    }
    (args.output_dir / "temporal_diagnostics.json").write_text(
        json.dumps(temporal, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    print(json.dumps({"achieved": achieved, "gates": gates}, indent=2))
    failed = [gate for gate in gates if not gate["pass"]]
    print(f"\n{len(failed)} gate(s) failing" if failed else "\nall gates pass")


if __name__ == "__main__":
    main()
