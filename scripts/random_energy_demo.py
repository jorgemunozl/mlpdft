#!/usr/bin/env python3
"""
Pedagogical random "energy model" demo.

Reads FitSNAP-style JSON configs (like examples/lifbf4/NEWJSON/output_*.json) that already
contain DFT energies/forces/positions, computes a few simple features per frame,
then predicts an energy using random linear weights.

This is NOT physically meaningful; it's for understanding the data pipeline:
structure -> features -> linear model -> energy.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable


def _read_json_allowing_header(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "#":
            _ = f.readline()
        return json.loads(f.read())


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _norm3(v: Iterable[float]) -> float:
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def _sum_inv_r_pairs(positions: list[list[float]], cutoff: float) -> float:
    s = 0.0
    n = len(positions)
    c2 = cutoff * cutoff
    for i in range(n):
        xi, yi, zi = positions[i]
        for j in range(i + 1, n):
            xj, yj, zj = positions[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r2 = dx * dx + dy * dy + dz * dz
            if 1e-24 < r2 <= c2:
                s += 1.0 / math.sqrt(r2)
    return s


@dataclass(frozen=True)
class FrameFeatures:
    n_atoms: float
    volume: float
    sum_force_norms: float
    sum_inv_r: float

    def as_vector(self) -> list[float]:
        return [1.0, self.n_atoms, self.volume, self.sum_force_norms, self.sum_inv_r]


def extract_features(frame: dict, pair_cutoff: float) -> FrameFeatures:
    if "Dataset" in frame:
        frame = frame["Dataset"]
    if "Data" in frame:
        frame = dict(frame)
        frame.update(frame["Data"][0])

    positions = frame["Positions"]
    forces = frame["Forces"]
    lattice = frame["Lattice"]

    n_atoms = float(frame.get("NumAtoms", len(positions)))
    volume = abs(_det3(lattice))
    sum_force_norms = float(sum(_norm3(f) for f in forces))
    sum_inv_r = float(_sum_inv_r_pairs(positions, cutoff=pair_cutoff))

    return FrameFeatures(
        n_atoms=n_atoms,
        volume=volume,
        sum_force_norms=sum_force_norms,
        sum_inv_r=sum_inv_r,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_glob = str(repo_root / "examples/lifbf4/NEWJSON/output_*.json")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        dest="glob_pattern",
        default=default_glob,
        help="Glob for input JSON frames",
    )
    ap.add_argument("--max_frames", type=int, default=20, help="How many frames to read")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for random weights")
    ap.add_argument(
        "--pair_cutoff",
        type=float,
        default=4.0,
        help="Cutoff (same units as Positions) for sum(1/r) pair feature",
    )
    ap.add_argument(
        "--weight_scale",
        type=float,
        default=1.0,
        help="Stddev of random weights (Gaussian)",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob_pattern))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob_pattern!r}")

    paths = paths[: max(args.max_frames, 1)]

    random.seed(args.seed)
    w = [random.gauss(0.0, args.weight_scale) for _ in range(5)]

    print("# random_energy_demo.py")
    print(f"# files={len(paths)} seed={args.seed} pair_cutoff={args.pair_cutoff} weight_scale={args.weight_scale}")
    print(f"# weights(bias,n_atoms,volume,sum|F|,sum1/r)={w}")
    print("file,E_true,E_pred_random,features=[n_atoms,volume,sum|F|,sum1/r]")

    for p in paths:
        obj = _read_json_allowing_header(p)
        ds = obj["Dataset"]
        data0 = ds["Data"][0]
        e_true = float(data0["Energy"])

        feats = extract_features(ds, pair_cutoff=args.pair_cutoff)
        x = feats.as_vector()
        e_pred = sum(wi * xi for wi, xi in zip(w, x))

        rel = os.path.relpath(p, repo_root)
        print(
            f"{rel},{e_true:.6f},{e_pred:.6f},"
            f"[{feats.n_atoms:.0f},{feats.volume:.6f},{feats.sum_force_norms:.6f},{feats.sum_inv_r:.6f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

