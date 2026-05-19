#!/usr/bin/env python3
"""
Convert FitSNAP-style JSON frames to extended XYZ for MACE training.

Uses the same train/test split rule as FitSNAP with random_sampling=0:
sorted JSON paths, first training_frac -> train, next testing_frac -> test.

Example (LiF64 NEWJSON):
  uv run python scripts/fitsnap_json_to_extxyz.py \\
    --json-dir examples/LiF64_kjpaw_v2/NEWJSON/DEFAULT \\
    --out-dir examples/LiF64_kjpaw_v2/xyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import compute_fitsnap_split, load_json_as_atoms


def _write_xyz(path: Path, frames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, atoms in enumerate(frames):
        write(path, atoms, format="extxyz", append=i > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Directory containing *.json (e.g. .../NEWJSON/DEFAULT)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for train.xyz and test.xyz",
    )
    parser.add_argument(
        "--train-name",
        default="train.xyz",
        help="Output filename for training set (default: train.xyz)",
    )
    parser.add_argument(
        "--test-name",
        default="test.xyz",
        help="Output filename for test set (default: test.xyz)",
    )
    parser.add_argument(
        "--training-frac",
        type=float,
        default=0.8,
        help="Training fraction (default: 0.8, matches LiF64-NEWJSON.in)",
    )
    parser.add_argument(
        "--testing-frac",
        type=float,
        default=0.2,
        help="Testing fraction (default: 0.2)",
    )
    parser.add_argument(
        "--config-type",
        default="Default",
        help="ASE info['config_type'] for MACE weights (default: Default)",
    )
    args = parser.parse_args()

    json_dir = args.json_dir.resolve()
    if not json_dir.is_dir():
        print(f"Error: not a directory: {json_dir}", file=sys.stderr)
        return 1

    all_json: List[Path] = sorted(json_dir.glob("*.json"))
    if not all_json:
        print(f"Error: no *.json in {json_dir}", file=sys.stderr)
        return 1

    train_paths, test_paths = compute_fitsnap_split(
        all_json, args.training_frac, args.testing_frac
    )

    def load_batch(paths: List[Path]) -> list:
        out = []
        for p in paths:
            atoms = load_json_as_atoms(p)
            energy = float(atoms.info["energy_truth"])
            forces = atoms.arrays["forces_truth"]
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            atoms.info["config_type"] = args.config_type
            atoms.info.pop("energy_truth", None)
            atoms.arrays.pop("forces_truth", None)
            out.append(atoms)
        return out

    train_frames = load_batch(train_paths)
    test_frames = load_batch(test_paths)

    out_dir = args.out_dir.resolve()
    train_file = out_dir / args.train_name
    test_file = out_dir / args.test_name

    _write_xyz(train_file, train_frames)
    _write_xyz(test_file, test_frames)

    print(f"Wrote {len(train_frames)} configs -> {train_file}")
    print(f"Wrote {len(test_frames)} configs -> {test_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
