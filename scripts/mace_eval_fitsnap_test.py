#!/usr/bin/env python3
"""
Evaluate MACE-MP on FitSNAP-style test set from LiBF4 JSON data.

Reads FitSNAP perconfig.dat (or computes train/test split like FitSNAP with random_sampling=0),
runs MACE-MP on test configurations, and compares to DFT energies/forces.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


def _read_json_allowing_header(path: str) -> dict:
    """Read JSON, skipping comment header if present."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "#":
            _ = f.readline()
        return json.loads(f.read())


@dataclass(frozen=True)
class ConfigRow:
    filename: str
    group: str
    natoms: int
    energy_truth: float
    energy_pred: Optional[float]
    testing_bool: bool


def parse_perconfig(path: Path) -> List[ConfigRow]:
    """Parse perconfig.dat and return list of ConfigRow."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=" ")
        for raw in reader:
            # Handle possible extra whitespace
            row = {k.strip(): v.strip() for k, v in raw.items() if k.strip()}
            rows.append(
                ConfigRow(
                    filename=row["Filename"],
                    group=row["Group"],
                    natoms=int(row["Natoms"]),
                    energy_truth=float(row["Energy_Truth"]),
                    energy_pred=float(row["Energy_Pred"]) if row.get("Energy_Pred") else None,
                    testing_bool=row["Testing_Bool"].lower() == "true",
                )
            )
    return rows


def compute_fitsnap_split(
    json_paths: List[Path], training_frac: float = 0.8, testing_frac: float = 0.2
) -> tuple[List[Path], List[Path]]:
    """
    Mimic FitSNAP random_sampling=0 behavior:
    - Files are processed in sorted order (portable alternative to os.listdir)
    - First training_frac files -> training
    - Last testing_frac files -> testing
    """
    n = len(json_paths)
    n_train = int(n * training_frac + 0.5)
    n_test = int(n * testing_frac + 0.5)
    # Ensure we don't exceed total
    if n_train + n_test > n:
        n_test = n - n_train
    train = json_paths[:n_train]
    test = json_paths[n_train : n_train + n_test]
    return train, test


def load_json_as_atoms(json_path: Path):
    """Load FitSNAP JSON and return ASE Atoms object."""
    from ase import Atoms

    data = _read_json_allowing_header(str(json_path))
    if "Dataset" in data:
        data = data["Dataset"]
    if "Data" in data:
        frame = dict(data)
        frame.update(data["Data"][0])
    else:
        frame = data

    positions = np.asarray(frame["Positions"])
    lattice = np.asarray(frame["Lattice"])
    symbols = frame["AtomTypes"]
    energy = float(frame["Energy"])
    forces = np.asarray(frame["Forces"])

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=lattice,
        pbc=True,
    )
    # Store reference data
    atoms.info["energy_truth"] = energy
    atoms.arrays["forces_truth"] = forces
    return atoms


def evaluate_mace_on_atoms(atoms, calc) -> dict:
    """Run MACE calculator on atoms and return metrics."""
    atoms.calc = calc
    mace_energy = float(atoms.get_potential_energy())
    mace_forces = np.asarray(atoms.get_forces())

    truth_energy = atoms.info["energy_truth"]
    truth_forces = atoms.arrays["forces_truth"]

    # Energy error
    dE = mace_energy - truth_energy
    dE_per_atom = dE / len(atoms)

    # Force errors
    df = mace_forces - truth_forces
    force_rmse = np.sqrt(np.mean(df**2))
    force_max = np.max(np.linalg.norm(df, axis=1))
    force_mae = np.mean(np.linalg.norm(df, axis=1))

    return {
        "mace_energy": mace_energy,
        "truth_energy": truth_energy,
        "dE": dE,
        "dE_per_atom": dE_per_atom,
        "force_rmse": force_rmse,
        "force_max": force_max,
        "force_mae": force_mae,
        "natoms": len(atoms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate MACE-MP on LiBF4 test set from FitSNAP JSON"
    )
    parser.add_argument(
        "--perconfig",
        type=Path,
        help="Path to perconfig.dat from FitSNAP (optional). If not provided, computes split directly from JSON files",
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=None,
        help="Root directory containing JSON files. Default: examples/lifbf4/NEWJSON",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="DEFAULT",
        help="Group name for path resolution (default: DEFAULT)",
    )
    parser.add_argument(
        "--training-frac",
        type=float,
        default=0.8,
        help="Training fraction when computing split (default: 0.8)",
    )
    parser.add_argument(
        "--testing-frac",
        type=float,
        default=0.2,
        help="Testing fraction when computing split (default: 0.2)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="MACE-MP model key (default: small)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Default dtype (default: float32)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/mace_lifbf4_test_results.csv"),
        help="Output CSV path (default: outputs/mace_lifbf4_test_results.csv)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (for testing)",
    )
    args = parser.parse_args()

    # Resolve JSON root
    repo_root = Path(__file__).resolve().parent.parent
    if args.json_root is None:
        json_root = repo_root / "examples" / "lifbf4" / "NEWJSON"
    else:
        json_root = Path(args.json_root)

    if not json_root.exists():
        print(f"Error: JSON root not found: {json_root}", file=sys.stderr)
        return 1

    # Get test files
    test_files: List[Path] = []

    if args.perconfig:
        # Use perconfig.dat
        print(f"Reading perconfig from: {args.perconfig}")
        rows = parse_perconfig(args.perconfig)
        test_rows = [r for r in rows if r.testing_bool]
        print(f"Found {len(rows)} total configs, {len(test_rows)} test configs")

        for row in test_rows:
            # Resolve path: json_root / group / filename
            json_path = json_root / row.group / row.filename
            if not json_path.exists():
                # Try without group subdirectory
                json_path = json_root / row.filename
            if json_path.exists():
                test_files.append(json_path)
            else:
                print(f"Warning: JSON not found: {json_path}", file=sys.stderr)
    else:
        # Compute split from directory
        print(f"Computing train/test split from: {json_root}")
        all_json = sorted(json_root.glob("*.json"))
        if not all_json:
            print(f"Error: No JSON files found in {json_root}", file=sys.stderr)
            return 1

        train_files, test_files = compute_fitsnap_split(
            all_json, args.training_frac, args.testing_frac
        )
        print(f"Total: {len(all_json)}, Train: {len(train_files)}, Test: {len(test_files)}")

    if not test_files:
        print("Error: No test files to process", file=sys.stderr)
        return 1

    if args.max_frames:
        test_files = test_files[: args.max_frames]
        print(f"Limited to {len(test_files)} test frames")

    # Initialize MACE
    print(f"Loading MACE-MP model: {args.model} (device={args.device}, dtype={args.dtype})")
    from mace.calculators import mace_mp

    calc = mace_mp(model=args.model, device=args.device, default_dtype=args.dtype)

    # Process test files
    print(f"Processing {len(test_files)} test configurations...")
    results = []

    for i, json_path in enumerate(test_files):
        try:
            atoms = load_json_as_atoms(json_path)
            metrics = evaluate_mace_on_atoms(atoms, calc)
            metrics["filename"] = json_path.name
            metrics["filepath"] = str(json_path)
            results.append(metrics)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processed {i + 1}/{len(test_files)}: {json_path.name}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}", file=sys.stderr)
            continue

    if not results:
        print("Error: No successful evaluations", file=sys.stderr)
        return 1

    # Compute aggregate statistics
    dE_values = [r["dE"] for r in results]
    dE_per_atom_values = [r["dE_per_atom"] for r in results]
    force_rmse_values = [r["force_rmse"] for r in results]
    force_max_values = [r["force_max"] for r in results]
    force_mae_values = [r["force_mae"] for r in results]

    summary = {
        "n_configs": len(results),
        "energy_mae": np.mean(np.abs(dE_values)),
        "energy_rmse": np.sqrt(np.mean(np.square(dE_values))),
        "energy_std": np.std(dE_values),
        "energy_per_atom_mae": np.mean(np.abs(dE_per_atom_values)),
        "energy_per_atom_rmse": np.sqrt(np.mean(np.square(dE_per_atom_values))),
        "force_rmse_mean": np.mean(force_rmse_values),
        "force_rmse_std": np.std(force_rmse_values),
        "force_mae_mean": np.mean(force_mae_values),
        "force_max_mean": np.mean(force_max_values),
        "force_max_std": np.std(force_max_values),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("MACE-MP Evaluation Summary (LiBF4 Test Set)")
    print("=" * 60)
    print(f"Model: {args.model}, Device: {args.device}, Dtype: {args.dtype}")
    print(f"Test configurations: {summary['n_configs']}")
    print()
    print("Energy Errors (eV):")
    print(f"  MAE:  {summary['energy_mae']:.6f}")
    print(f"  RMSE: {summary['energy_rmse']:.6f}")
    print(f"  Std:  {summary['energy_std']:.6f}")
    print()
    print("Energy Errors per Atom (eV/atom):")
    print(f"  MAE:  {summary['energy_per_atom_mae']:.6f}")
    print(f"  RMSE: {summary['energy_per_atom_rmse']:.6f}")
    print()
    print("Force Errors (eV/Å):")
    print(f"  RMSE (mean over configs): {summary['force_rmse_mean']:.6f}")
    print(f"  RMSE (std over configs):  {summary['force_rmse_std']:.6f}")
    print(f"  MAE (mean):               {summary['force_mae_mean']:.6f}")
    print(f"  Max |ΔF| (mean):          {summary['force_max_mean']:.6f}")
    print(f"  Max |ΔF| (std):           {summary['force_max_std']:.6f}")
    print("=" * 60)

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "natoms",
                "truth_energy",
                "mace_energy",
                "dE",
                "dE_per_atom",
                "force_rmse",
                "force_mae",
                "force_max",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nResults written to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
