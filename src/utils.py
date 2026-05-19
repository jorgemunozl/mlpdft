"""Shared helpers for FitSNAP JSON, splits, and MACE evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


def read_json_allowing_header(path: str) -> dict:
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
    - Files are processed in sorted order
    - First training_frac files -> training
    - Last testing_frac files -> testing
    """
    n = len(json_paths)
    n_train = int(n * training_frac + 0.5)
    n_test = int(n * testing_frac + 0.5)
    if n_train + n_test > n:
        n_test = n - n_train
    train = json_paths[:n_train]
    test = json_paths[n_train : n_train + n_test]
    return train, test


def load_json_as_atoms(json_path: Path):
    """Load FitSNAP JSON and return ASE Atoms with reference energy/forces."""
    from ase import Atoms

    data = read_json_allowing_header(str(json_path))
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
    atoms.info["energy_truth"] = energy
    atoms.arrays["forces_truth"] = forces
    return atoms


def attach_reference_from_calc(atoms):
    """
    Ensure atoms carry energy_truth / forces_truth for evaluate_mace_on_atoms.

    ASE-extxyz frames usually expose reference data via get_potential_energy /
    get_forces when a SinglePointCalculator is attached.
    """
    if "energy_truth" in atoms.info and "forces_truth" in atoms.arrays:
        return atoms
    if atoms.calc is not None:
        atoms.info["energy_truth"] = float(atoms.get_potential_energy())
        atoms.arrays["forces_truth"] = np.asarray(atoms.get_forces())
        return atoms
    energy = atoms.info.get("energy")
    if energy is None:
        energy = atoms.info.get("free_energy")
    forces = atoms.arrays.get("forces")
    if energy is None or forces is None:
        raise ValueError(
            "Atoms have no calculator and no energy/forces arrays for reference labels"
        )
    atoms.info["energy_truth"] = float(energy)
    atoms.arrays["forces_truth"] = np.asarray(forces)
    return atoms


def evaluate_mace_on_atoms(atoms, calc) -> dict:
    """Run MACE calculator on atoms and return metrics."""
    atoms.calc = calc
    mace_energy = float(atoms.get_potential_energy())
    mace_forces = np.asarray(atoms.get_forces())

    truth_energy = atoms.info["energy_truth"]
    truth_forces = atoms.arrays["forces_truth"]

    dE = mace_energy - truth_energy
    dE_per_atom = dE / len(atoms)

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


def _print_and_write_summary(
    results: list,
    config: MaceEvalConfig,
    *,
    summary_title: str,
    model_label: str,
) -> None:
    if not results:
        print("Error: No successful evaluations", file=sys.stderr)
        raise SystemExit(1)

    dE_values = [r["dE"] for r in results]
    dE_per_atom_values = [r["dE_per_atom"] for r in results]

    mean_offset = np.mean(dE_values)
    dE_values = [dE - mean_offset for dE in dE_values]
    dE_per_atom_values = [
        dE_per_atom - mean_offset for dE_per_atom in dE_per_atom_values
    ]

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

    print("\n" + "=" * 60)
    print(f"MACE Evaluation Summary ({summary_title})")
    print("=" * 60)
    print(f"Model: {model_label}, Device: {config.device}, Dtype: {config.dtype}")
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

    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(config.out_csv, "w", newline="", encoding="utf-8") as f:
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

    print(f"\nResults written to: {config.out_csv}")
