#!/usr/bin/env python3
"""Run a MACE-MP foundation model on the last frame of a Quantum ESPRESSO pw.x output."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from ase.io import read


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate MACE-MP on the last configuration in a QE .out file."
    )
    parser.add_argument(
        "--qe-out",
        required=True,
        help="Path to pw.x output (e.g. LiF64_kjpaw.out)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="MACE-MP model key or path/URL to a .model file (default: small)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for MACE",
    )
    parser.add_argument(
        "--default-dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point dtype for MACE",
    )
    args = parser.parse_args()
    qe_path = args.qe_out

    try:
        atoms = read(qe_path, format="espresso-out", index=-1)
    except Exception as exc:  # noqa: BLE001 — surface ASE parse errors clearly
        print(f"Failed to read QE output as espresso-out: {exc}", file=sys.stderr)
        return 1

    qe_calc = atoms.calc
    qe_energy = qe_calc.results.get("energy") if qe_calc is not None else None
    qe_forces = qe_calc.results.get("forces") if qe_calc is not None else None

    from mace.calculators import mace_mp

    atoms.calc = mace_mp(
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
    )
    mace_energy = float(atoms.get_potential_energy())
    mace_forces = np.asarray(atoms.get_forces())

    print(f"QE file: {qe_path}")
    print(f"MACE model: {args.model!r}  device={args.device!r}  dtype={args.default_dtype!r}")
    print(f"MACE energy (eV): {mace_energy:.8f}")
    print(f"MACE max |F| (eV/Å): {np.max(np.linalg.norm(mace_forces, axis=1)):.8f}")

    if qe_energy is not None:
        print(f"QE   energy (eV, from OUT): {float(qe_energy):.8f}")
    if qe_forces is not None:
        qf = np.asarray(qe_forces)
        df = mace_forces - qf
        print(
            "QE vs MACE forces: "
            f"RMSE={np.sqrt(np.mean(df**2)):.6e} eV/Å, "
            f"max|ΔF|={np.max(np.linalg.norm(df, axis=1)):.6e} eV/Å"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
