#!/usr/bin/env python3
"""Extract labelled frames from a Quantum ESPRESSO pw.x MD log into extxyz.

Works for fixed-cell (``md``) and variable-cell (``vc-md``) logs and streams the
file, so multi-GB logs fit in memory.

Pairing rule (the same as ASE's ``read_espresso_out``): a configuration owns
the results printed after it and before the next configuration. QE prints the
new ``ATOMIC_POSITIONS`` (and, for vc-md, ``CELL_PARAMETERS``) after each MD
step, so SCF block k (k >= 1) pairs with the positions printed in MD
iteration k. Block 0 uses the input structure from the header and is never
exported (it is always inside the burn-in).

Labels: ``REF_energy`` is the ``!`` total energy in eV (the free energy for
smeared runs, consistent with the forces); ``REF_forces`` is in eV/Å. Units
use QE's CODATA 2006 constants. Checked against ASE's reader on fixed- and
variable-cell logs: identical positions, cells, energies and forces, while a
one-block shift changes the forces by 0.05-4.5 eV/Å.

Usage
-----
    PYTHONPATH=src python3 src/mlpdft/utils/extract_qe_trajectory.py \
        --out dataset/LIF64_KJPAW_FINAL/LIF64_KJPAW_FINAL.out \
        --env bulk_LiF --burn-in 500 --stride 10 --stop 15860 \
        --output train_pool.extxyz
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

# QE's own constants (CODATA 2006, as in its constants.f90 and in ASE's
# espresso reader), so labels match QE's conversions and earlier extxyz files.
RY_TO_EV = 13.605691930242388
BOHR_TO_A = 0.52917720859
NUM = r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)"
SYMBOL = re.compile(r"^[A-Z][a-z]?\d*$")


def iter_blocks(path: Path) -> Iterator[dict]:
    """Yield every complete SCF block with the configuration it belongs to."""
    alat = None
    axes: list[list[float]] = []
    cell: np.ndarray | None = None
    positions: list[list[float]] | None = None
    symbols: list[str] | None = None
    block = -1
    current: dict | None = None
    with path.open(errors="ignore") as fh:
        lines = iter(fh)
        for line in lines:
            if alat is None and "lattice parameter (alat)" in line:
                alat = float(re.search(NUM, line).group(1)) * BOHR_TO_A
            elif cell is None and re.match(r"\s+a\(\d\) = \(", line):
                axes.append([float(v) for v in re.findall(NUM, line)])
                if len(axes) == 3:
                    cell = np.asarray(axes) * alat
            elif line.startswith("CELL_PARAMETERS"):
                if "angstrom" not in line.lower():
                    raise ValueError(f"{path}: only CELL_PARAMETERS (angstrom) is supported")
                cell = np.array([[float(v) for v in next(lines).split()[:3]] for _ in range(3)])
            elif line.startswith("ATOMIC_POSITIONS"):
                if "angstrom" not in line.lower():
                    raise ValueError(f"{path}: only ATOMIC_POSITIONS (angstrom) is supported")
                new_symbols, new_positions = [], []
                for row in lines:
                    parts = row.split()
                    if len(parts) < 4 or not SYMBOL.match(parts[0]):
                        break
                    try:
                        new_positions.append([float(v) for v in parts[1:4]])
                    except ValueError:
                        break
                    new_symbols.append(parts[0])
                symbols, positions = new_symbols, new_positions
            elif line.startswith("!") and "total energy" in line:
                if current is not None and "forces" in current:
                    yield current
                block += 1
                current = {
                    "block": block,
                    "energy": float(re.search(NUM, line).group(1)) * RY_TO_EV,
                    "symbols": symbols,
                    "positions": None if positions is None else np.asarray(positions),
                    "cell": None if cell is None else cell.copy(),
                }
            elif current is not None and "Forces acting on atoms" in line and "forces" not in current:
                forces = []
                for row in lines:
                    if "force =" in row and row.split()[0] == "atom":
                        forces.append([float(v) for v in row.split("=", 1)[1].split()[:3]])
                    elif forces and not row.strip():
                        break
                current["forces"] = np.asarray(forces) * RY_TO_EV / BOHR_TO_A
            elif current is not None and "P=" in line and "stress" in line:
                current["pressure_kbar"] = float(line.split("P=")[1].split()[0])
    if current is not None and "forces" in current:
        yield current


def extract(path: Path, *, env: str, run: str, burn_in: int, stride: int,
            start: int | None = None, stop: int | None = None) -> list[Atoms]:
    """Frames of SCF blocks k with max(burn_in, start, 1) <= k < stop and (k - first) % stride == 0."""
    first = max(burn_in, start or 0, 1)
    frames = []
    for blk in iter_blocks(path):
        k = blk["block"]
        if k < first or (stop is not None and k >= stop) or (k - first) % stride:
            continue
        if blk["positions"] is None or len(blk["positions"]) != len(blk["forces"]):
            raise ValueError(f"{path}: block {k} has {len(blk['forces'])} forces but no matching positions")
        atoms = Atoms(symbols=blk["symbols"], positions=blk["positions"], cell=blk["cell"], pbc=True)
        atoms.info.update({"REF_energy": blk["energy"], "config_type": env, "run": run, "scf_step": k})
        if "pressure_kbar" in blk:
            atoms.info["pressure_kbar"] = blk["pressure_kbar"]
        atoms.arrays["REF_forces"] = blk["forces"]
        frames.append(atoms)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="pw.x MD log")
    ap.add_argument("--env", required=True, help="config_type label")
    ap.add_argument("--run", default=None, help="run label (default: log file stem)")
    ap.add_argument("--burn-in", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--stop", type=int, default=None, help="first SCF block NOT included")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    frames = extract(args.out, env=args.env, run=args.run or args.out.stem, burn_in=args.burn_in,
                     stride=args.stride, start=args.start, stop=args.stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, frames, format="extxyz")
    steps = [a.info["scf_step"] for a in frames]
    print(json.dumps({"output": str(args.output), "frames": len(frames),
                      "first_step": steps[0] if steps else None, "last_step": steps[-1] if steps else None}))


if __name__ == "__main__":
    main()
