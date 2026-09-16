#!/usr/bin/env python3
"""
Materialize the toy LiF cell (8 atoms, periodic) as an extxyz file.

Usage:
    python write_toy_cell.py [output_path]

Writes the toy cell and prints its location. Used as a minimal starting
configuration for MD / diffusion sanity checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from mlpdft.constants import TOY_CELL, TOY_CELL_PATH


def build_toy_atoms() -> Atoms:
    """Return the 8-atom rock-salt LiF toy cell in the primitive (perovskite-like) layout."""
    a = TOY_CELL["a"]
    symbols = TOY_CELL["symbols"]
    # Li at the FCC positions, F at the octahedral (edge-center) positions.
    positions = [
        [0, 0, 0],
        [a / 2, a / 2, 0],
        [a / 2, 0, a / 2],
        [0, a / 2, a / 2],
        [a / 2, 0, 0],
        [0, a / 2, 0],
        [0, 0, a / 2],
        [a / 2, a / 2, a / 2],
    ]
    atoms = Atoms(symbols, positions=positions, cell=[a, a, a], pbc=True)
    atoms.info["config_type"] = "toy_lif"
    return atoms


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else TOY_CELL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    atoms = build_toy_atoms()
    write(str(output), atoms)
    print(f"Wrote {len(atoms)}-atom toy LiF cell to: {output}")


if __name__ == "__main__":
    main()
