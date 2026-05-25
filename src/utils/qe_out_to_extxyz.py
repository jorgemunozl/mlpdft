#!/usr/bin/env python3
"""
Convert Quantum ESPRESSO pw.x output (.out) to multi-frame extxyz for MACE.
Dataclass-driven workflow (no argparse).
Edit `DEFAULT_CONFIG` below and run the script.
"""

from __future__ import annotations

from pathlib import Path
from re import M
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from config import MaceConfig
from constants import LIF_KJPAW_GROUP


def load_frames_with_ase(cfg: MaceConfig) -> list[Atoms]:
    raw = read(str(cfg.data_in_path), format="espresso-out", index=":")

    frames = [raw] if isinstance(raw, Atoms) else list(raw)

    if not frames:
        return []

    selected = frames[:: cfg.frame_stride]
    if cfg.max_frames is not None:
        selected = selected[: cfg.max_frames]

    labeled: list[Atoms] = []
    skipped = 0

    for atoms in selected:
        calc = atoms.calc
        results: dict[str, Any] = (
            getattr(calc, "results", {}) if calc is not None else {}
        )

        energy = results.get("energy")
        forces = results.get("forces")

        if energy is None:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:  # noqa: BLE001
                energy = None

        if forces is None:
            try:
                forces = np.asarray(atoms.get_forces())
            except Exception:  # noqa: BLE001
                forces = None

        if energy is None or forces is None:
            skipped += 1
            continue

        atoms = atoms.copy()
        atoms.info["config_type"] = cfg.config_type

        sp_kwargs: dict[str, Any] = {
            "energy": float(energy),
            "forces": np.asarray(forces, dtype=float),
        }

        if cfg.include_stress:
            stress = results.get("stress")
            if stress is not None:
                sp_kwargs["stress"] = np.asarray(stress, dtype=float)

        atoms.calc = SinglePointCalculator(atoms, **sp_kwargs)
        labeled.append(atoms)

    if skipped:
        print(f"Warning: skipped {skipped} frame(s) with missing energy/forces labels")

    return labeled


def write_extxyz(path: Path, frames: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, atoms in enumerate(frames):
        write(path, atoms, format="extxyz", append=i > 0)


def convert_qe_out_to_extxyz(cfg: MaceConfig) -> int:
    try:
        cfg.validate()
        frames = load_frames_with_ase(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"Conversion error: {exc}")
        return 1

    if not frames:
        print("Error: no labeled frames found in QE output")
        return 1

    write_extxyz(cfg.data_out_path, frames)

    print("Field descriptions:")
    for name, desc in MaceConfig.describe_fields().items():
        print(f"- {name}: {desc}")
    print(f"Parsed frames: {len(frames)}")
    print(f"Wrote extxyz: {cfg.data_out_path}")
    return 0


def main() -> int:
    config = MaceConfig(
        group=LIF_KJPAW_GROUP,
        frame_stride=10,
        max_frames=20,
    )
    return convert_qe_out_to_extxyz(config)


if __name__ == "__main__":
    raise SystemExit(main())
