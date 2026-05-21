#!/usr/bin/env python3
"""
Convert Quantum ESPRESSO pw.x output (.out) to multi-frame extxyz for MACE.
Dataclass-driven workflow (no argparse).
Edit `DEFAULT_CONFIG` below and run the script.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


@dataclass(frozen=True)
class QEToExtXYZConfig:
    """Configuration for converting a QE output file into extxyz."""
    qe_out: Path = field(
        default=Path("qe.out"),
        metadata={"description": "Path to Quantum ESPRESSO pw.x .out file"},
    )
    out: Path = field(
        default=Path("out.extxyz"),
        metadata={"description": "Output multi-frame extxyz path"},
    )
    frame_stride: int = field(
        default=1,
        metadata={"description": "Keep one frame every N parsed frames (>=1)"},
    )
    max_frames: int | None = field(
        default=None,
        metadata={"description": "Optional cap on written frames after striding"},
    )
    include_stress: bool = field(
        default=False,
        metadata={
            "description": "If true, copy QE stress labels into output frames when available"
        },
    )
    config_type: str = field(
        default="Default",
        metadata={
            "description": "Value stored in Atoms.info['config_type'] for each frame"
        },
    )

    @classmethod
    def describe_fields(cls) -> dict[str, str]:
        """Return a field->description map for documentation or logging."""
        return {f.name: f.metadata.get("description", "") for f in fields(cls)}

    def validate(self) -> None:
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be >= 1")


def load_frames_with_ase(cfg: QEToExtXYZConfig) -> list[Atoms]:
    raw = read(str(cfg.qe_out), format="espresso-out", index=":")

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


def convert_qe_out_to_extxyz(cfg: QEToExtXYZConfig) -> int:
    try:
        cfg.validate()
        frames = load_frames_with_ase(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"Conversion error: {exc}")
        return 1

    if not frames:
        print("Error: no labeled frames found in QE output")
        return 1

    write_extxyz(cfg.out, frames)

    print("Field descriptions:")
    for name, desc in QEToExtXYZConfig.describe_fields().items():
        print(f"- {name}: {desc}")
    print(f"Parsed frames: {len(frames)}")
    print(f"Wrote extxyz: {cfg.out}")
    return 0


def main() -> int:
    return convert_qe_out_to_extxyz(QEToExtXYZConfig())


if __name__ == "__main__":
    raise SystemExit(main())
