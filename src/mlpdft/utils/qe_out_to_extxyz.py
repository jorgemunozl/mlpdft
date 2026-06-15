#!/usr/bin/env python3
"""
Convert Quantum ESPRESSO pw.x output (.out) to multi-frame extxyz for MACE.
Dataclass-driven workflow (no argparse).
Edit `DEFAULT_CONFIG` below and run the script.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from mlpdft.config import MaceConfig
from mlpdft.constants import ENERGY_KEY, FORCE_KEY, GROUPS_LIF


def load_frames_with_ase(cfg: MaceConfig) -> list[Atoms]:
    # Some QE output files embed binary data alongside text (e.g. from
    # high-precision or restart dumps).  Using latin-1 maps every byte 1:1
    # to a character, preserving the text sections that ASE's parser needs
    # while keeping the binary sections benign (no replacement chars).
    text = cfg.data_in_path.read_text(encoding="latin-1")
    raw = read(StringIO(text), format="espresso-out", index=":")

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

        # Store custom keys directly on atoms (no SinglePointCalculator)
        # SinglePointCalculator only accepts keys in ase's all_properties,
        # so we bypass it and set atoms.info / atoms.arrays directly.
        # The extxyz writer will pick these up when write_results=False.
        atoms.info[ENERGY_KEY] = float(energy)
        atoms.arrays[FORCE_KEY] = np.asarray(forces, dtype=float)

        if cfg.include_stress:
            stress = results.get("stress")
            if stress is not None:
                atoms.info["stress"] = np.asarray(stress, dtype=float)

        labeled.append(atoms)

    if skipped:
        print(f"Warning: skipped {skipped} frame(s) with missing energy/forces labels")

    return labeled


def write_extxyz(path: Path, frames: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, atoms in enumerate(frames):
        # write_results=False: we bypassed SinglePointCalculator so there is
        # nothing to extract from a calculator.  The data is already in
        # atoms.info (REF_energy, config_type, …) and atoms.arrays
        # (REF_forces, …), which the extxyz writer handles natively.
        write(path, atoms, format="extxyz", append=i > 0, write_results=False)


def convert_qe_out_to_extxyz(cfg: MaceConfig) -> None:
    frames: list[Atoms] = []
    try:
        cfg.validate()
        frames = load_frames_with_ase(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"Conversion error: {exc}")

    if not frames:
        print("Error: no labeled frames found in QE output")
        return

    write_extxyz(cfg.data_out_path, frames)

    print("Field descriptions:")
    for name, desc in MaceConfig.describe_fields().items():
        print(f"- {name}: {desc}")
    print(f"Parsed frames: {len(frames)}")
    print(f"Wrote extxyz: {cfg.data_out_path}")


def main() -> None:
    ready = [
        "LIWITHF_V3",
        "LIFINTERFACE_KJPAW_V1",
        "LIFINTERFACE_KJPAW_NPT_V2",
        "LIFINTERFACE_KJPAW_NPT",
    ]
    lack = GROUPS_LIF
    lack = [group for group in lack if group not in ready]
    for group in lack:
        config = MaceConfig(
            group=group,
            frame_stride=5,
            max_frames=None,
        )
        convert_qe_out_to_extxyz(config)


if __name__ == "__main__":
    raise SystemExit(main())
