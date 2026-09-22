"""
Dataset handling: convert Quantum ESPRESSO pw.x output (.out) to labeled
multi-frame extxyz for MACE.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from matplotlib import pyplot as plt

from mlpdft.config import DataSetConfig
from mlpdft.constants import (
    DATA_DIR,
    ENERGY_KEY,
    FORCE_KEY,
    GROUPS,
    TEST_GROUPS,
    TEST_MAX_FRAMES,
    TEST_STRIDE,
    XYZ_DIR,
)


class DataSet:
    """Owns a dataset: parses QE output, selects and labels frames, writes extxyz."""

    def __init__(self, config: DataSetConfig):
        self.config: DataSetConfig = config
        self._raw_frames: list[Atoms] | None = None
        self.REGISTRY: list[str] = GROUPS

    def read_raw_frames(self) -> list[Atoms]:
        """Parse the QE .out file once and cache the raw frame list."""
        if self._raw_frames is not None:
            return self._raw_frames
        path = self.config.data_in_path
        print(f"[DataSet.read_raw_frames] {path}")
        if not path.exists():
            self._raw_frames = []
            return []
        text = path.read_text(encoding="latin-1")
        raw = read(StringIO(text), format="espresso-out", index=":")
        self._raw_frames = [raw] if isinstance(raw, Atoms) else list(raw)
        return self._raw_frames

    def count_frames(self) -> int:
        """Number of raw frames in the QE output."""
        return len(self.read_raw_frames())

    def resolve_paths(self) -> None:
        """Derive ``data_in_path`` /``data_out_path`` from ``group``."""
        cfg = self.config
        if not cfg.config_type:
            cfg.config_type = cfg.group
        cfg.data_in_path = DATA_DIR / cfg.group / Path(str(cfg.group) + ".out")
        if cfg.frame_stride is None:
            cfg.frame_stride = 1
        if cfg.max_frames is None:
            cfg.max_frames = int(len(self.read_raw_frames()) / cfg.frame_stride)

        cfg.data_out_path = (
            DATA_DIR
            / cfg.group
            / XYZ_DIR
            / f"{cfg.group}_{cfg.frame_stride}_{cfg.max_frames}.extxyz"
        )

    def load_labeled_frames(self) -> list[Atoms]:
        """Select frames (equilibration, stride, cap) and attach REF labels."""
        cfg = self.config
        raw = self.read_raw_frames()
        if not raw:
            return []

        if cfg.equilibration_cutoff:
            raw = raw[cfg.equilibration_cutoff :]

        energies = np.array([atoms.get_potential_energy() for atoms in raw])
        plt.plot(range(len(energies)), energies)
        plt.show()
        t0, g, _ = detect_equilibration(energies)
        print(f"Equilibration detected: t0={t0}, g={g}")
        indices = subsample_correlated_data(energies[t0:], g=g)
        print(f"Subsampled indices: {indices}")
        selected = [raw[t0:][i] for i in indices]
        selected = raw[:: (cfg.frame_stride or 1)]

        if cfg.max_frames:  # positive → cap; 0/None → use all
            selected = selected[: cfg.max_frames]

        labeled: list[Atoms] = []
        skipped = 0
        for atoms in selected:
            calc = atoms.calc
            results = getattr(calc, "results", {}) if calc is not None else {}

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
            # Store labels directly on info/arrays (no SinglePointCalculator),
            # so the extxyz writer picks them up natively.
            atoms.info[ENERGY_KEY] = float(energy)
            atoms.arrays[FORCE_KEY] = np.asarray(forces, dtype=float)

            if cfg.include_stress:
                stress = results.get("stress")
                if stress is not None:
                    atoms.info["stress"] = np.asarray(stress, dtype=float)

            labeled.append(atoms)

        if skipped:
            print(
                f"Warning: skipped {skipped} frame(s) with missing energy/forces labels"
            )
        return labeled

    def write_extxyz(self, frames: list[Atoms], path: Path | None = None) -> None:
        """Write labeled frames to an extxyz file (one frame per append)."""
        path = path or self.config.data_out_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for i, atoms in enumerate(frames):
            write(path, atoms, format="extxyz", append=i > 0)

    def validate(self) -> None:
        if self.config.frame_stride is not None and self.config.frame_stride <= 0:
            raise ValueError("frame_stride must be >= 1")

    def convert_qe_out_to_extxyz(self) -> None:
        """Full pipeline: resolve paths → parse → label → write extxyz."""
        self.validate()
        self.resolve_paths()
        frames = self.load_labeled_frames()
        if not frames:
            print("Error: no labeled frames found in QE output")
            return
        self.write_extxyz(frames)
        print(f"Parsed frames: {len(frames)}")
        print(f"Wrote extxyz: {self.config.data_out_path}")

    def get_dataset_numbers(self) -> None:
        """
        Print the number of raw frames for every group in the registry.
        """
        for group in self.REGISTRY:
            ds = DataSet(DataSetConfig(group=group))
            ds.resolve_paths()
            print(f"{group}: {ds.count_frames()}")

    def upload_to_hf(self) -> None:
        """Upload the dataset to Hugging Face."""
        pass


def convert_groups() -> None:
    for group in TEST_GROUPS:
        ds = DataSet(
            DataSetConfig(
                group=group,
                frame_stride=TEST_STRIDE,
                max_frames=TEST_MAX_FRAMES,
            )
        )
        ds.convert_qe_out_to_extxyz()


def main() -> None:
    config = DataSetConfig(
        # group="LIFINTERFACE_KJPAW_V2",
        group="LIF64_KJPAW_NPT",
    )
    ds = DataSet(config)
    ds.get_dataset_numbers()


if __name__ == "__main__":
    config = DataSetConfig()
    ds = DataSet(config)
    ds.get_dataset_numbers()
