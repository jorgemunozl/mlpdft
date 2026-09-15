"""
Molecular dynamics with a MACE potential.

Reads an initial configuration, assigns Maxwell-Boltzmann velocities, and runs
constant-temperature MD via ASE (Langevin / Nose-Hoover) or NVE (Velocity-Verlet).
Used for active learning and transport-property (activation-energy) studies.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import ase.io
import numpy as np
from ase import Atoms, units
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

from mlpdft.config import MolecularDynamicsConfig
from mlpdft.constants import DATA_DIR
from mlpdft.mace_scrap import MaceScrap


class MolecularDynamics:
    """Run an MD trajectory with a MACE calculator."""

    def __init__(self, config: MolecularDynamicsConfig):
        self.config: MolecularDynamicsConfig = config

    # ── internal helpers ──────────────────────────────────────────
    def _resolve_paths(self) -> tuple[Path, Path]:
        """Return (trajectory_path, log_path), deriving sensible defaults."""
        cfg = self.config
        if cfg.trajectory_path is not None:
            traj = Path(cfg.trajectory_path)
        else:
            stem = (
                f"md_{cfg.model_key}_{cfg.group or 'cfg'}"
                f"_T{cfg.temperature_K:.0f}K"
                f"_dt{cfg.timestep}fs"
                f"_N{cfg.nsteps}"
            )
            traj = (cfg.model_output.parent / stem).with_suffix(".extxyz")

        log = (
            Path(cfg.log_path) if cfg.log_path is not None else traj.with_suffix(".log")
        )
        return traj, log

    def _build_integrator(self, atoms: Atoms, timestep_ase: float):
        """Instantiate the requested ASE dynamics object."""
        cfg = self.config
        thermostat = cfg.thermostat
        if thermostat == "langevin":
            # ASE wants friction in inverse ASE-time units; config gives 1/fs.
            return Langevin(
                atoms,
                timestep=timestep_ase,
                temperature_K=cfg.temperature_K,
                friction=cfg.friction / units.fs,
            )
        if thermostat == "nose-hoover":
            return NoseHooverChainNVT(
                atoms,
                timestep=timestep_ase,
                temperature_K=cfg.temperature_K,
                tdamp=cfg.tdamp * units.fs,
            )
        if thermostat == "velocity-verlet":
            return VelocityVerlet(atoms, timestep=timestep_ase)
        raise ValueError(f"Unknown thermostat: {thermostat!r}")

    # ── main entry ────────────────────────────────────────────────
    def run(self) -> None:
        cfg = self.config

        calc = MaceScrap(config=cfg).build_calculator()
        atoms = cast(Atoms, ase.io.read(cfg.initial_config, index=0))
        atoms.calc = calc

        # Initial velocities + remove rigid-body motion.
        rng = np.random.RandomState(cfg.rng_seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature_K, rng=rng)
        if cfg.remove_translation:
            Stationary(atoms)
        if cfg.remove_rotation:
            ZeroRotation(atoms)

        traj_path, log_path = self._resolve_paths()
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        # Start fresh each run (append would silently mix old+new frames).
        traj_path.unlink(missing_ok=True)

        timestep_ase = cfg.timestep * units.fs
        dyn = self._build_integrator(atoms, timestep_ase)

        def _write_frame() -> None:
            ase.io.write(str(traj_path), atoms, append=True)

        dyn.attach(_write_frame, interval=cfg.trajectory_interval)
        dyn.attach(
            MDLogger(dyn, atoms, str(log_path), header=True),
            interval=cfg.log_interval,
        )

        print(
            f"Running MD: {cfg.thermostat} | T = {cfg.temperature_K} K | dt = {cfg.timestep} fs | {cfg.nsteps} steps"
        )
        print(f"Trajectory  -> {traj_path}")
        print(f"Log         -> {log_path}")
        _ = dyn.run(cfg.nsteps)

        # Final snapshot.
        final_path = traj_path.with_name(traj_path.stem + ".final" + traj_path.suffix)
        ase.io.write(str(final_path), atoms, format="extxyz")
        print(f"Final frame -> {final_path}")


def run_md_from_config() -> None:
    """Example entry point (uses the bulk-LiF dataset frame)."""
    config = MolecularDynamicsConfig(
        model_key="mace_omat_medium",
        initial_config=str(
            DATA_DIR / "LIF64_ISOLATED" / "xyz_files" / "LIF64_ISOLATED_3_150.extxyz"
        ),
        temperature_K=300.0,
        timestep=1.0,
        nsteps=10_000,
        thermostat="langevin",
        friction=0.01,
        rng_seed=42,
        trajectory_interval=100,
        log_interval=100,
    )
    MolecularDynamics(config).run()


if __name__ == "__main__":
    run_md_from_config()
