"""
Molecular dynamics with a MACE potential.

Reads an initial configuration, assigns Maxwell-Boltzmann velocities, and runs
constant-temperature MD via ASE (Langevin / Nose-Hoover) or NVE (Velocity-Verlet).
Used for active learning and transport-property (activation-energy) studies.
"""

from __future__ import annotations

import dataclasses
import hashlib
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
from mlpdft.constants import DATA_DIR, TOY_CELL_PATH
from mlpdft.mace_scrap import MaceScrap


def unwrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Unwrap periodic positions so the MSD grows linearly at long times.

    ``positions`` has shape (n_frames, n_atoms, 3); ``cell`` is the 3x3 box.
    """
    inv = np.linalg.inv(cell)
    frac = positions @ inv  # fractional coordinates
    dfrac = np.diff(frac, axis=0)
    dfrac -= np.round(dfrac)  # minimum-image displacement
    unwrapped_frac = np.concatenate(
        [frac[:1], frac[:1] + np.cumsum(dfrac, axis=0)], axis=0
    )
    return unwrapped_frac @ cell


def msd_multi_origin(unwrapped: np.ndarray, species_mask: np.ndarray) -> np.ndarray:
    """Mean-squared displacement averaged over atoms and time origins."""
    r = unwrapped[:, species_mask, :]  # (n_frames, n_species, 3)
    n_frames = len(r)
    max_lag = int(0.5 * n_frames)
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag):
        d = r[lag:] - r[:-lag]  # (n_frames - lag, n_species, 3)
        msd[lag] = float(np.mean(np.sum(d**2, axis=2)))
    return msd


class MolecularDynamics:
    """Run one MD trajectory with a MACE calculator."""

    def __init__(self, config: MolecularDynamicsConfig):
        self.config: MolecularDynamicsConfig = config

    def _resolve_paths(self) -> tuple[Path, Path]:
        """Return (trajectory_path, log_path), deriving sensible defaults."""
        cfg = self.config
        if cfg.trajectory_path is not None:
            traj = Path(cfg.trajectory_path)
        else:
            # Include a config-fingerprint so different initial_configs that
            # share the same group do not overwrite each other when run in
            # parallel.
            tag = hashlib.sha1(str(cfg.initial_config).encode()).hexdigest()[:8]
            stem = (
                f"md_{cfg.model_key}_{cfg.group or 'cfg'}"
                f"_T{cfg.temperature_K:.0f}K"
                f"_dt{cfg.timestep:g}fs"
                f"_N{cfg.nsteps}_{tag}"
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

    def run_md(self) -> None:
        """Run a single trajectory from ``config.initial_config``."""
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

    def run_md_many(
        self,
        initial_configs: list[str],
        workers: int = 1,
    ) -> None:
        """Run one trajectory per initial configuration, optionally in parallel.

        Each entry in ``initial_configs`` reuses this object's ``config`` as a
        template and gets a distinct ``rng_seed``, so trajectories are
        independent and their output filenames do not collide.

        With ``workers > 1`` the trajectories run in separate processes (one
        per config) -- each trajectory itself is sequential, but independent
        runs are launched concurrently. On a single GPU, set ``workers`` to the
        number of runs that fit in memory; otherwise distribute across devices.
        """
        base = self.config
        configs: list[MolecularDynamicsConfig] = []
        for i, initial in enumerate(initial_configs):
            cfg = dataclasses.replace(base, initial_config=initial)
            cfg.rng_seed = base.rng_seed + i
            configs.append(cfg)

        def _run_one(cfg: MolecularDynamicsConfig) -> None:
            MolecularDynamics(cfg).run_md()

        if workers <= 1:
            for cfg in configs:
                _run_one(cfg)
            return

        from multiprocessing import get_context

        ctx = get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            pool.map(_run_one, configs)


class ActiveLearning(MolecularDynamics):
    """
    Wrapper from MACE active learning
    """

    def __init__(self, config: MolecularDynamicsConfig):
        super().__init__(config)
        self.config: MolecularDynamicsConfig = config

    def run(self) -> None:
        """
        I guess that is possible run Quantum Expresso on the fly.
        """
        super().run_md()


def run_md_from_config() -> None:
    """Example entry point (single bulk-LiF frame)."""

    config = MolecularDynamicsConfig(
        model_key="mace_omat_medium",
        initial_config=str(
            DATA_DIR / "LIF64_ISOLATED" / "xyz_files" / "LIF64_ISOLATED_3_150.extxyz"
        ),
        temperature_K=300.0,
        timestep=1.0,
        nsteps=10,
        thermostat="langevin",
        friction=0.01,
        rng_seed=42,
        trajectory_interval=1,
        log_interval=1,
    )
    MolecularDynamics(config).run_md()


if __name__ == "__main__":
    run_md_from_config()
