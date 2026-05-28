"""
Molecular dynamics with a MACE potential.

Reads a single initial configuration from the dataset (first frame),
assigns Maxwell–Boltzmann velocities, and runs MD via ASE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import ase.io
import numpy as np
from ase import units
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from config import MaceConfig
from constants import LIF_KJPAW_GROUP
from mace_scrap import MACE_SCRAP


def run_md(
    config: MaceConfig,
    *,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    n_steps: int = 10_000,
    trajectory_interval: int = 100,
    log_interval: int = 100,
    thermostat: Literal["langevin", "nose-hoover", "velocity-verlet"] = "langevin",
    friction: float = 0.01,  # (1/fs) for Langevin
    trajectory_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    remove_translation: bool = True,
    remove_rotation: bool = True,
    rng_seed: Optional[int] = None,
) -> None:
    """
    Run molecular dynamics from the first frame of the dataset.

    Parameters
    ----------
    config:
        MaceConfig pointing to the model and dataset to use.
    temperature_K:
        Target temperature (Kelvin).
    timestep_fs:
        Integration timestep in femtoseconds.
    n_steps:
        Total number of MD steps to run.
    trajectory_interval:
        Save a frame every N steps.
    log_interval:
        Print log message every N steps.
    thermostat:
        Which thermostat to use.
    friction:
        Langevin friction coefficient (1/fs). Only used for ``langevin``.
    trajectory_path:
        Output path for the trajectory extxyz file.
        Default: auto-generated next to ``config.model_output``.
    log_path:
        Output path for the MD log. Default: auto-generated.
    remove_translation:
        Whether to shift velocities so total momentum is zero.
    remove_rotation:
        Whether to remove overall angular momentum.
    rng_seed:
        Random seed for velocity initialisation (repeatable runs).
    """
    # ---------- build calculator ----------
    scrap = MACE_SCRAP(config=config)
    calc = scrap.build_calculator()

    # ---------- initial structure (first frame) ----------
    atoms = ase.io.read(config.data_out_path, index=0)
    atoms.calc = calc

    # ---------- initial velocities ----------
    rng = np.random.RandomState(rng_seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    if remove_translation:
        Stationary(atoms)  # zero total momentum
    if remove_rotation:
        ZeroRotation(atoms)  # zero angular momentum

    # ---------- output paths ----------
    if trajectory_path is None:
        stem = (
            f"md_{config.model_key}_{config.group}"
            f"_T{temperature_K:.0f}K"
            f"_dt{timestep_fs}fs"
            f"_N{n_steps}"
        )
        trajectory_path = config.model_output.parent / f"{stem}.extxyz"
    if log_path is None:
        log_path = trajectory_path.with_suffix(".log")

    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- MD integrator ----------
    timestep_ase = timestep_fs * units.fs  # ASE internal time unit

    if thermostat == "langevin":
        dyn = Langevin(
            atoms,
            timestep=timestep_ase,
            temperature_K=temperature_K,
            friction=friction / units.fs,
        )
    elif thermostat == "nose-hoover":
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_ase,
            temperature_K=temperature_K,
        )
    elif thermostat == "velocity-verlet":
        dyn = VelocityVerlet(atoms, timestep=timestep_ase)
    else:
        raise ValueError(f"Unknown thermostat: {thermostat}")

    # ---------- attach observers ----------
    traj = Trajectory(str(trajectory_path), mode="w", atoms=atoms)
    dyn.attach(traj.write, interval=trajectory_interval)

    dyn.attach(
        MDLogger(dyn, atoms, str(log_path), header=True),
        interval=log_interval,
    )

    # ---------- run ----------
    print(
        f"Running MD:  {thermostat}  |  "
        f"T = {temperature_K} K  |  dt = {timestep_fs} fs  |  "
        f"{n_steps} steps"
    )
    print(f"Trajectory  -> {trajectory_path}")
    print(f"Log         -> {log_path}")
    dyn.run(n_steps)

    # Write final snapshot as well
    final_path = trajectory_path.with_suffix(".final.extxyz")
    ase.io.write(str(final_path), atoms, format="extxyz")
    print(f"Final frame -> {final_path}")


def main() -> None:
    config = MaceConfig(
        model_key="0-omat-medium",
        group=LIF_KJPAW_GROUP,
        frame_stride=10,
        max_frames=20,
    )
    run_md(
        config,
        temperature_K=300.0,
        timestep_fs=1.0,
        n_steps=10_000,
        trajectory_interval=100,
        log_interval=100,
        thermostat="langevin",
        friction=0.01,
        rng_seed=42,
    )


if __name__ == "__main__":
    main()
