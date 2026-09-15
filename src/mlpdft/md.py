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
from mlpdft.constants import DATA_DIR, K_B
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
    """Run an MD trajectory with a MACE calculator."""

    def __init__(self, config: MolecularDynamicsConfig):
        self.config: MolecularDynamicsConfig = config
        self.atoms: Atoms

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

    def diffusion_coefficient(self) -> float:
        pos = np.asarray(positions)
        times_fs = np.arange(len(pos)) * save_interval * timestep
        unwrapped = unwrap_positions(pos, np.asarray(atoms.cell))
        msd = msd_multi_origin(unwrapped, species_mask)

        i0 = max(1, int(self.config.fit_range[0] * len(msd)))
        i1 = min(len(msd), int(self.config.fit_range[1] * len(msd)))

        if i1 - i0 < 2:
            raise ValueError("linear-fit window too small")
        slope, _ = np.polyfit(times_fs[i0:i1], msd[i0:i1], 1)
        D = slope / 6.0 * 0.1  # Å^2/fs -> cm^2/s
        return D


    def activation_energy(self, Ds: list[float], Ts: list[float]) -> tuple[float, float]:
        """From a set of tuples (D,T), fit a linear model to estimate activation energy."""
        Ts = np.asarray(Ts)
        invT = 1.0 / (K_B * Ts)  # 1/eV
        lnD = np.log(np.asarray(Ds))
        slope, intercept = np.polyfit(invT, lnD, 1)
        Ea = -slope  # eV
        D0 = float(np.exp(intercept))
        return Ea, D0

class ActiveLearning(MolecularDynamics):
    """
    Wrapper from MACE active learning
    """
    def __init__(self, config: ActiveLearningConfig):
        super().__init__(config)
        self.config = ActiveLearningConfig

    def run(self) -> None:
        """
        I guess that is possible run Quantum Expresso on the fly.
        """
        super().run()

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
