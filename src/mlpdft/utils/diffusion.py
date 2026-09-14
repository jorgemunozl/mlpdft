#!/usr/bin/env python3
"""
Compute the self-diffusion coefficient from MACE NVT molecular dynamics.

Runs an NVT (Langevin) trajectory with a MACE calculator, unwraps positions
across periodic boundaries, computes the mean-squared displacement (MSD) of a
target species, and fits the Einstein relation to extract the diffusion
coefficient D.

Usage:
    python diffusion.py --species Li --temperature 400

Default model: MACE-MP-0 OMAT medium foundation model.
Default structure: LIF64_ISOLATED bulk LiF (periodic, 64 atoms).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ── CPU-only workaround (e3nn JIT) ────────────────────────────────
import torch as _torch

_jit_load_original = _torch.jit.load


def _jit_load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _jit_load_original(*args, **kwargs)


_torch.jit.load = _jit_load_cpu  # type: ignore[assignment]
# ──────────────────────────────────────────────────────────────────

from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators.mace import MACECalculator

from mlpdft.constants import DATA_DIR, OUTPUTS_DIR  # noqa: E402

DEFAULT_MODEL = str(OUTPUTS_DIR / "mace_omat_medium" / "mace_omat_medium.model")
DEFAULT_CONFIG = str(
    DATA_DIR / "LIF64_ISOLATED" / "xyz_files" / "LIF64_ISOLATED_3_150.extxyz"
)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute D from MACE MD via the Einstein relation."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="MACE .model path"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG, help="Starting XYZ"
    )
    parser.add_argument("--config-index", type=int, default=0, help="Frame index")
    parser.add_argument("--species", type=str, default="Li", help="Species to track")
    parser.add_argument(
        "--temperature", type=float, default=400, help="Temperature (K)"
    )
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep (fs)")
    parser.add_argument(
        "--friction", type=float, default=0.01, help="Langevin friction (fs^-1)"
    )
    parser.add_argument(
        "--equil-steps", type=int, default=1000, help="Equilibration steps (discarded)"
    )
    parser.add_argument("--steps", type=int, default=5000, help="Production steps")
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save every N steps"
    )
    parser.add_argument(
        "--fit-start", type=float, default=0.2, help="Start of linear fit (fraction)"
    )
    parser.add_argument(
        "--fit-end", type=float, default=0.5, help="End of linear fit (fraction)"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional CSV for MSD(t)"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    atoms = read(args.config, index=args.config_index)
    atoms.calc = MACECalculator(
        model_paths=[args.model], device=args.device, default_dtype=args.dtype
    )
    atoms.get_forces()  # verify the calculator runs

    species_mask = np.asarray(atoms.get_chemical_symbols()) == args.species
    n_species = int(species_mask.sum())
    if n_species == 0:
        parser.error(f"species '{args.species}' not found in the configuration")

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        friction=args.friction,
    )

    # Equilibration (not saved)
    dyn.run(args.equil_steps)

    # Production: record positions
    positions: list[np.ndarray] = []

    def save(dyn=None) -> None:
        positions.append(dyn.atoms.get_positions().copy())

    dyn.attach(save, interval=args.save_interval, dyn=dyn)
    dyn.run(args.steps)

    pos = np.asarray(positions)
    times_fs = np.arange(len(pos)) * args.save_interval * args.timestep
    cell = np.asarray(atoms.cell)

    unwrapped = unwrap_positions(pos, cell)
    msd = msd_multi_origin(unwrapped, species_mask)

    if len(msd) < 10:
        parser.error(
            f"Only {len(msd)} MSD points available ({len(positions)} saved frames); "
            "increase --steps or decrease --save-interval."
        )

    i0 = max(1, int(args.fit_start * len(msd)))
    i1 = min(len(msd), int(args.fit_end * len(msd)))
    if i1 - i0 < 2:
        parser.error("Linear-fit window is too small; adjust --fit-start / --fit-end.")
    slope, _ = np.polyfit(times_fs[i0:i1], msd[i0:i1], 1)

    # Einstein relation in 3D: D = slope / 6  (slope in Å^2/fs)
    d_ang2_per_fs = slope / 6.0
    d_cm2_per_s = d_ang2_per_fs * 0.1  # 1 Å^2/fs = 0.1 cm^2/s

    print("=" * 70)
    print(f"  Model        : {args.model}")
    print(f"  Structure    : {args.config} (index {args.config_index})")
    print(f"  Species      : {args.species} ({n_species} atoms)")
    print(f"  Temperature  : {args.temperature} K")
    print(f"  Frames saved : {len(positions)}")
    print("=" * 70)
    print(f"  MSD slope    : {slope:.6e} Å^2/fs")
    print(f"  D            : {d_ang2_per_fs:.6e} Å^2/fs")
    print(f"                {d_cm2_per_s:.6e} cm^2/s")
    print("=" * 70)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output,
            np.column_stack([times_fs, msd]),
            header="time_fs  msd_ang2",
            comments="# ",
        )
        print(f"  Wrote MSD(t) to: {args.output}")


if __name__ == "__main__":
    main()
