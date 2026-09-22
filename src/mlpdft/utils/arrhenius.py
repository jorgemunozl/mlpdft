#!/usr/bin/env python3
"""
Compute the Arrhenius activation energy E_a for diffusion (single command).

Runs NVT (Langevin) MD with a MACE calculator at several temperatures,
extracts D at each temperature, and fits ln(D) vs 1/(k_B T) to get E_a.

Writes, under outputs/diffusion/ by default:
    msd_<T>K.csv     MSD(t) at each temperature
    arrhenius.csv    T (K) and D (cm^2/s)
    summary.txt      D values, E_a, D_0

Usage:
    python arrhenius.py            # defaults: cuda, 300-600 K
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

from ase import units  # noqa: E402
from ase.io import read  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from diffusion import (  # noqa: E402
    msd_diagnostics,
    msd_multi_origin,
    unwrap_positions,
)
from mace.calculators.mace import MACECalculator  # noqa: E402

from mlpdft.constants import DATA_DIR, OUTPUTS_DIR  # noqa: E402

DEFAULT_MODEL = str(OUTPUTS_DIR / "mace_omat_medium" / "mace_omat_medium.model")
# Dense, periodic bulk LiF (cell ~8.16 A, ~8.5 A^3/atom). Do NOT use the
# LIF64_ISOLATED config: it is a 64-atom cluster in a 28 A vacuum box, so it
# measures rigid-body cluster motion instead of bulk Li diffusion.
DEFAULT_CONFIG = str(
    DATA_DIR / "LIF64_KJPAW_V2" / "xyz_files" / "LIF64_KJPAW_V2_10_200.extxyz"
)
DEFAULT_OUTPUT = OUTPUTS_DIR / "diffusion"
K_B = 8.617333262e-5  # eV / K


def diffusion_cm2_per_s(
    atoms,
    species,
    temperature,
    timestep,
    friction,
    equil_steps,
    steps,
    save_interval,
    seed,
    fit_start,
    fit_end,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Run one NVT trajectory; return (D_cm2_s, times_fs, msd, mean_T_K)."""
    np.random.seed(seed)
    species_mask = np.asarray(atoms.get_chemical_symbols()) == species
    if int(species_mask.sum()) == 0:
        raise ValueError(f"species '{species}' not found in configuration")

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=friction,
    )
    dyn.run(equil_steps)

    positions: list[np.ndarray] = []
    temperatures: list[float] = []

    def save(dyn=None) -> None:
        positions.append(dyn.atoms.get_positions().copy())
        temperatures.append(dyn.atoms.get_temperature())

    dyn.attach(save, interval=save_interval, dyn=dyn)
    dyn.run(steps)

    pos = np.asarray(positions)
    times_fs = np.arange(len(pos)) * save_interval * timestep
    masses = atoms.get_masses()
    unwrapped = unwrap_positions(pos, np.asarray(atoms.cell))
    msd = msd_multi_origin(unwrapped, species_mask, masses=masses)

    i0 = max(1, int(fit_start * len(msd)))
    i1 = min(len(msd), int(fit_end * len(msd)))
    if i1 - i0 < 2:
        raise ValueError("linear-fit window too small; adjust --fit-start/--fit-end")
    slope, _ = np.polyfit(times_fs[i0:i1], msd[i0:i1], 1)
    D = slope / 6.0 * 0.1  # Å^2/fs -> cm^2/s
    mean_T = float(np.mean(temperatures)) if temperatures else float(temperature)
    return D, times_fs, msd, mean_T


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arrhenius activation energy from MACE MD."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--species", type=str, default="Li")
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[300, 400, 500, 600]
    )
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--equil-steps", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--fit-start", type=float, default=0.2)
    parser.add_argument("--fit-end", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    calc = MACECalculator(
        model_paths=[args.model], device=args.device, default_dtype=args.dtype
    )

    Ds: list[float] = []
    Ts_meas: list[float] = []
    lines = ["# T_set_K  T_meas_K  D_cm2_per_s", ""]
    print("=" * 60)
    for T in args.temperatures:
        atoms = read(args.config, index=args.config_index)  # fresh structure
        atoms.calc = calc
        D, times_fs, msd, mean_T = diffusion_cm2_per_s(
            atoms,
            args.species,
            T,
            args.timestep,
            args.friction,
            args.equil_steps,
            args.steps,
            args.save_interval,
            args.seed,
            args.fit_start,
            args.fit_end,
        )
        Ds.append(D)
        Ts_meas.append(mean_T)
        lines.append(f"{T:g}  {mean_T:.2f}  {D:.6e}")
        np.savetxt(
            args.output / f"msd_{T:g}K.csv",
            np.column_stack([times_fs, msd]),
            header="time_fs  msd_ang2",
            comments="# ",
        )
        print(f"  T set = {T:>4.0f} K   T obs = {mean_T:6.1f} K   D = {D:.6e} cm^2/s")
        for w in msd_diagnostics(times_fs, msd, (args.fit_start, args.fit_end)):
            print(f"    [warn] {w}")
    print("=" * 60)

    # Fit ln D vs 1/(k_B T) using the measured (not requested) temperature.
    Ts = np.asarray(Ts_meas)
    invT = 1.0 / (K_B * Ts)  # 1/eV
    lnD = np.log(np.asarray(Ds))
    slope, intercept = np.polyfit(invT, lnD, 1)
    Ea = -slope  # eV
    D0 = float(np.exp(intercept))

    np.savetxt(
        args.output / "arrhenius.csv",
        np.column_stack([np.asarray(args.temperatures), Ts, np.asarray(Ds)]),
        header="T_set_K T_meas_K D_cm2_per_s",
        comments="# ",
    )

    lines += ["", f"E_a = {Ea:.4f} eV", f"D_0 = {D0:.4e} cm^2/s"]
    (args.output / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"  E_a = {Ea:.4f} eV")
    print(f"  D_0 = {D0:.4e} cm^2/s")
    if Ea < 0.1:
        print(
            f"  [warn] E_a = {Ea:.3f} eV is far below typical solid-state "
            + "barriers (0.3-0.8 eV); the system may be liquid-like or drifting"
        )
    if max(Ds) > 1e-3:
        print(
            "  [warn] D is orders of magnitude above solid-state diffusion; "
            + "verify the cell is dense and periodic"
        )
    print(f"  Results written to: {args.output}")


if __name__ == "__main__":
    main()
