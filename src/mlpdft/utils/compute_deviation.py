#!/usr/bin/env python3
"""
Compute the committee deviation (uncertainty) for a single configuration.

Runs single-point inference with a list of MACE models and reports the
per-atom force standard deviation across the committee, the energy standard
deviation, and the relative force error used by MACE's active-learning MD
stopping criterion.

Usage:
    python compute_deviation.py <config.extxyz> [--models m1.model m2.model ...]

Defaults (both under outputs/):
    models = committee_s123, committee_s124, committee_s125
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ── CPU-only workaround ───────────────────────────────────────────
# e3nn model pickles call torch.jit.load(buffer) without map_location,
# which initialises a CUDA context even when we want CPU. Patch to default
# to CPU so the models load on GPU-less machines.
import torch as _torch

_jit_load_original = _torch.jit.load


def _jit_load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _jit_load_original(*args, **kwargs)


_torch.jit.load = _jit_load_cpu  # type: ignore[assignment]
# ──────────────────────────────────────────────────────────────────

from ase.io import read, write  # noqa: E402
from mace.calculators.mace import MACECalculator  # noqa: E402

from mlpdft.constants import OUTPUTS_DIR  # noqa: E402

# Matches the regularisation in mace/cli/active_learning_md.py:stop_error.
REG = 0.2

DEFAULT_MODELS = [
    str(OUTPUTS_DIR / "committee_s123" / "models" / "committee_s123.model"),
    str(OUTPUTS_DIR / "committee_s124" / "models" / "committee_s124.model"),
    str(OUTPUTS_DIR / "committee_s125" / "models" / "committee_s125.model"),
]


def load_calculator(models: list[str], device: str, dtype: str) -> MACECalculator:
    """Build the committee calculator (load models once)."""
    return MACECalculator(model_paths=models, device=device, default_dtype=dtype)


def evaluate_deviation(atoms, calc: MACECalculator, num_models: int) -> dict:
    """Run inference for one frame and return committee-deviation statistics."""
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    forces_comm = calc.results["forces_comm"]  # (n_models, n_atoms, 3)
    energy_var = calc.results["energy_var"]

    force_var = np.var(forces_comm, axis=0)  # (n_atoms, 3)
    per_atom_force_std = np.sqrt(np.sum(force_var, axis=1))  # (n_atoms,)
    ferr_rel = per_atom_force_std / (np.linalg.norm(forces, axis=1) + REG)

    return {
        "num_models": num_models,
        "energy": energy,
        "energy_std": float(np.sqrt(energy_var)),
        "mean_force_std": float(np.mean(per_atom_force_std)),
        "max_force_std": float(np.max(per_atom_force_std)),
        "max_ferr_rel": float(np.max(ferr_rel)),
        "per_atom_force_std": per_atom_force_std,
        "ferr_rel": ferr_rel,
    }


def compute_deviation(
    config: str,
    models: list[str],
    config_index: int = -1,
    device: str = "cpu",
    dtype: str = "float64",
) -> dict:
    """Run inference on a single frame and return the deviation summary."""
    atoms = read(config, index=config_index)
    calc = load_calculator(models, device, dtype)
    stats = evaluate_deviation(atoms, calc, len(models))
    stats["atoms"] = atoms
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute committee deviation for a single configuration."
    )
    parser.add_argument("config", type=Path, help="Path to an XYZ configuration file")
    parser.add_argument(
        "--config_index",
        type=int,
        default=-1,
        help="Frame index in the XYZ file (-1 = last frame)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model paths (default: the three committee seeds)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Model dtype (your fine-tuned models are float64)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional extxyz path to write the frame with per-atom deviation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute deviation for every frame in the file (batch mode)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"config not found: {args.config}")
    for model_path in args.models:
        if not Path(model_path).exists():
            parser.error(f"model not found: {model_path}")

    if args.all:
        frames = read(str(args.config), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        calc = load_calculator(args.models, args.device, args.dtype)
        print(
            f"{'#':>3} {'group':<24} {'atoms':>5} "
            f"{'E std (eV)':>12} {'F std mean':>12} {'F std max':>12} {'ferr_rel':>9}"
        )
        print("-" * 83)
        rows = []
        for i, atoms in enumerate(frames):
            stats = evaluate_deviation(atoms, calc, len(args.models))
            group = atoms.info.get("group", "?")
            rows.append((i, group, len(atoms), stats))
            print(
                f"{i:>3} {group:<24} {len(atoms):>5} "
                f"{stats['energy_std']:>12.3e} {stats['mean_force_std']:>12.3e} "
                f"{stats['max_force_std']:>12.3e} {stats['max_ferr_rel']:>9.4f}"
            )
        print("-" * 83)
        if args.output:
            out_frames = []
            for i, group, _n, stats in rows:
                atoms = frames[i].copy()
                atoms.arrays["mlff_forces_std"] = stats["per_atom_force_std"]
                atoms.info["mlff_energy_var"] = stats["energy_std"] ** 2
                atoms.info["mlff_max_ferr_rel"] = stats["max_ferr_rel"]
                out_frames.append(atoms)
            write(args.output, out_frames)
            print(
                f"Wrote per-atom deviation for {len(out_frames)} frames to: {args.output}"
            )
        return

    stats = compute_deviation(
        str(args.config),
        args.models,
        config_index=args.config_index,
        device=args.device,
        dtype=args.dtype,
    )

    print("=" * 70)
    print(f"  Configuration : {args.config}  (index {args.config_index})")
    print(f"  Committee     : {stats['num_models']} models")
    print("=" * 70)
    print(f"  Energy (mean) : {stats['energy']:+.6f} eV")
    print(f"  Energy std    : {stats['energy_std']:.6e} eV")
    print(f"  Force std     : mean = {stats['mean_force_std']:.6e} eV/A")
    print(f"                 max  = {stats['max_force_std']:.6e} eV/A")
    print(f"  Max rel. force error (ferr_rel) : {stats['max_ferr_rel']:.4f}")
    print("=" * 70)

    if args.output:
        atoms_out = stats["atoms"].copy()
        atoms_out.arrays["mlff_forces_std"] = stats["per_atom_force_std"]
        atoms_out.info["mlff_energy_var"] = stats["energy_std"] ** 2
        atoms_out.info["mlff_max_ferr_rel"] = stats["max_ferr_rel"]
        write(args.output, atoms_out)
        print(f"  Wrote per-atom deviation to: {args.output}")


if __name__ == "__main__":
    main()
