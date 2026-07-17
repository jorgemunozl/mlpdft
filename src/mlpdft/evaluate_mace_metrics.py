#!/usr/bin/env python3
"""
Evaluate Model on all available groups and save metrics
to outputs/metrics/ in the same format as the existing FitSNAP metrics.
Use compiled model
"""

from __future__ import annotations

import json
from pathlib import Path

import ase.io
import numpy as np
import torch
from mace import data
from mace.tools import torch_geometric, torch_tools, utils

from mlpdft.config import MaceConfig
from mlpdft.constants import (
    DATA_DIR,
    ENERGY_KEY,
    FORCE_KEY,
    GROUPS_LIF,
    SRC_DIR,
    XYZ_DIR,
)

DEVICE = "cpu"
DTYPE = "float32"

FRAME_STRIDE = 5
MAX_FRAMES = 100


def _get_model_float_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.get_default_dtype()


def _cast_batch_to_dtype(
    batch_dict: dict[str, torch.Tensor], target_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            out[key] = value.to(dtype=target_dtype)
        else:
            out[key] = value
    return out


def compute_metrics_for_group(
    model: torch.nn.Module,
    group: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict | None:
    """Compute energy and force metrics for one group. Returns dict or None on failure."""
    extxyz_path = (
        DATA_DIR / group / XYZ_DIR / f"{group}_{FRAME_STRIDE}_{MAX_FRAMES}.extxyz"
    )
    if not extxyz_path.exists():
        print(f"  [SKIP] extxyz not found: {extxyz_path}")
        return None

    print(f"  Loading: {extxyz_path}")
    atoms_list = ase.io.read(str(extxyz_path), index=":")

    # Filter frames that have reference data
    valid = []
    for atoms in atoms_list:
        has_energy = ENERGY_KEY in atoms.info
        has_forces = FORCE_KEY in atoms.arrays
        if has_energy and has_forces:
            valid.append(atoms)
        else:
            print(f"    Warning: skipping frame missing {ENERGY_KEY} or {FORCE_KEY}")

    if not valid:
        print("  [SKIP] no valid frames with reference data")
        return None

    atoms_list = valid
    print(f"  Valid frames: {len(atoms_list)}")

    # Prepare MACE data
    head_name = "Default"
    configs = [
        data.config_from_atoms(atoms, head_name=head_name) for atoms in atoms_list
    ]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    # Collect predictions
    energies_pred = []
    forces_pred_list = []

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = _cast_batch_to_dtype(batch.to_dict(), model_dtype)
        output = model(batch_dict)

        energies_pred.append(torch_tools.to_numpy(output["energy"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_pred_list.append(forces[:-1])  # drop last empty split

    energies_pred = np.concatenate(energies_pred, axis=0)
    forces_pred = [f for sublist in forces_pred_list for f in sublist]

    assert len(atoms_list) == len(energies_pred) == len(forces_pred)

    # Collect reference values
    energies_ref = np.array([atoms.info[ENERGY_KEY] for atoms in atoms_list])

    natoms_list = np.array([len(atoms) for atoms in atoms_list])

    # Energy metrics: total energy (eV)
    energy_errors = energies_pred - energies_ref
    energy_mae = np.mean(np.abs(energy_errors))
    energy_rmse = np.sqrt(np.mean(energy_errors**2))
    energy_maxae = np.max(np.abs(energy_errors))

    # Energy metrics: per-atom (eV/atom) — matching MACE's rmse_e_per_atom
    energy_errors_per_atom = energy_errors / natoms_list
    energy_mae_per_atom = np.mean(np.abs(energy_errors_per_atom))
    energy_rmse_per_atom = np.sqrt(np.mean(energy_errors_per_atom**2))
    energy_maxae_per_atom = np.max(np.abs(energy_errors_per_atom))

    # Force metrics
    all_force_errors = []
    force_errors_x = []
    force_errors_y = []
    force_errors_z = []
    frame_force_max = []

    for atoms, f_pred in zip(atoms_list, forces_pred):
        f_ref = atoms.arrays[FORCE_KEY]
        df = f_pred - f_ref
        all_force_errors.extend(df.ravel())
        force_errors_x.extend(df[:, 0])
        force_errors_y.extend(df[:, 1])
        force_errors_z.extend(df[:, 2])
        frame_force_max.append(np.max(np.abs(df)))

    all_force_errors = np.array(all_force_errors)
    force_errors_x = np.array(force_errors_x)
    force_errors_y = np.array(force_errors_y)
    force_errors_z = np.array(force_errors_z)

    force_mae_all = np.mean(np.abs(all_force_errors))
    force_rmse_all = np.sqrt(np.mean(all_force_errors**2))
    force_mae_x = np.mean(np.abs(force_errors_x))
    force_mae_y = np.mean(np.abs(force_errors_y))
    force_mae_z = np.mean(np.abs(force_errors_z))
    force_maxae = np.max(frame_force_max)

    return {
        "group": group,
        "n_configs": len(atoms_list),
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "energy_maxae": energy_maxae,
        "energy_mae_per_atom": energy_mae_per_atom,
        "energy_rmse_per_atom": energy_rmse_per_atom,
        "energy_maxae_per_atom": energy_maxae_per_atom,
        "force_mae_all": force_mae_all,
        "force_rmse_all": force_rmse_all,
        "force_mae_x": force_mae_x,
        "force_mae_y": force_mae_y,
        "force_mae_z": force_mae_z,
        "force_maxae": force_maxae,
    }


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def write_metrics_json(
    all_results: list[dict],
    output_path: Path,
    *,
    model_key: str,
    frame_stride: int,
    max_frames: int,
) -> None:
    """Write all group metrics to a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": model_key,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "results": _to_native(all_results),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    torch_tools.set_default_dtype(DTYPE)

    device = torch.device(DEVICE)
    config = MaceConfig(
        model_key="mace_omat_lora_v1", frame_stride=FRAME_STRIDE, max_frames=MAX_FRAMES
    )

    print(f"Loading model: {config.model_key:}")
    print(f"  Path: {config.model.compiled_path}")
    model = torch.load(str(config.model.compiled_path), map_location=device)
    model = model.to(device)
    model.eval()
    model_dtype = _get_model_float_dtype(model)
    print(f"  Dtype: {model_dtype}")

    metrics_dir = SRC_DIR / "metrics" / config.model_key
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    ready = ["LIFINTERFACE_KJPAW_NPT_V2", "LIFINTERFACE_KJPAW_NPT_V1"]
    print(f"Ready groups: {ready}")

    lack = [group for group in GROUPS_LIF if group not in ready]

    print(f"Remaining groups: {lack}")

    for group in lack:
        print(f"\n{'=' * 60}")
        print(f"Group: {group}")
        print(f"{'=' * 60}")

        metrics = compute_metrics_for_group(model, group, device, model_dtype)
        if metrics is None:
            continue

        all_results.append(metrics)

    # Write single JSON
    json_path = metrics_dir / f"{config.model_key}.json"
    write_metrics_json(
        all_results,
        json_path,
        model_key=config.model_key,
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )
    print(f"\nSaved: {json_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY — {config.model_key} metrics")
    print(f"{'=' * 80}")
    print(
        f"{'Group':<35} {'E RMSE tot':>12} {'E RMSE/at':>12} {'E RMSE/at':>12} {'F RMSE':>10}"
    )
    print(f"{'':<35} {'(eV)':>12} {'(eV/atom)':>12} {'(meV/atom)':>12} {'(eV/Å)':>10}")
    print("-" * 81)
    for r in all_results:
        print(
            f"{r['group']:<35} {r['energy_rmse']:>12.4f} {r['energy_rmse_per_atom']:>12.6f} "
            f"{r['energy_rmse_per_atom'] * 1000:>12.2f} {r['force_rmse_all']:>10.4f}"
        )
    print(f"{'=' * 80}")
    print(f"Results saved in: {json_path}")


def _compute_ref_force_rms(group: str) -> float | None:
    """Compute RMS of reference forces for a group, for relative force RMSE."""
    extxyz_path = (
        DATA_DIR / group / XYZ_DIR / f"{group}_{FRAME_STRIDE}_{MAX_FRAMES}.extxyz"
    )
    if not extxyz_path.exists():
        return None
    atoms_list = ase.io.read(str(extxyz_path), index=":")
    all_forces = []
    for atoms in atoms_list:
        if FORCE_KEY in atoms.arrays:
            all_forces.append(atoms.arrays[FORCE_KEY].ravel())
    if not all_forces:
        return None
    all_forces = np.concatenate(all_forces)
    return np.sqrt(np.mean(all_forces**2))


if __name__ == "__main__":
    main()
