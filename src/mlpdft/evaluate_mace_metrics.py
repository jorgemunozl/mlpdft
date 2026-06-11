#!/usr/bin/env python3
"""
Evaluate MACE omat-medium model on all available groups and save metrics
to outputs/metrics/ in the same format as the existing FitSNAP metrics.
"""

from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import torch
from mace import data
from mace.tools import torch_geometric, torch_tools, utils

from mlpdft.constants import (
    DATA_DIR,
    ENERGY_KEY,
    FORCE_KEY,
    MODEL_REGISTRY,
    OUTPUTS_DIR,
    XYZ_DIR,
)

GROUPS = [
    "LIF64_ISOLATED",
    "LIF64_KJPAW_V2",
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_NPT",
    "LIFINTERFACE_KJPAW_NPT_V2",
    "LIF_KJPAW",
    "LIWITHF_V3",
    "LIWITHF_ISOLATED",
    "LIWITHF_NPT_FINAL",
    "BLI_V2",
    "LIBF4_V4",
]

FRAME_STRIDE = 5
MAX_FRAMES = 100
MODEL_KEY = "0-omat-medium"
DEVICE = "cpu"
DTYPE = "float32"


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

    # Energy metrics (total energy, eV)
    energy_errors = energies_pred - energies_ref
    energy_mae = np.mean(np.abs(energy_errors))
    energy_rmse = np.sqrt(np.mean(energy_errors**2))
    energy_maxae = np.max(np.abs(energy_errors))

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
        "force_mae_all": force_mae_all,
        "force_rmse_all": force_rmse_all,
        "force_mae_x": force_mae_x,
        "force_mae_y": force_mae_y,
        "force_mae_z": force_mae_z,
        "force_maxae": force_maxae,
    }


def write_metrics_txt(metrics: dict, output_path: Path) -> None:
    """Write metrics in the same format as the existing FitSNAP .txt files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "============================================================",
        "RESULTS",
        "============================================================",
        "",
        f"  Energy (eV):",
        f"    MAE  = {metrics['energy_mae']:.6f}",
        f"    RMSE = {metrics['energy_rmse']:.6f}",
        f"    MaxAE= {metrics['energy_maxae']:.6f}",
        "",
        f"  Forces (eV/Å):",
        f"    MAE  (all)     = {metrics['force_mae_all']:.6f}",
        f"    RMSE (all)     = {metrics['force_rmse_all']:.6f}",
        f"    MAE  (x/y/z)   = {metrics['force_mae_x']:.6f}  {metrics['force_mae_y']:.6f}  {metrics['force_mae_z']:.6f}",
        f"    MaxAE          = {metrics['force_maxae']:.6f}",
        "",
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    torch_tools.set_default_dtype(DTYPE)

    device = torch.device(DEVICE)
    model_spec = MODEL_REGISTRY[MODEL_KEY]
    model_path = Path(model_spec.path).expanduser()

    print(f"Loading model: {model_spec.name}")
    print(f"  Path: {model_path}")
    if not model_path.exists():
        print(f"  ERROR: model file not found at {model_path}")
        raise SystemExit(1)

    model = torch.load(str(model_path), map_location=device)
    model = model.to(device)
    model.eval()
    model_dtype = _get_model_float_dtype(model)
    print(f"  Dtype: {model_dtype}")

    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for group in GROUPS:
        print(f"\n{'=' * 60}")
        print(f"Group: {group}")
        print(f"{'=' * 60}")

        metrics = compute_metrics_for_group(model, group, device, model_dtype)
        if metrics is None:
            continue

        output_filename = f"{group}_{FRAME_STRIDE}_{MAX_FRAMES}.txt"
        output_path = metrics_dir / output_filename
        write_metrics_txt(metrics, output_path)
        print(f"  Saved: {output_path}")

        all_results.append(metrics)

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY — MACE omat-medium metrics")
    print(f"{'=' * 80}")
    print(
        f"{'Group':<35} {'Energy MAE':>12} {'Energy RMSE':>12} {'Force MAE':>10} {'Force RMSE':>10} {'Force MaxAE':>12}"
    )
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['group']:<35} {r['energy_mae']:>12.4f} {r['energy_rmse']:>12.4f} "
            f"{r['force_mae_all']:>10.4f} {r['force_rmse_all']:>10.4f} {r['force_maxae']:>12.4f}"
        )
    print(f"{'=' * 80}")
    print(f"Results saved in: {metrics_dir}")


if __name__ == "__main__":
    main()
