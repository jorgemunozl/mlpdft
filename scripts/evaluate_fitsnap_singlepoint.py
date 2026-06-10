#!/usr/bin/env python3
"""
Single-point evaluation of a trained FitTorch model on DFT reference data.

Reads configurations from an .extxyz file, evaluates the ML model on each
one using LAMMPS + mliappy, and prints error metrics (MAE/RMSE for energy
and forces) compared to the DFT reference.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ase.io import read

# ── Paths ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent

MODEL_PATH = REPO / "lammps_example" / "FitTorch_Pytorch.pt"
DESCRIPTOR_PATH = REPO / "lammps_example" / "LiF_pot.mliap.descriptor"
XYZ_PATH = REPO / "dataset" / "LIF_KJPAW" / "xyz_files" / "LIF_KJPAW_10_20.extxyz"

N_ELEMENTS = 2
ELEMENT_NAMES = ["Li", "F"]
ATOM_TYPE_MAP = {"Li": 1, "F": 2}
MASSES = {1: 6.941, 2: 18.998}


# ── Helper: build subnetworks from a loaded TorchWrapper ──────────────
def extract_subnets(torch_wrapper):
    """
    Given the loaded TorchWrapper (from fitsnap3lib), extract the parameters
    from its internal ElemwiseModels and build fresh nn.Sequential modules
    compatible with lammps.mliap.pytorch.
    """
    n_desc = int(torch_wrapper.n_descriptors)
    subnets = []

    for elem_idx, orig_subnet in enumerate(torch_wrapper.model.subnets):
        sd = orig_subnet.state_dict()
        layers = []
        weight_keys = sorted(
            [k for k in sd if k.endswith(".weight")],
            key=lambda k: int(k.split(".")[0]),
        )
        n_layers = len(weight_keys)
        for i, wk in enumerate(weight_keys):
            bk = wk.replace(".weight", ".bias")
            w = sd[wk].to(torch.float32)
            b = sd[bk].to(torch.float32)
            linear = nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data = w
            linear.bias.data = b
            layers.append(linear)
            if i < n_layers - 1:
                layers.append(nn.ReLU())

        subnet = nn.Sequential(*layers).eval()
        nparams = sum(p.numel() for p in subnet.parameters())
        print(f"    Subnet [{ELEMENT_NAMES[elem_idx]}]: {nparams} parameters")
        subnets.append(subnet)

    return subnets, n_desc


# ── Inference on a single ASE Atoms object ────────────────────────────
def evaluate_frame(atoms, subnets, n_desc):
    import lammps
    import lammps.mliap
    from lammps.mliap.pytorch import ElemwiseModels, TorchWrapper

    lmp = lammps.lammps(cmdargs=["-echo", "none"])
    lammps.mliap.activate_mliappy(lmp)

    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")

    # Use a placeholder model name; the real model is loaded below
    lmp.command(
        f"pair_style mliap model mliappy LATER descriptor sna {DESCRIPTOR_PATH}"
    )

    # Create box
    cell = atoms.cell
    lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
    lmp.command(f"region box block 0 {lx:.10f} 0 {ly:.10f} 0 {lz:.10f}")
    lmp.command("create_box 2 box")
    lmp.command("pair_coeff * * Li F")

    # Load the PyTorch model into LAMMPS
    elemwise_model = ElemwiseModels(subnets, N_ELEMENTS)
    wrapped = TorchWrapper(
        model=elemwise_model,
        n_descriptors=n_desc,
        n_elements=N_ELEMENTS,
        dtype=torch.float64,
    )
    lammps.mliap.load_model(wrapped)

    # Create atoms, wrapping positions into the box
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions(wrap=True)
    for t, (x, y, z) in zip([ATOM_TYPE_MAP[s] for s in symbols], positions):
        lmp.command(f"create_atoms {t} single {x:.10f} {y:.10f} {z:.10f} units box")

    lmp.command("mass 1 6.941")
    lmp.command("mass 2 18.998")

    # Evaluate
    lmp.command("run 0")
    energy_mliap = lmp.get_thermo("pe")

    # Extract forces
    n_atoms = lmp.get_natoms()
    f_ptr = lmp.extract_atom("f", 2)
    # LP_c_double → raw ctypes pointer address → numpy array
    ptr = ctypes.cast(f_ptr, ctypes.c_void_p)
    buf = (ctypes.c_double * (n_atoms * 3)).from_address(ptr.value)
    forces = np.frombuffer(buf, dtype=np.float64).copy().reshape(n_atoms, 3)

    lmp.close()
    return energy_mliap, forces


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FitSNAP — Single-point evaluation")
    print("=" * 60)
    print(f"\nModel:  {MODEL_PATH}")
    print(f"Desc:   {DESCRIPTOR_PATH}")
    print(f"Data:   {XYZ_PATH}")

    print("\n[1] Loading model checkpoint...")
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    subnets, n_desc = extract_subnets(model)
    print(f"    Descriptors per atom: {n_desc}")

    # 2. Load reference data
    print("\n[2] Loading reference data...")
    atoms_list = read(str(XYZ_PATH), index=":")
    print(f"    {len(atoms_list)} frames, {len(atoms_list[0])} atoms each")

    # 3. Evaluate each frame
    print("\n[3] Evaluating frames...")
    e_preds = []
    e_refs = []
    f_preds = []
    f_refs = []

    for i, atoms in enumerate(atoms_list):
        e_ref = atoms.info["REF_energy"]
        f_ref = atoms.arrays["REF_forces"]

        e_pred, f_pred = evaluate_frame(atoms, subnets, n_desc)

        # Check atom count consistency
        if len(f_pred) != len(f_ref):
            print(
                f"    ⚠ Frame {i}: predicted {len(f_pred)} atoms vs "
                f"reference {len(f_ref)} atoms (SKIPPING)"
            )
            continue

        e_preds.append(e_pred)
        e_refs.append(e_ref)
        f_preds.append(f_pred)
        f_refs.append(f_ref)

        print(
            f"    Frame {i:2d}:  E_ref={e_ref:14.6f}  "
            f"E_pred={e_pred:14.6f}  ΔE={e_pred - e_ref:+.6f}"
        )

    if not e_preds:
        print("\nNo valid frames to compute metrics.")
        return

    # 4. Metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    e_preds = np.array(e_preds)
    e_refs = np.array(e_refs)

    energy_mae = np.abs(e_preds - e_refs).mean()
    energy_rmse = np.sqrt(((e_preds - e_refs) ** 2).mean())
    energy_maxae = np.abs(e_preds - e_refs).max()

    print(f"\n  Energy (eV):")
    print(f"    MAE  = {energy_mae:.6f}")
    print(f"    RMSE = {energy_rmse:.6f}")
    print(f"    MaxAE= {energy_maxae:.6f}")

    all_f_pred = np.concatenate(f_preds)
    all_f_ref = np.concatenate(f_refs)

    force_diff = all_f_pred - all_f_ref
    force_mae = np.abs(force_diff).mean()
    force_rmse = np.sqrt((force_diff**2).mean())
    force_component_mae = np.abs(force_diff).mean(axis=0)
    force_maxae = np.abs(force_diff).max()

    print(f"\n  Forces (eV/Å):")
    print(f"    MAE  (all)     = {force_mae:.6f}")
    print(f"    RMSE (all)     = {force_rmse:.6f}")
    print(
        f"    MAE  (x/y/z)   = {force_component_mae[0]:.6f}  "
        f"{force_component_mae[1]:.6f}  {force_component_mae[2]:.6f}"
    )
    print(f"    MaxAE          = {force_maxae:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
