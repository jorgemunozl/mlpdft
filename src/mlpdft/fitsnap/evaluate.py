#!/usr/bin/env python3
"""
Evaluación de modelo FitSNAP (SNAP + PyTorch) para LiF.

Usa LAMMPS directamente (modo librería) con el modelo mliappy.
NO usa LAMMPSlib porque necesitamos activate_mliappy() antes de pair_style.
"""

import ctypes
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ase.build import bulk

# ── Configuración ──────────────────────────────────────────────────────
FITSNAP_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitsnap_models" / "LI_F"
)
DESCRIPTOR_FILE = str(FITSNAP_DIR / "LiF64_NEWJSON_pot.mliap.descriptor")
CHECKPOINT_FILE = str(FITSNAP_DIR / "checkpoints" / "LiF_Pytorch.pt")

NUM_DESCRIPTORS = 440  # twojmax=8, 2 elementos, chemflag=1
NUM_ELEMENTS = 2  # Li, F


# ── Reconstruir red desde checkpoint FitSNAP ──────────────────────────
def build_subnetwork(state_dict, prefix):
    """Reconstruye nn.Sequential desde state_dict de FitSNAP."""
    indices = set()
    for key in state_dict:
        match = re.match(rf"^{re.escape(prefix)}\.(\d+)\.weight$", key)
        if match:
            indices.add(int(match.group(1)))

    layers = []
    max_idx = max(indices) if indices else 0
    for i in sorted(indices):
        w_key = f"{prefix}.{i}.weight"
        b_key = f"{prefix}.{i}.bias"
        if w_key in state_dict:
            w = state_dict[w_key].to(torch.float32)
            b = state_dict[b_key].to(torch.float32)
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            layers[-1].weight.data = w
            layers[-1].bias.data = b
            if i < max_idx:
                layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def load_fitsnap_checkpoint(path):
    """Carga checkpoint FitSNAP y reconstruye subredes."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]
    subnets = []
    for elem_idx in range(NUM_ELEMENTS):
        subnet = build_subnetwork(sd, f"network_architecture{elem_idx}")
        subnet.eval()
        subnets.append(subnet)
        nparams = sum(p.numel() for p in subnet.parameters())
        print(f"  Red [{['Li', 'F'][elem_idx]}]: {nparams} parámetros")
    return subnets


# ── Inferencia con LAMMPS + mliappy ───────────────────────────────────
def run_inference(atoms, subnets):
    import lammps
    import lammps.mliap
    from lammps.mliap.pytorch import ElemwiseModels, TorchWrapper

    # 1. Inicializar LAMMPS
    lmp = lammps.lammps(cmdargs=["-echo", "screen"])

    # 2. Activar mliappy (OBLIGATORIO antes de pair_style)
    lammps.mliap.activate_mliappy(lmp)

    # 3. Configurar simulación
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")
    lmp.command(
        f"pair_style mliap model mliappy LATER descriptor sna {DESCRIPTOR_FILE}"
    )

    # 4. Crear caja y átomos
    cell = atoms.cell.array
    pos = atoms.positions
    symbols = atoms.get_chemical_symbols()
    type_map = {"Li": 1, "F": 2}
    atom_types = [type_map[s] for s in symbols]

    lx, ly, lz = cell[0][0], cell[1][1], cell[2][2]
    lmp.command(f"region mybox block 0 {lx:.6f} 0 {ly:.6f} 0 {lz:.6f}")
    lmp.command("create_box 2 mybox")

    # Cargar modelo PyTorch ANTES de pair_coeff
    from lammps.mliap.pytorch import ElemwiseModels, TorchWrapper

    elemwise_model = ElemwiseModels(subnets, NUM_ELEMENTS)
    wrapped = TorchWrapper(
        model=elemwise_model,
        n_descriptors=NUM_DESCRIPTORS,
        n_elements=NUM_ELEMENTS,
        dtype=torch.float64,
    )
    lammps.mliap.load_model(wrapped)

    lmp.command("pair_coeff * * mliap Li F")

    for p, t in zip(pos, atom_types):
        lmp.command(
            f"create_atoms {t} single {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} units box"
        )

    lmp.command("mass 1 6.941")
    lmp.command("mass 2 18.998")

    # 6. Calcular energía
    lmp.command("run 0")
    energy = lmp.get_thermo("pe")

    # 7. Extraer fuerzas
    n_atoms = lmp.get_natoms()
    f_ptr = lmp.extract_atom("f", 2)
    buf = (ctypes.c_double * (n_atoms * 3)).from_address(
        ctypes.addressof(ctypes.c_double.from_address(ctypes.addressof(f_ptr)))
    )
    forces = np.frombuffer(buf, dtype=np.float64).copy().reshape(n_atoms, 3)

    lmp.close()
    return energy, forces


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FitSNAP LiF — Inferencia con LAMMPS + PyTorch")
    print("=" * 60)

    # 1. Crear estructura de prueba (LiF roca salina)
    print("\n[1] Creando estructura...")
    atoms = bulk("LiF", crystalstructure="rocksalt", a=4.02, cubic=True)
    print(f"    {len(atoms)} átomos: {atoms.get_chemical_symbols()}")

    # 2. Cargar checkpoint
    print(f"\n[2] Cargando checkpoint: {CHECKPOINT_FILE}")
    subnets = load_fitsnap_checkpoint(CHECKPOINT_FILE)

    # 3. Inferencia
    print("\n[3] Ejecutando inferencia con LAMMPS...")
    energy, forces = run_inference(atoms, subnets)

    # 4. Resultados
    print(f"\n✅ Energía total: {energy:.6f} eV")
    print(f"   Fuerzas (eV/Å):")
    for i in range(min(3, len(forces))):
        print(
            f"     átomo {i}: {forces[i][0]:10.6f}  {forces[i][1]:10.6f}  {forces[i][2]:10.6f}"
        )


if __name__ == "__main__":
    main()
