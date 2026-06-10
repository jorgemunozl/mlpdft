import re

import torch
from torch import nn

model = torch.load(
    "lammps_example/FitTorch_Pytorch.pt", map_location="cpu", weights_only=False
)
sd = model.model_state_dict()


# Each subnetwork is a stack of Linear + ReLU layers
# Keys look like: network_architecture0.0.weight, network_architecture0.0.bias, ...
def build_subnetwork(sd, prefix, n_desc):
    layers = []
    i = 0
    while f"{prefix}.{i}.weight" in sd:
        w = sd[f"{prefix}.{i}.weight"]
        b = sd[f"{prefix}.{i}.bias"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        layers[-1].weight.data = w
        layers[-1].bias.data = b
        if f"{prefix}.{i + 1}.weight" in sd:  # has next layer → add ReLU
            layers.append(nn.ReLU())
        i += 1
    return nn.Sequential(*layers).eval()


num_desc = sd["network_architecture0.0.weight"].shape[1]  # input dim
subnets = [
    build_subnetwork(sd, f"network_architecture{e}", num_desc)
    for e in range(model.n_elements)
]  # 2 elements: Li, F
import lammps
import lammps.mliap
from ase.io import read
from lammps.mliap.pytorch import ElemwiseModels, TorchWrapper

atoms_list = read("dataset/LIF_KJPAW/xyz_files/LIF_KJPAW_10_20.extxyz", index=":")

results = []
for atoms in atoms_list:
    lmp = lammps.lammps(cmdargs=["-echo", "none"])
    lammps.mliap.activate_mliappy(lmp)

    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")
    lmp.command(
        f"pair_style mliap model mliappy model_placeholder "
        f"descriptor sna lammps_example/LiF_pot.mliap.descriptor"
    )

    cell = atoms.cell
    lmp.command(f"region box block 0 {cell[0][0]} 0 {cell[1][1]} 0 {cell[2][2]}")
    lmp.command("create_box 2 box")
    lmp.command("pair_coeff * * Li F")

    # Load the PyTorch model into LAMMPS
    elemwise = ElemwiseModels(subnets, model.n_elements)
    wrapped = TorchWrapper(
        model=elemwise,
        n_descriptors=num_desc,
        n_elements=model.n_elements,
        dtype=torch.float64,
    )
    lammps.mliap.load_model(wrapped)

    # Create atoms
    type_map = {"Li": 1, "F": 2}
    for atom in atoms:
        t = type_map[atom.symbol]
        x, y, z = atom.position
        lmp.command(f"create_atoms {t} single {x} {y} {z} units box")

    lmp.command("mass 1 6.941")
    lmp.command("mass 2 18.998")
    lmp.command("run 0")

    # Extract energy and forces
    e_pred = lmp.get_thermo("pe")
    # ... extract forces from lmp.extract_atom("f", ...) ...

    e_ref = atoms.info["REF_energy"]
    f_ref = atoms.arrays["REF_forces"]

    results.append((e_pred, e_ref, f_pred, f_ref))
    lmp.close()
