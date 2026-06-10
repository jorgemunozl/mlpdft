import lammps
import lammps.mliap
import torch
from ase.io import read

from mlpdft.constants import (
    DATA_DIR,
    FITSNAP_LIF_CHECKPOINTS_DIR,
    LIF_KJPAW_GROUP,
    XYZ_DIR,
)

DESCRIPTOR_FILE = str(
    FITSNAP_LIF_CHECKPOINTS_DIR / "LiF64_NEWJSON_pot.mliap.descriptor"
)
CHECKPOINT_FILE = str(FITSNAP_LIF_CHECKPOINTS_DIR / "LiF_Pytorch.pt")
NUM_DESCRIPTORS = 440  # twojmax=8, 2 elements, chemflag=1
NUM_ELEMENTS = 2  # Li, F

GROUND_TRUTH_FILE = DATA_DIR / LIF_KJPAW_GROUP / XYZ_DIR / "LIF_KJPAW_10_20.extxyz"

# Read with ase
configs = read(GROUND_TRUTH_FILE, index=":")

for config in configs:
    lmp = lammps.lammps(cmdargs=["-echo", "none"])
    lammps.mliap.activate_mliappy(lmp)
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")
    cell = config.cell
    lmp.command(f"region box block 0 {cell[0][0]} 0 {cell[1][1]} 0 {cell[2][2]}")
    lmp.command("create_box 2 box")
    lmp.command(
        f"pair_style mliap model mliappy LATER descriptor sna lammps_example/LiF_pot.mliap.descriptor"
    )
    lmp.command("pair_coeff * * Li F")
    model = torch.load(CHECKPOINT_FILE)
    lammps.mliap.load_model(model)
    # lmp.command(
    #     f"pair_style hybrid/overlay lj/cut 10.0 mliap model mliappy lammps_example/FitTorch_Pytorch.pt descriptor sna lammps_example/LiF_pot.mliap.descriptor"
    # )

    cell = config.cell
    lmp.command(f"region box block 0 {cell[0][0]} 0 {cell[1][1]} 0 {cell[2][2]}")
    lmp.command("create_box 2 box")
