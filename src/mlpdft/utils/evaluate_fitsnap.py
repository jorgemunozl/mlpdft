from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read

from config import MaceConfig

# 1. Load your test structure (e.g., an extxyz file)
config = MaceConfig()
atoms = read(config.data_out_path)


# 2. Define the LAMMPS commands required to load your FitSNAP model
# Adjust "linear", your file names, and element mapping as necessary
lammps_cmds = [
    "pair_style mliap model linear my_model.mliap.model descriptor sna my_model.mliap.descriptor",
    "pair_coeff * * Ta W",
]

# 3. Set up the ASE LAMMPSlib calculator
# We map the ASE atomic symbols to LAMMPS types (e.g., Ta is type 1, W is type 2)
calc = LAMMPSlib(lmpcmds=lammps_cmds, atom_types={"Ta": 1, "W": 2}, keep_alive=True)

atoms.calc = calc

# 4. Predict Energy and Forces (Inference)
predicted_energy = atoms.get_potential_energy()
predicted_forces = atoms.get_forces()

print(f"Predicted Energy: {predicted_energy} eV")
print("Predicted Forces:\n", predicted_forces)
