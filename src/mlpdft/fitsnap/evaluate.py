#!/usr/bin/env python3
"""
Evaluación de modelo FitSNAP (SNAP no lineal + PyTorch) para LiF.
Usa LAMMPS en modo librería vía ASE.
"""

from pathlib import Path

from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read

# === CONFIGURACIÓN ===
FITSNAP_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitsnap_models" / "LI_F"
)
DESCRIPTOR_FILE = str(FITSNAP_DIR / "LiF64_NEWJSON_pot.mliap.descriptor")
CHECKPOINT_FILE = str(FITSNAP_DIR / "checkpoints" / "LiF_Pytorch.pt")

# 1. Crear o cargar una estructura de prueba
# Opción A: crear una estructura simple de LiF (roca salina)
atoms = bulk("LiF", crystalstructure="rocksalt", a=4.02)

# Opción B: cargar desde archivo (descomenta la que uses)
# atoms = read("tu_estructura.extxyz")

# 2. Comandos LAMMPS para cargar el modelo FitSNAP PyTorch
lammps_cmds = [
    # Unidades (deben coincidir con las del entrenamiento: metal)
    "units metal",
    "atom_style atomic",
    # Cargar descriptor SNAP + modelo PyTorch
    f"pair_style mliap model mliap_model.python descriptor mliap_descriptor.snap model_filename {CHECKPOINT_FILE}",
    # También puedes especificar el descriptor por separado:
    # Necesitamos que LAMMPS lea el descriptor
]

# Para LAMMPSlib, necesitas configurar los archivos de descriptor
# ya que mliap necesita acceso al archivo .snap o .mliap.descriptor
# La forma más limpia es usar LAMMPS directamente con comandos:

lammps_cmds = [
    "units metal",
    "atom_style atomic",
    f"pair_style mliap model mliap_model.python descriptor mliap_descriptor.snap model_filename {CHECKPOINT_FILE}",
    "pair_coeff * * Li F",
]

# 3. Configurar el calculador LAMMPSlib
calc = LAMMPSlib(
    lmpcmds=lammps_cmds,
    atom_types={"Li": 1, "F": 2},
    keep_alive=True,
    # Es importante que LAMMPS pueda encontrar el archivo descriptor
    # Una opción es copiarlo al directorio de trabajo o usar ruta absoluta
)

# Alternativa: usar lammps directamente sin ASE para más control
# (ver abajo)

atoms.calc = calc

# 4. Evaluar
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energía predicha: {energy:.6f} eV")
print(f"Fuerzas (primeros 3 átomos):\n{forces[:3]}")
