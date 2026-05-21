from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from ase.atoms import Atoms
from mace.calculators import mace_mp

from config import MaceConfig


class MACE(nn.Module):
    def __init__(self, config: MaceConfig):
        self.config = config

    def build_calculator(self):
        return mace_mp(
            model=self.config.model,
            device=self.config.device,
            default_dtype=self.config.dtype,
        )

    def evaluate(self, atoms: Atoms, calculator) -> dict:
        """
        Take a configuration of atoms and return force and energies
        """
        atoms.calc = calculator
        mace_energy = float(atoms.get_potential_energy())
        mace_forces = np.asarray(atoms.get_forces())

        truth_energy = atoms.info["energy_truth"]
        truth_forces = atoms.arrays["forces_truth"]

        # Energy error
        dE = mace_energy - truth_energy
        dE_per_atom = dE / len(atoms)

        df = mace_forces - truth_forces
        force_mse = np.sqrt(np.mean(df**2))

        return {
            "mace_energy": mace_energy,
            "truth_energy": truth_energy,
            "dE": dE,
            "dE_per_atom": dE_per_atom,
            "natoms": len(atoms),
            "force_rmse": force_mse,
        }
