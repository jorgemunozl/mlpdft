from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from ase.atoms import Atoms
from mace.calculators import mace_mp

from config import MaceConfig


class MACE_SCRAP(nn.Module):
    def __init__(self, config: MaceConfig):
        self.config = config

    def build_calculator(self):
        return mace_mp(
            device=self.config.device,
            default_dtype=self.config.dtype,
        )

    def build_model(self):
        model = torch.load(self.config.model_path, map_location=self.config.device)
        return model
