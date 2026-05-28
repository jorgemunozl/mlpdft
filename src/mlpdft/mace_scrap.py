from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from mace.calculators import mace_mp

from mlpdft.config import MaceConfig


class MACE_SCRAP(nn.Module):
    def __init__(self, config: MaceConfig):
        self.config = config

    def build_calculator(self):
        return mace_mp(
            model=str(self.config.model.path.expanduser()),
            device=self.config.device,
            default_dtype=self.config.dtype,
        )

    def build_model(self):
        model_path = Path(self.config.model.path).expanduser()
        model = torch.load(model_path, map_location=self.config.device)
        return model
