"""Small helpers to run a saved MACE model directly on ASE frames.

The UQ baselines (shallow NLL ensemble, LLPR) need gradients through the model,
which ``MACECalculator`` does not expose.  These helpers build the same graphs
the calculator builds (same z-table, cutoff and head list) so predictions match
it to floating-point precision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from ase import Atoms

from mace import data as mace_data
from mace.tools import torch_geometric, utils

from mlpdft.constants import ENERGY_KEY, FORCE_KEY

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def load_model(path: str | Path, device: str = "cpu", dtype: str = "float64"):
    """Load a full MACE model object, cast to ``dtype`` and freeze nothing."""
    model = torch.load(str(path), map_location=device, weights_only=False)
    model = model.to(device=device, dtype=_DTYPES[dtype])
    torch.set_default_dtype(_DTYPES[dtype])
    return model


def model_heads(model) -> list[str]:
    heads = getattr(model, "heads", None)
    return list(heads) if heads else ["Default"]


def resolve_head(model, head: str | None) -> tuple[str, int]:
    """Return (name, index) of ``head``; ``None`` means the model's only head.

    Unlike ``MACECalculator`` this never falls back silently to another head.
    """
    heads = model_heads(model)
    if head is None:
        if len(heads) == 1:
            return heads[0], 0
        defaults = [name for name in heads if name.lower() == "default"]
        if len(defaults) != 1:
            raise ValueError(f"Model has heads {heads}; pass head=")
        head = defaults[0]
    if head not in heads:
        raise ValueError(f"Head {head!r} not in model heads {heads}")
    return head, heads.index(head)


def frames_to_data(model, frames: Sequence[Atoms], head: str | None = None) -> list:
    """Convert frames to ``AtomicData`` graphs for ``model`` (one per frame)."""
    head_name, _ = resolve_head(model, head)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    heads = model_heads(model)
    out = []
    for atoms in frames:
        config = mace_data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=head_name
        )
        out.append(
            mace_data.AtomicData.from_config(
                config, z_table=z_table, cutoff=cutoff, heads=heads
            )
        )
    return out


def collate(data_list: Iterable, device: str = "cpu") -> dict[str, torch.Tensor]:
    batch = torch_geometric.Batch.from_data_list(list(data_list)).to(device)
    return batch.to_dict()


def reference_forces(frames: Sequence[Atoms]) -> list[np.ndarray]:
    return [np.asarray(atoms.arrays[FORCE_KEY], dtype=float) for atoms in frames]


def reference_energies(frames: Sequence[Atoms]) -> np.ndarray:
    return np.asarray([float(atoms.info[ENERGY_KEY]) for atoms in frames])


def with_head(batch: dict[str, torch.Tensor], head_index: int) -> dict[str, torch.Tensor]:
    """Shallow copy of ``batch`` with every graph assigned to ``head_index``."""
    out = dict(batch)
    out["head"] = torch.full_like(batch["head"], int(head_index))
    return out
