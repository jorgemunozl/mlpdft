"""A tiny randomly initialised multi-head ScaleShiftMACE for CPU unit tests.

Same block types as MACE-OMAT-medium (linear readout on layer 0, non-linear
readout on the last layer, ZBL pair repulsion, per-head scale/shift) at a few
channels, so the LLPR and shallow-ensemble code paths run in seconds.
"""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from ase.build import bulk
from e3nn import o3

from mace import modules

FOUNDATION_MODEL = "/home/jorge/project/mlpdft/src/mlpdft/outputs/mace_omat_medium/mace_omat_medium.model"


def toy_model(heads=("h0", "h1", "h2"), seed=0):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    n_heads = len(heads)
    block = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    model = modules.ScaleShiftMACE(
        r_max=3.5,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=block,
        interaction_cls_first=block,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("6x0e+6x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.tile([-1.5, -2.5], (n_heads, 1)),
        avg_num_neighbors=10.0,
        atomic_numbers=[3, 9],
        correlation=2,
        gate=torch.nn.functional.silu,
        pair_repulsion=True,
        heads=list(heads),
        atomic_inter_scale=[0.8 + 0.2 * k for k in range(n_heads)],
        atomic_inter_shift=[0.05 * k for k in range(n_heads)],
    )
    return model.double()


def lif_frames(n=3, seed=0, label_model=None):
    """Rattled 8-atom rock-salt LiF cells with synthetic REF labels."""
    rng = np.random.default_rng(seed)
    frames = []
    for index in range(n):
        atoms = bulk("LiF", "rocksalt", a=4.03, cubic=True)
        atoms.positions += rng.normal(scale=0.08, size=atoms.positions.shape)
        atoms.info["REF_energy"] = float(-20.0 + rng.normal())
        atoms.arrays["REF_forces"] = rng.normal(scale=0.3, size=(len(atoms), 3))
        atoms.info["config_type"] = "bulk_LiF" if index % 2 == 0 else "vacancy"
        frames.append(atoms)
    return frames
