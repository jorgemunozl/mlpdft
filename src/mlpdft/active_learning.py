import os

import torch

# ── CPU-only workaround ───────────────────────────────────────────
# The foundation models contain e3nn JIT submodules serialized for
# CUDA.  e3nn's __setstate__ calls torch.jit.load(buffer) *without*
# map_location, so the JIT loader tries to initialise a CUDA context
# even when we want CPU.  Monkey-patch to default to CPU.
_jit_load_original = torch.jit.load


def _jit_load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _jit_load_original(*args, **kwargs)


torch.jit.load = _jit_load_cpu  # type: ignore[assignment]
# ──────────────────────────────────────────────────────────────────

from mace.cli.active_learning_md import run

from mlpdft.config import ActiveLearningConfig
from mlpdft.constants import (
    ACTIVE_LEARNING_DIR,
    DATA_DIR,
    GROUPS_LIF,
    OUTPUTS_DIR,
)

# Committee of fine-tuned MACE models (same architecture, different seeds ->
# their disagreement is the epistemic-uncertainty signal). Swap in any
# combination, e.g. ["baseline_ft", "snapshot_warm", "committee_s123"].
COMMITTEE_KEYS = ["committee_s123", "committee_s124", "committee_s125"]
models = [
    str(OUTPUTS_DIR / key / "models" / f"{key}.model") for key in COMMITTEE_KEYS
]

group = GROUPS_LIF[0]

initial_config = str(
    DATA_DIR / group / "xyz_files" / "LIFINTERFACE_KJPAW_V1_5_100.extxyz"
)

output_dir = str(ACTIVE_LEARNING_DIR)
os.makedirs(output_dir, exist_ok=True)
output = str(ACTIVE_LEARNING_DIR / "test.extxyz")

config = ActiveLearningConfig(
    model_paths=models,
    config=initial_config,
    output=output,
)

if __name__ == "__main__":
    run(config.to_namespace())
