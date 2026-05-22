from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    key: str
    path: Path
    name: str
    energy_offset_per_atom: float = 0.0


PATH_REPO = Path(__file__).resolve().parent.parent

# Groups
LIF64_GROUP = "LIF64_KJPAW_V2"
LIF_KJPAW_GROUP = "LIF_KJPAW"

DATA_DIR = PATH_REPO / "dataset"
XYZ_DIR = "xyz_files"

CACHE_DIR = Path("~/.cache/mace").expanduser()

OUTPUTS_DIR = PATH_REPO / "src" / "outputs"
PREDICTION_DIR = OUTPUTS_DIR / "predictions"
UTILS_DIR = PATH_REPO / "src" / "utils"

MODEL_REGISTRY = {
    "0b3-medium": ModelSpec(
        key="0b3-medium",
        path=CACHE_DIR / "mace-mp-0b3-medium.model",
        name="mace-mp-0b3-medium",
        energy_offset_per_atom=0.0,
    ),
    "0-small": ModelSpec(
        key="0-small",
        path=CACHE_DIR / "20231210mace128L0_energy_epoch249model",
        name="mace-mp-0-small",
        energy_offset_per_atom=0.0,
    ),
}
