from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    key: str
    path: Path
    name: str
    energy_offset_per_atom: float = 0.0


PATH_REPO = Path(__file__).resolve().parent.parent.parent

# Groups
LIF64_GROUP = "LIF64_KJPAW_V2"
LIF_KJPAW_GROUP = "LIF_KJPAW"

DATA_DIR = PATH_REPO / "dataset"
XYZ_DIR = "xyz_files"

CACHE_DIR = Path("~/.cache/mace").expanduser()

OUTPUTS_DIR = PATH_REPO / "src" / "outputs"
PREDICTION_DIR = OUTPUTS_DIR / "predictions"
UTILS_DIR = PATH_REPO / "src" / "utils"

ENERGY_KEY = "REF_energy"
FORCE_KEY = "REF_forces"


RY_TO_EV = 13.6056980659

LI_ISOLATED = -15.11995216 * RY_TO_EV
F_ISOLATED = -58.46236447 * RY_TO_EV

ENERGY_OFFSET = {
    3: LI_ISOLATED,
    9: F_ISOLATED,
}

OUTPUTS_DIR = PATH_REPO / "src" / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"

FITSNAP_DIR = PATH_REPO / "fitsnap_models"

FITSNAP_LIF_CHECKPOINTS_DIR = FITSNAP_DIR / "LI_F" / "checkpoints"


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
    "0-omat-medium": ModelSpec(
        key="0-omat-medium",
        path=CACHE_DIR / "mace-omat-0-medium.model",
        name="mace-mp-0-omat-medium",
        energy_offset_per_atom=0.0,
    ),
}
