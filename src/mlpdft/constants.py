from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    def __init__(self, key: str):
        self.key = key
        self.path = Path(OUTPUTS_DIR / self.key / f"{self.key}.model")
        self.hf_id = PREFIX_HF + "/" + self.key
        self.compiled_path = Path(OUTPUTS_DIR / self.key / f"{self.key}_compiled.model")


# ---------- dataset settings ----------
PREFIX_HF = "jorgemunozl"

# MAIN DATASET
DATASET_NAME = "first_mace_test"
MERGED_FILENAME = "minimal_li_f_mace_dataset.extxyz"

FRAME_STRIDE = 5
MAX_FRAMES = None  # use all frames after striding

# Template path (sibling of this script)


PATH_REPO = Path(__file__).resolve().parent.parent.parent

# Groups
GROUPS_LIF = [
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_NPT_V2",
    "LIFINTERFACE_KJPAW_NPT",
    "LIWITHF_NPT_FINAL",
    "LIWITHF_ISOLATED",
    "LIF64_KJPAW_V2",
    "LIWITHF_V3",
    "LIF64_ISOLATED",
]

GROUPS_BLIF = ["BLI_V2", "LIBF4_V4", "LIBF4"]

GROUPS = GROUPS_LIF + GROUPS_BLIF

DATA_DIR = PATH_REPO / "dataset"
XYZ_DIR = "xyz_files"

SRC_DIR = PATH_REPO / "src" / "mlpdft"
OUTPUTS_DIR = SRC_DIR / "outputs"
PREDICTION_DIR = OUTPUTS_DIR / "predictions"
UTILS_DIR = SRC_DIR / "utils"

TEMPLATE_PATH = UTILS_DIR / "dataset_readme_template.md"

ENERGY_KEY = "REF_energy"
FORCE_KEY = "REF_forces"


RY_TO_EV = 13.6056980659

LI_ISOLATED = -15.11995216 * RY_TO_EV
F_ISOLATED = -58.46236447 * RY_TO_EV

ENERGY_OFFSET = {
    3: LI_ISOLATED,
    9: F_ISOLATED,
}

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"

FITSNAP_DIR = PATH_REPO / "fitsnap_models" / "LI_F"


MODEL_REGISTRY = {
    "mace_omat_medium": ModelSpec(
        key="mace_omat_medium",
    ),
    "mock_2_test": ModelSpec(
        key="mock_2_test",
    ),
}
