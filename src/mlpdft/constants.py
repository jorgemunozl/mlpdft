from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    def __init__(self, key: str):
        self.key: str = key
        prefix = Path(OUTPUTS_DIR / self.key)
        self.path: Path = prefix / f"{self.key}.model"
        self.hf_id: str | None = PREFIX_HF + "/" + self.key
        self.compiled_path: Path = prefix / f"{self.key}_compiled.model"


# ---------- dataset settings ----------
PREFIX_HF = "jorgemunozl"

# MAIN DATASET
DATASET_NAME_1 = "molecular_dynamics_li_f_qe"
DATASET_NAME_2 = "molecular_dynamics_li_f_qe_v2"
DATASET_NAME_3 = "molecular_dynamics_li_f_qe_v3"

MERGED_FILENAME_DS_3 = f"{DATASET_NAME_3}.extxyz"
MERGED_FILENAME_DS_1 = "minimal_li_f_mace_dataset.extxyz"
MERGED_FILENAME_DS_2 = "li_f_mace_dataset_v2.extxyz"

FRAME_STRIDE = 3
MAX_FRAMES = None  # use all frames after striding

# Template path (sibling of this script)
PATH_REPO = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PATH_REPO / "dataset"
XYZ_DIR = "xyz_files"

SLIDES_REPO = PATH_REPO / "mlpdft_touying"

PLOT_DIR = SLIDES_REPO / "plots"


# Li–F interface systems
GROUPS_LIF_INTERFACE = [
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_V2",
    "LIFINTERFACE_KJPAW_NPT",
    "LIFINTERFACE_KJPAW_NPT_V2",
]

GROUP_LIWITHF = [
    "LIWITHF_NPT_FINAL",
    "LIWITHF_ISOLATED",
    "LIWITHF_V3",
]

GROUPS_LIF64 = [
    "LIF64_KJPAW_V2",
    "LIF64_KJPAW_NPT",
    "LIF64_KJPAW_NPT_V2",
    "LIF64_KJPAW_NPT_V3",
    "LIF64_KJPAW_NPT_FINAL",
    "LIF64_ISOLATED",
]

GROUPS_LIF = GROUP_LIWITHF + GROUPS_LIF64

GROUPS_LIBF4 = [
   "LIBF4_V4",
   "LIBF4_V2",
   "LIBF4",
   "LIBF4_FINAL",
   "LIBF4_NPT",
   "LIBF4_NPT_FINAL",
]

GROUPS_BLI = [
    "BLI_V2",
    "BLI_NPT",
    "BLI_ISOLATED",
]

GROUPS_BLI_INTERFACE = [
    "BLI_INTERFACE_NPT",
    "BLI_INTERFACE_NPT_FINAL",
    "BLI_INTERFACE_FINAL",
]

# Whole catalog (every group directory present in the dataset directory)
GROUPS = sorted(
    path.name
    for path in DATA_DIR.iterdir()
    if path.is_dir() and path.name != XYZ_DIR
) if DATA_DIR.is_dir() else []
GROUPS_INTERFACE = GROUPS_LIF_INTERFACE + GROUPS_BLI_INTERFACE

# Bulk and isolated cells (no interface)
GROUPS_BULK = [
    "LIWITHF_NPT_FINAL",
    "LIWITHF_ISOLATED",
    "LIF64_KJPAW_V2",
    "LIF64_KJPAW_NPT",
    "LIF64_KJPAW_NPT_V2",
    "LIF64_KJPAW_NPT_V3",
    "LIF64_KJPAW_NPT_FINAL",
    "LIWITHF_V3",
    "LIF64_ISOLATED",
    "BLI_V2",
    "BLI_NPT",
    "BLI_ISOLATED",
    "LIBF4_V4",
    "LIBF4_V2",
    "LIBF4",
    "LIBF4_FINAL",
    "LIBF4_NPT",
    "LIBF4_NPT_FINAL",
]

# ── by ensemble (NPT vs. NVT) ────────────────────────────────────
# NPT = variable-cell MD (calculation='vc-md'); NVT = fixed-cell MD.

GROUPS_NPT = [
    "LIFINTERFACE_KJPAW_NPT",
    "LIFINTERFACE_KJPAW_NPT_V2",
    "LIWITHF_NPT_FINAL",
    "LIF64_KJPAW_NPT",
    "LIF64_KJPAW_NPT_V2",
    "LIF64_KJPAW_NPT_V3",
    "LIF64_KJPAW_NPT_FINAL",
    "BLI_NPT",
    "BLI_INTERFACE_NPT",
    "BLI_INTERFACE_NPT_FINAL",
    "LIBF4_NPT",
    "LIBF4_NPT_FINAL",
]

GROUPS_NVT = [
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_V2",
    "LIWITHF_ISOLATED",
    "LIF64_KJPAW_V2",
    "LIWITHF_V3",
    "LIF64_ISOLATED",
    "BLI_V2",
    "BLI_INTERFACE_FINAL",
    "BLI_ISOLATED",
    "LIBF4_V4",
    "LIBF4_V2",
    "LIBF4",
    "LIBF4_FINAL",
]

# ── geometry × ensemble ──────────────────────────────────────────
GROUPS_INTERFACE_NPT = [
    "LIFINTERFACE_KJPAW_NPT",
    "LIFINTERFACE_KJPAW_NPT_V2",
    "BLI_INTERFACE_NPT",
    "BLI_INTERFACE_NPT_FINAL",
]

GROUPS_INTERFACE_NVT = [
    "LIFINTERFACE_KJPAW_V1",
    "LIFINTERFACE_KJPAW_V2",
    "BLI_INTERFACE_FINAL",
]

GROUPS_BULK_NPT = [
    "LIWITHF_NPT_FINAL",
    "LIF64_KJPAW_NPT",
    "LIF64_KJPAW_NPT_V2",
    "LIF64_KJPAW_NPT_V3",
    "LIF64_KJPAW_NPT_FINAL",
    "BLI_NPT",
    "LIBF4_NPT",
    "LIBF4_NPT_FINAL",
]

GROUPS_BULK_NVT = [
    "LIWITHF_ISOLATED",
    "LIF64_KJPAW_V2",
    "LIWITHF_V3",
    "LIF64_ISOLATED",
    "BLI_V2",
    "BLI_ISOLATED",
    "LIBF4_V4",
    "LIBF4_V2",
    "LIBF4",
    "LIBF4_FINAL",
]

# Groups that still lack a converted dataset; their raw QE jobs live under ./lack
LACK_DIR = PATH_REPO / "lack"
LACK_GROUPS = [
    "BLi3_v2",
    "BLi_interface",
    "EMIMLITFSI_ANODE_RV2_LACK",
    "LIBF4_LACK",
    "LIBF4_RELAX_LACK",
    "LIBF4_V3_LACK",
    "LIF64_ISOLATED_LACK",
    "LIF64_KJPAW_FINAL_LACK",
    "LIF64_KJPAW_LACK",
    "LIF64_KJPAW_NPT_LACK",
    "LIFINTERFACE_KJPAW_FINAL_LACK",
    "LIFINTERFACE_KJPAW_NPT_FINAL_LACK",
    "LIFINTERFACE_KJPAW_NPT_LACK",
    "LIFINTERFACE_KJPAW_NPT_V2_LACK",
    "LIFINTERFACE_KJPAW_V1_LACK",
]

# ── Small test subset for snapshot ensemble experiment ──
TEST_GROUPS = [
    "LIFINTERFACE_KJPAW_V1",  # interface
    "LIWITHF_ISOLATED",  # Li-rich isolated
    "LIF64_ISOLATED",  # bulk isolated
    "LIWITHF_NPT_FINAL",  # Li-rich NPT
    "BLI_V2",  # B-Li
    "BLI_NPT",  # B-Li NPT
    "LIBF4",  # salt
    "LIBF4_NPT",  # salt NPT
    "LIBF4_NPT_FINAL",  # salt NPT final
]

TEST_STRIDE = 3
TEST_MAX_FRAMES = 150
TEST_DATASET_NAME = "li_f_snapshot_test"


SRC_DIR = PATH_REPO / "src" / "mlpdft"
OUTPUTS_DIR = SRC_DIR / "outputs"
PREDICTION_DIR = OUTPUTS_DIR / "predictions"
UTILS_DIR = SRC_DIR / "utils"

TEMPLATE_PATH = UTILS_DIR / "dataset_readme_template.md"

ENERGY_KEY = "REF_energy"
FORCE_KEY = "REF_forces"

K_B = 8.617333262e-5  # eV / K

RY_TO_EV = 13.6056980659

LI_ISOLATED = -15.11995216 * RY_TO_EV
F_ISOLATED = -58.46236447 * RY_TO_EV
B_ISOLATED = -11.03228409 * RY_TO_EV

ENERGY_OFFSET = {
    3: LI_ISOLATED,
    9: F_ISOLATED,
    5: B_ISOLATED,
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
    "mace_omat_lora_v1": ModelSpec(
        key="mace_omat_lora_v1",
    ),
}

ACTIVE_LEARNING_DIR = OUTPUTS_DIR / "active_learning"
MOLECULAR_DYNAMICS_DIR = OUTPUTS_DIR / "molecular_dynamics"

# Small toy cell for testing and sanity checks (8-atom LiF, PBC)
TOY_CELL = {
    "symbols": ["Li", "Li", "Li", "Li", "F", "F", "F", "F"],
    "a": 4.02,
}

TOY_CELL_PATH = OUTPUTS_DIR / "toy_lif.extxyz"
