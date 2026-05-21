from pathlib import Path

PATH_REPO = Path(__file__).resolve().parent.parent

# Groups
LIF64_GROUP = "LIF64_KJPAW_V2"
LIF_KJPAW_GROUP = "LIF_KJPAW"

DATA_DIR = PATH_REPO / "dataset"
XYZ_DIR = "xyz_files"

CACHE_DIR = Path("~/.cache/mace").expanduser()
MACE_MP_0_MODEL = CACHE_DIR / "20231210mace128L0_energy_epoch249model"
MACE_MP_0B3_MODEL = CACHE_DIR / "mace-mp-0b3-medium.model"


OUTPUTS_DIR = PATH_REPO / "src" / "outputs"
PREDICTION_DIR = OUTPUTS_DIR / "predictions"
