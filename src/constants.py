from pathlib import Path

PATH_REPO = Path(__file__).resolve().parent.parent

DATA_DIR = PATH_REPO / "dataset"
LIF64_GROUP = "LIF64_KJPAW_V2"
LIF64_DIR = DATA_DIR / LIF64_GROUP
XYZ_DIR = "xyz_files"

PATH_MODEL = "/home/jorge/.cache/mace/20231210mace128L0_energy_epoch249model"
OUTPUTS_DIR = PATH_REPO / "src" / "outputs"
PREDICTION_DIR = OUTPUTS_DIR / "predictions"
