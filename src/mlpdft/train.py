from huggingface_hub import hf_hub_download
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

from mlpdft.config import Mace_TrainerConfig
from mlpdft.constants import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    ENERGY_OFFSET,
    HF_REPO_ID,
    LOGS_DIR,
    MERGED_FILENAME,
    MODELS_DIR,
    RESULTS_DIR,
    XYZ_DIR,
)

# Scrap data from Hugging Face
dataset = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MERGED_FILENAME,
    repo_type="dataset",
    local_dir=str(DATA_DIR / XYZ_DIR),
)

path_data = str(DATA_DIR / XYZ_DIR / MERGED_FILENAME)

# Config Proof of Concept
config_poc = Mace_TrainerConfig(
    model_key="0-omat-medium",
    experiment_name="first training attempt",
    r_max=5.0,  # Å — 0.1 was way too small for any chemical bond
    device="cpu",
    max_num_epochs=300,
    batch_size=10,
)

config = config_poc

parser = build_default_arg_parser()
args = parser.parse_args(["--name", config.experiment_name])

args.seed = 123

# DIRS
args.checkpoints_dir = CHECKPOINTS_DIR
args.results_dir = RESULTS_DIR
args.model_dir = MODELS_DIR
args.log_dir = LOGS_DIR

# PRECISION
args.default_dtype = (
    "float64"  # 0-omat-medium is float64; must match or LoRA dtypes conflict
)

# DEVICE
args.device = config.device

# MODEL ARCHITECTURE — inherited from the foundation model, but we spell them out explicitly
args.model = "MACE"
args.r_max = config.r_max
args.num_channels = 128
args.max_L = 1  # 0-omat-medium uses L=1 (vectors); 0-small uses L=0 (scalars only)
args.max_ell = 3  # spherical harmonics l_max
args.num_interactions = 2  # number of interaction blocks
args.correlation = 3  # body order (3 = 4-body)
args.num_radial_basis = 8
args.num_cutoff_basis = 5

# DATA KEYS
args.energy_key = config.energy_key
args.force_key = config.force_key

# DATASET
args.pin_memory = config.pin_memory
args.E0s = str(ENERGY_OFFSET)

# Here set up the hf dataset
args.train_file = path_data

# LOSS — gentle weights for fine-tuning (foundation model already predicts well)
args.loss = "weighted"
args.forces_weight = 1.0  # default is 100 — too aggressive for FT
args.energy_weight = 1.0
args.valid_batch_size = config.valid_batch_size
args.valid_frac = config.valid_frac
args.batch_size = config.batch_size
args.max_num_epochs = config.max_num_epochs

# OPTIMIZER
args.optimizer = "adam"
args.lr = 0.01  # standard LR for MACE fine-tuning (from README)
args.amsgrad = True
args.weight_decay = 5e-7  # light L2 regularization
args.clip_grad = 10.0  # prevent gradient explosion

# SCHEDULER
args.scheduler = "ReduceLROnPlateau"
args.lr_factor = 0.8
args.scheduler_patience = 50

# STAGE TWO (previously called SWA) — boosts energy accuracy in final ~20% epochs
args.swa = True
args.start_swa = 240  # begin stage two at epoch 240
args.swa_lr = 1e-3

# EMA — smooths final weights for better stability
args.ema = True
args.ema_decay = 0.99

# EARLY STOPPING
args.patience = 100  # stop if validation loss plateaus for 100 epochs
args.eval_interval = 1

# Fine-tuning
args.foundation_model = config.model.path
args.lora = True
args.lora_rank = 8

# Train the model
run(args)

config.send_model_to_hf()
