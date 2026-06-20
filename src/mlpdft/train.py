from huggingface_hub import hf_hub_download
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

from mlpdft.config import (
    Mace_TrainerConfig,
    MaceTrainingHyperparams,
    MaceTrainingMetadata,
)
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

# Download dataset from Hugging Face
dataset = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MERGED_FILENAME,
    repo_type="dataset",
    local_dir=str(DATA_DIR / XYZ_DIR),
)

path_data = str(DATA_DIR / XYZ_DIR / MERGED_FILENAME)
# path_data = str(DATA_DIR / XYZ_DIR / "test.extxyz")

# ---------------------------------------------------------------------------
# Config — all hyperparameter values come from the dataclass defaults.
# Only override what differs from the defaults.
# ---------------------------------------------------------------------------
config = Mace_TrainerConfig(
    model_key="0-omat-medium",
    device="cuda",
    dtype="float64",  # 0-omat-medium is float64; must match or LoRA dtypes conflict
    hyperparams=MaceTrainingHyperparams(
        r_max=0.1,
        max_num_epochs=1,
        batch_size=1,
    ),
    metadata=MaceTrainingMetadata(
        experiment_name="mock_test",
    ),
)
config.write_config_train()

# ---------------------------------------------------------------------------
# Build the MACE argument namespace from config fields
# ---------------------------------------------------------------------------
parser = build_default_arg_parser()
args = parser.parse_args(["--name", config.metadata.experiment_name])

args.seed = config.metadata.seed

# Dirs
args.checkpoints_dir = CHECKPOINTS_DIR
args.results_dir = RESULTS_DIR
args.model_dir = MODELS_DIR
args.log_dir = LOGS_DIR

# Precision & device
args.default_dtype = config.dtype
args.device = config.device

# Model architecture
args.model = "MACE"
args.r_max = config.hyperparams.r_max
args.num_channels = config.hyperparams.num_channels
args.max_L = config.hyperparams.max_L
args.max_ell = config.hyperparams.max_ell
args.num_interactions = config.hyperparams.num_interactions
args.correlation = config.hyperparams.correlation
args.num_radial_basis = config.hyperparams.num_radial_basis
args.num_cutoff_basis = config.hyperparams.num_cutoff_basis

# Data keys
args.energy_key = config.hyperparams.energy_key
args.force_key = config.hyperparams.force_key

# Dataset
args.pin_memory = config.hyperparams.pin_memory
args.E0s = str(ENERGY_OFFSET)
args.train_file = path_data

# Loss
args.loss = config.hyperparams.loss
args.forces_weight = config.hyperparams.forces_weight
args.energy_weight = config.hyperparams.energy_weight
args.valid_batch_size = config.hyperparams.valid_batch_size
args.valid_frac = config.hyperparams.valid_frac
args.batch_size = config.hyperparams.batch_size
args.max_num_epochs = config.hyperparams.max_num_epochs

# Optimizer
args.optimizer = config.hyperparams.optimizer
args.lr = config.hyperparams.lr
args.amsgrad = config.hyperparams.amsgrad
args.weight_decay = config.hyperparams.weight_decay
args.clip_grad = config.hyperparams.clip_grad

# Scheduler
args.scheduler = config.hyperparams.scheduler
args.lr_factor = config.hyperparams.lr_factor
args.scheduler_patience = config.hyperparams.scheduler_patience

# Stage two (SWA)
args.swa = config.hyperparams.swa
args.start_swa = config.hyperparams.start_swa
args.swa_lr = config.hyperparams.swa_lr

# EMA
args.ema = config.hyperparams.ema
args.ema_decay = config.hyperparams.ema_decay

# Early stopping
args.patience = config.hyperparams.patience
args.eval_interval = config.hyperparams.eval_interval

# LoRA fine-tuning
args.foundation_model = config.model.path  # resolved from model_key in __post_init__
args.lora = config.hyperparams.lora
args.lora_rank = config.hyperparams.lora_rank

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
run(args)

config.write_config_train()
config.send_model_to_hf()
