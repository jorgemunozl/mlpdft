import warnings
from pathlib import Path

import torch

# Suppress routine PyTorch/e3nn/cuequivariance warnings
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*not available.*")

from huggingface_hub import hf_hub_download
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

from mlpdft.config import (
    Mace_TrainerConfig,
    MaceTrainingHyperparams,
    MaceTrainingMetadata,
)
from mlpdft.constants import (
    DATA_DIR,
    DATASET_NAME_3,
    ENERGY_OFFSET,
    MERGED_FILENAME_DS_3,
    OUTPUTS_DIR,
    PREFIX_HF,
    SRC_DIR,
    XYZ_DIR,
)

# Download dataset from Hugging Face
dataset = hf_hub_download(
    repo_id=PREFIX_HF + "/" + DATASET_NAME_3,
    filename=MERGED_FILENAME_DS_3,
    repo_type="dataset",
    local_dir=str(DATA_DIR / XYZ_DIR),
)

path_data = str(DATA_DIR / XYZ_DIR / MERGED_FILENAME_DS_3)

# Test for check that training is working properly
# path_data = str(DATA_DIR / XYZ_DIR / "test.extxyz")

# ---------------------------------------------------------------------------
# Config — all hyperparameter values come from the dataclass defaults.
# Only override what differs from the defaults.
# ---------------------------------------------------------------------------
config = Mace_TrainerConfig(
    model_key="mace_omat_medium",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float64",  # must match foundation model dtype (mace_omat_medium is float64)
    hyperparams=MaceTrainingHyperparams(
        r_max=8.5,
        max_num_epochs=120,
        batch_size=8,
        patience=10,
        eval_interval=10,
        valid_frac=0.15,
        swa=False,
        num_channels=128,
        num_cutoff_basis=5,
        max_L=1,
        max_ell=3,
        num_interactions=2,
        correlation=3,
        num_radial_basis=8,
    ),
    metadata=MaceTrainingMetadata(
        experiment_name="mace_omat_lora_v2",
    ),
)
config.write_config_train()

# Load the config from the saved JSON file
# path = Path(SRC_DIR) / "configs" / "mock_test.json"
# config = Mace_TrainerConfig.load_config_train(path)

parser = build_default_arg_parser()
args = parser.parse_args(["--name", config.metadata.experiment_name])

# INDEPENDENT
args.E0s = str(ENERGY_OFFSET)
args.train_file = path_data

args.seed = config.metadata.seed

# Dirs
args.work_dir = str(OUTPUTS_DIR / config.metadata.experiment_name)
args.checkpoints_dir = OUTPUTS_DIR / config.metadata.experiment_name / "checkpoints"
args.results_dir = OUTPUTS_DIR / config.metadata.experiment_name / "results"
args.model_dir = OUTPUTS_DIR / config.metadata.experiment_name / "models"
args.log_dir = OUTPUTS_DIR / config.metadata.experiment_name / "logs"

# Keep every checkpoint saved during training (one per eval_interval epoch)
args.keep_checkpoints = True
args.save_all_checkpoints = True

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
config.send_model_train_to_hf()
