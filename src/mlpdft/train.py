from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

from config import Mace_TrainerConfig
from constants import (
    CHECKPOINTS_DIR,
    ENERGY_OFFSET,
    LIF_KJPAW_GROUP,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)

config_poc = Mace_TrainerConfig(
    model_key="0-small",
    group=LIF_KJPAW_GROUP,
    experiment_name="small_first_second",
    r_max=0.1,
    device="cpu",
    max_num_epochs=2,
    batch_size=1,
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

# DEVICE
args.device = config.device

# MODEL ARCHITECTURE
args.model = "MACE"
args.r_max = config.r_max
args.num_channels = 128
args.max_L = 1

# DATA KEYS
args.energy_key = config.energy_key
args.force_key = config.force_key

# DATASET
args.pin_memory = config.pin_memory
args.E0s = str(ENERGY_OFFSET)
args.train_file = str(config.data_out_path)

# LOSS AND VALIDATION
args.valid_batch_size = config.valid_batch_size
args.valid_frac = config.valid_frac
args.batch_size = config.batch_size
args.max_num_epochs = config.max_num_epochs

# Fine-tuning
args.foundation_model = config.model.path
args.lora = True
args.lora_rank = 8

# Train the model
run(args)
