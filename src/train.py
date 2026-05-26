from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

from config import Mace_TrainerConfig
from constants import LIF_KJPAW_GROUP

config_poc = Mace_TrainerConfig(
    model_key="0-small",
    group=LIF_KJPAW_GROUP,
    experiment_name="small_first",
    r_max=0.1,
    device="cpu",
    max_num_epochs=2,
    batch_size=1,
)

config = config_poc

parser = build_default_arg_parser()
args = parser.parse_args(["--name", config.experiment_name])

args.device = config.device
args.seed = 123
args.model = "MACE"
args.r_max = config.r_max
args.num_channels = 128
args.max_L = 1
args.batch_size = config.batch_size
args.max_num_epochs = config.max_num_epochs

args.device = config.device
args.default_dtype = "float32"

args.foundation_model = config.model.path
args.lora = True
args.lora_rank = 8

args.train_file = str(config.data_out_path)

run(args)
