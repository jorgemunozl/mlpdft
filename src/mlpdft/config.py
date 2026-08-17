from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from io import StringIO
from pathlib import Path
from typing import Literal

from ase import Atoms
from ase.io import read
from huggingface_hub import HfApi, create_repo, snapshot_download

from mlpdft.constants import (
    DATA_DIR,
    MODEL_REGISTRY,
    OUTPUTS_DIR,
    PREDICTION_DIR,
    PREFIX_HF,
    SRC_DIR,
    XYZ_DIR,
    ModelSpec,
)


@dataclass
class MaceConfig:
    """
    Settings for a MACE model evaluation.
    """

    group: str = field(
        default="",
        metadata={"description": "Group name e.g. LiF64_kjpaw"},
    )

    model_key: Literal["mace_omat_lora_v1", "mace_omat_medium"] = field(
        default="mace_omat_medium",
        metadata={"description": "MACE model key"},
    )

    model: ModelSpec = field(
        init=False,
        metadata={"description": "Key, path and name of the MACE model"},
    )

    energy_offset_per_atom: float | None = field(
        default=None,
        metadata={
            "description": "Optional energy shift (eV/atom). If None, use model default."
        },
    )

    resolved_energy_offset_per_atom: float = field(
        init=False,
        metadata={"description": "Final energy shift applied at inference (eV/atom)"},
    )

    data_in_path: Path = field(
        default=Path("LiF64_kjpaw.out"),
        metadata={"description": "Path to Quantum ESPRESSO .out file"},
    )

    data_out_path: Path = field(
        default=Path("out.extxyz"),
        metadata={"description": "Output multi-frame extxyz path"},
    )

    model_output: Path = field(
        default=Path(PREDICTION_DIR / "pred_LIF64_10_20.extxyz"),
        metadata={"description": "Output path"},
    )

    frame_stride: int | None = field(
        default=10,
        metadata={"description": "Keep one frame every N parsed frames (>=1)"},
    )

    max_frames: int | None = field(
        default=0,
        metadata={"description": "Optional cap on written frames after striding"},
    )

    node_energy: bool = field(
        default=True,
        metadata={"description": "Compute node energy"},
    )

    device: Literal["cpu", "cuda"] = field(
        default="cpu",
        metadata={"description": "Device to use for training"},
    )

    dtype: Literal["float32", "float64"] = "float32"

    config_type: str = field(
        default="",
        metadata={
            "description": "Config type label written into each frame (defaults to group name)"
        },
    )

    include_stress: bool = field(
        default=False,
        metadata={"description": "Whether to include stress tensor from QE output"},
    )

    _raw_frames: list | None = field(default=None, init=False, repr=False)

    def read_raw_frames(self) -> list:
        """Parse the QE .out file once and cache the raw frame list."""
        if self._raw_frames is not None:
            return self._raw_frames
        print(f"[read_raw_frames] {self.data_in_path}")
        if not self.data_in_path.exists():
            self._raw_frames = []
            return []
        text = self.data_in_path.read_text(encoding="latin-1")
        raw = read(StringIO(text), format="espresso-out", index=":")
        self._raw_frames = [raw] if isinstance(raw, Atoms) else list(raw)
        return self._raw_frames

    @staticmethod
    def obtain_max_frames(data_in_path: Path) -> int | None:
        """
        Count the number of valid frames in a Quantum ESPRESSO pw.x output file
        by parsing it with ASE's espresso-out reader.

        Returns the total frame count, or None if the file does not exist.
        """
        print(f"[obtain_max_frames] {data_in_path}")
        if not data_in_path.exists():
            return None
        text = data_in_path.read_text(encoding="latin-1")
        raw = read(StringIO(text), format="espresso-out", index=":")
        frames = [raw] if isinstance(raw, Atoms) else list(raw)
        return len(frames)

    def __post_init__(self):
        # Valid that model exist
        self.model = MODEL_REGISTRY[self.model_key]

        # Validate model path
        if not self.model.path.exists():
            print(f"Model path does not exist: {self.model.path}")
            print(f"Downloading model: {self.model.path}")
            self.download_model()

        self.solve_paths()
        self.resolved_energy_offset_per_atom = self.energy_offset_per_atom or 0.0

    def download_model(self):
        if self.model.hf_id is not None:
            dir = OUTPUTS_DIR / self.model_key
            os.makedirs(dir, exist_ok=True)
            model = snapshot_download(
                repo_id=self.model.hf_id,
                repo_type="model",
                local_dir=str(dir),
            )

    def solve_paths(self):
        if not self.config_type:
            self.config_type = self.group
        self.data_in_path = DATA_DIR / self.group / Path(str(self.group) + ".out")
        if self.frame_stride is None:
            self.frame_stride = 1
        if self.max_frames is None:
            raw = self.read_raw_frames()
            if raw:
                self.max_frames = int(len(raw) / self.frame_stride)
        data_out_path = (
            DATA_DIR
            / Path(self.group)
            / XYZ_DIR
            / f"{self.group}_{self.frame_stride}_{self.max_frames}.extxyz"
        )
        self.data_out_path = data_out_path

        # MACE MODEL PREDICTIONS
        model_output_path = (
            PREDICTION_DIR
            / self.group
            / Path(
                str(
                    self.model_key
                    + "_"
                    + self.group
                    + f"_{self.frame_stride}_{self.max_frames}"
                )
                + ".extxyz"
            )
        )
        os.makedirs(model_output_path.parent, exist_ok=True)
        self.model_output = model_output_path

    def validate(self) -> None:
        if self.frame_stride is not None and self.frame_stride <= 0:
            raise ValueError("frame_stride must be >= 1")

    @classmethod
    def describe_fields(cls) -> dict[str, str]:
        """Return a field->description map for documentation or logging."""
        return {f.name: f.metadata.get("description", "") for f in fields(cls)}


@dataclass
class ActiveLearningConfig(MaceConfig):
    # ── committee ────────────────────────────────────────────────
    model_paths: list[str] = field(
        default_factory=list,
        metadata={
            "description": (
                "Paths to 2+ MACE .model files.  May also include registry "
                "keys (e.g. 'mace_omat_medium') which are resolved to paths. "
                "A committee is formed when 2+ models are provided."
            )
        },
    )

    # ── input / output ───────────────────────────────────────────
    config: str = field(
        default="",
        metadata={"description": "Initial XYZ configuration file"},
    )
    output: str = field(
        default="",
        metadata={"description": "Output trajectory (.extxyz). Appends if it exists."},
    )

    # ── MD parameters ────────────────────────────────────────────
    temperature_K: float = field(
        default=300.0,
        metadata={"description": "Temperature (Kelvin)"},
    )
    timestep: float = field(
        default=1.0,
        metadata={"description": "Integration timestep (fs)"},
    )
    friction: float = field(
        default=0.01,
        metadata={"description": "Langevin friction coefficient (1/fs)"},
    )
    nsteps: int = field(
        default=1_000,
        metadata={"description": "Maximum number of MD steps"},
    )

    # ── active-learning control ──────────────────────────────────
    error_threshold: float = field(
        default=0.3,
        metadata={
            "description": (
                "Maximum relative force standard deviation across the committee "
                "before stopping the MD (higher = more tolerant)"
            )
        },
    )
    config_index: int = field(
        default=-1,
        metadata={"description": "Frame index in the XYZ file (-1 = last frame)"},
    )

    # ── I/O control ──────────────────────────────────────────────
    nsave: int = field(
        default=10,
        metadata={"description": "Save trajectory frame every N steps"},
    )
    nprint: int = field(
        default=10,
        metadata={"description": "Print status every N steps"},
    )
    ncheckerror: int = field(
        default=10,
        metadata={"description": "Check uncertainty every N steps"},
    )
    compute_stress: bool = field(
        default=False,
        metadata={"description": "Whether to compute stress"},
    )
    info_prefix: str = field(
        default="MACE_",
        metadata={"description": "Prefix for energy, forces and stress keys"},
    )

    # ── misc ─────────────────────────────────────────────────────
    seed: int = field(
        default=123,
        metadata={"description": "RNG seed for velocity initialisation"},
    )

    # ── post-init ────────────────────────────────────────────────
    def __post_init__(self):
        # Skip MaceConfig.__post_init__ — we don't need QE paths.
        # Resolve model paths through MODEL_REGISTRY if keys are used.
        self._resolve_model()

    def _resolve_model(self) -> None:
        """Expand registry keys in *model_paths* to real paths."""
        resolved: list[str] = []
        for item in self.model_paths:
            if item in MODEL_REGISTRY:
                spec = MODEL_REGISTRY[item]
                resolved.append(str(spec.path.expanduser()))
            else:
                resolved.append(item)
        self.model_paths = resolved

    # ── conversion ───────────────────────────────────────────────
    def to_namespace(self):
        """Produce an ``argparse.Namespace`` consumable by
        ``mace.cli.active_learning_md.run()``."""
        import argparse as _argparse

        ns = _argparse.Namespace()
        ns.model = self.model_paths
        ns.config = self.config
        ns.output = self.output
        ns.device = self.device
        ns.default_dtype = self.dtype
        ns.temperature_K = self.temperature_K
        ns.timestep = self.timestep
        ns.friction = self.friction
        ns.nsteps = self.nsteps
        ns.error_threshold = self.error_threshold
        ns.config_index = self.config_index
        ns.nsave = self.nsave
        ns.nprint = self.nprint
        ns.ncheckerror = self.ncheckerror
        ns.compute_stress = self.compute_stress
        ns.info_prefix = self.info_prefix
        return ns


@dataclass
class MaceTrainingHyperparams:
    """Core hyperparameters for MACE fine-tuning (model architecture, loss, optimizer, scheduler, etc.)."""

    # Model architecture
    r_max: float = field(
        default=5.0,
        metadata={"description": "Cutoff radius (Å)"},
    )
    num_channels: int = field(
        default=128,
        metadata={"description": "Number of message-passing channels"},
    )
    max_L: int = field(
        default=1,
        metadata={
            "description": "Maximum spherical harmonics L (0=scalars, 1=vectors)"
        },
    )
    max_ell: int = field(
        default=3,
        metadata={"description": "Maximum spherical harmonics ell order"},
    )
    num_interactions: int = field(
        default=2,
        metadata={"description": "Number of interaction blocks"},
    )
    correlation: int = field(
        default=3,
        metadata={"description": "Body order (3 = 4-body)"},
    )
    num_radial_basis: int = field(
        default=8,
        metadata={"description": "Number of radial basis functions"},
    )
    num_cutoff_basis: int = field(
        default=5,
        metadata={"description": "Number of cutoff basis functions"},
    )

    # Data keys
    energy_key: str = field(
        default="REF_energy",
        metadata={"description": "Key for reference energies in the dataset"},
    )
    force_key: str = field(
        default="REF_forces",
        metadata={"description": "Key for reference forces in the dataset"},
    )
    train_file: str = field(
        default="",
        metadata={"description": "Path to training data file"},
    )
    valid_file: str = field(
        default="",
        metadata={"description": "Path to validation data file"},
    )

    # Batch / epoch settings
    batch_size: int = field(
        default=10,
        metadata={"description": "Training batch size"},
    )
    max_num_epochs: int = field(
        default=300,
        metadata={"description": "Maximum number of training epochs"},
    )
    valid_batch_size: int = field(
        default=1,
        metadata={"description": "Validation batch size"},
    )
    valid_frac: float = field(
        default=0.1,
        metadata={"description": "Fraction of data held out for validation"},
    )
    pin_memory: bool = field(
        default=False,
        metadata={"description": "Pin memory for DataLoader"},
    )

    # Loss weights
    loss: str = field(
        default="weighted",
        metadata={"description": "Loss function type"},
    )
    forces_weight: float = field(
        default=1.0,
        metadata={"description": "Weight of the force contribution to the loss"},
    )
    energy_weight: float = field(
        default=1.0,
        metadata={"description": "Weight of the energy contribution to the loss"},
    )

    # Optimizer
    optimizer: str = field(
        default="adam",
        metadata={"description": "Optimizer name"},
    )
    lr: float = field(
        default=0.01,
        metadata={"description": "Learning rate"},
    )
    amsgrad: bool = field(
        default=True,
        metadata={"description": "Use AMSGrad variant of Adam"},
    )
    weight_decay: float = field(
        default=5e-7,
        metadata={"description": "L2 weight decay regularization"},
    )
    clip_grad: float = field(
        default=10.0,
        metadata={"description": "Gradient clipping threshld"},
    )

    # Scheduler
    scheduler: str = field(
        default="ReduceLROnPlateau",
        metadata={"description": "Learning rate scheduler type"},
    )
    lr_factor: float = field(
        default=0.8,
        metadata={"description": "LR scheduler decay factor"},
    )
    scheduler_patience: int = field(
        default=50,
        metadata={"description": "Epochs to wait before LR decay"},
    )

    # Stage two (SWA)
    swa: bool = field(
        default=True,
        metadata={"description": "Enable stage two (SWA)"},
    )
    start_swa: int = field(
        default=240,
        metadata={"description": "Epoch to begin stage two"},
    )
    swa_lr: float = field(
        default=1e-3,
        metadata={"description": "Learning rate for stage two"},
    )

    # EMA
    ema: bool = field(
        default=True,
        metadata={"description": "Enable exponential moving average of weights"},
    )
    ema_decay: float = field(
        default=0.99,
        metadata={"description": "EMA decay rate"},
    )

    # Early stopping
    patience: int = field(
        default=100,
        metadata={"description": "Patience for early stopping (epochs)"},
    )
    eval_interval: int = field(
        default=1,
        metadata={"description": "Evaluation interval (epochs)"},
    )

    # LoRA fine-tuning
    lora: bool = field(
        default=True,
        metadata={"description": "Enable LoRA fine-tuning"},
    )
    lora_rank: int = field(
        default=8,
        metadata={"description": "Rank of LoRA decomposition"},
    )


@dataclass
class MaceTrainingMetadata:
    """Metadata about the training run."""

    experiment_name: str = field(
        default="experiment",
        metadata={"description": "Name of the experiment (used for output files)"},
    )
    seed: int = field(
        default=123,
        metadata={"description": "Random seed for reproducibility"},
    )
    config_path: str = field(
        default="",
        metadata={"description": "Path to save / load the training config JSON"},
    )
    model_dir: str = field(
        default="models",
        metadata={"description": "Directory for model checkpoints and results"},
    )


@dataclass
class Mace_TrainerConfig(MaceConfig):
    """Training configuration that composes hyperparameters and metadata.

    Inherits model selection, device, dtype, and evaluation-oriented fields
    from MaceConfig.
    """

    hyperparams: MaceTrainingHyperparams = field(
        default_factory=MaceTrainingHyperparams,
        metadata={"description": "Core training hyperparameters"},
    )
    metadata: MaceTrainingMetadata = field(
        default_factory=MaceTrainingMetadata,
        metadata={"description": "Run metadata (experiment name, seed, etc.)"},
    )

    def __post_init__(self):
        # Resolve model spec and download if missing (same logic as MaceConfig)
        self.model = MODEL_REGISTRY[self.model_key]
        if not self.model.path.exists():
            print(f"Model path does not exist: {self.model.path}")
            print(f"Downloading foundation model: {self.model.path}")
            self.download_model()

        # Only resolve group paths if a group was explicitly set.
        if self.group:
            self.solve_paths()

        self.metadata.config_path = str(
            (SRC_DIR / "configs" / f"{self.metadata.experiment_name}.json")
        )

    def write_config_train(self):
        """Serialize training config (hyperparams + metadata) to a JSON file."""
        config_dict = {
            "hyperparams": asdict(self.hyperparams),
            "metadata": asdict(self.metadata),
            "dtype": self.dtype,
        }
        with open(self.metadata.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config_train(cls, path_json: Path) -> Mace_TrainerConfig:
        """Load training config from a JSON file and return a new instance."""
        with open(path_json, "r") as f:
            config_dict = json.load(f)
        hyperparams = MaceTrainingHyperparams(**config_dict["hyperparams"])
        metadata = MaceTrainingMetadata(**config_dict["metadata"])
        dtype = config_dict.get("dtype", "float64")
        return cls(hyperparams=hyperparams, metadata=metadata, dtype=dtype)

    def send_model_train_to_hf(self):
        # Send model, logs, resumes, config, everything I have!
        api = HfApi()

        HF_MODEL_REPO_ID = PREFIX_HF + "/" + self.metadata.experiment_name
        repo_url = create_repo(
            repo_id=HF_MODEL_REPO_ID,
            repo_type="model",
            exist_ok=True,
        )
        print(f"[hf] Model repo ready → {repo_url}")

        training_files = list(Path().glob(f"{self.metadata.experiment_name}*"))
        for file in training_files:
            api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=file.name,
                repo_id=HF_MODEL_REPO_ID,
                repo_type="model",
            )
            print(f"[hf] Uploaded {file.name}")
