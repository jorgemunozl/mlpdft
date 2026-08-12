#!/usr/bin/env python3
"""Proof of concept: baseline vs. committee vs. snapshot ensemble.

All runs use CosineAnnealingWarmRestarts — the only difference is T_0:
  - baseline / committee:  T_0 = 100  (single cosine decay, no restart)
  - snapshot (ours):       T_0 = 25   (4 warm-restart cycles)

Runs 5 trainings sequentially on the test dataset:
  1 × baseline
  3 × committee  (seeds 123, 124, 125)
  1 × snapshot
"""

import os
import warnings
from pathlib import Path

import torch

# wandb offline mode: metrics are written locally (never synced to the cloud).
# Sync later with:  wandb sync <run_dir>
os.environ["WANDB_MODE"] = "offline"

warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*not available.*")

from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

# ---------------------------------------------------------------------------
# Inject CosineAnnealingWarmRestarts into MACE's LRScheduler
# ---------------------------------------------------------------------------
from mace.tools.scripts_utils import LRScheduler as _LRScheduler

_original_init = _LRScheduler.__init__
_original_step = _LRScheduler.step


def _patched_init(self, optimizer, args):
    if args.scheduler == "CosineAnnealingWarmRestarts":
        self.scheduler = args.scheduler
        self._optimizer_type = args.optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=getattr(args, "cosine_T_0", 20),
            T_mult=getattr(args, "cosine_T_mult", 1),
            eta_min=getattr(args, "cosine_eta_min", 1e-6),
        )
    else:
        _original_init(self, optimizer, args)


def _patched_step(self, metrics=None, epoch=None):
    if self.scheduler == "CosineAnnealingWarmRestarts":
        self.lr_scheduler.step(epoch=epoch)
    else:
        _original_step(self, metrics=metrics, epoch=epoch)


_LRScheduler.__init__ = _patched_init
_LRScheduler.step = _patched_step
# ---------------------------------------------------------------------------

from huggingface_hub import hf_hub_download

from mlpdft.config import (
    Mace_TrainerConfig,
    MaceTrainingHyperparams,
    MaceTrainingMetadata,
)
from mlpdft.constants import (
    DATA_DIR,
    ENERGY_OFFSET,
    OUTPUTS_DIR,
    PREFIX_HF,
    TEST_DATASET_NAME,
    XYZ_DIR,
)

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_FILE = DATA_DIR / XYZ_DIR / f"{TEST_DATASET_NAME}.extxyz"


def ensure_dataset() -> str:
    """Download the test dataset from Hugging Face if missing locally."""
    if TRAIN_FILE.exists():
        return str(TRAIN_FILE)

    print(f"Dataset not found: {TRAIN_FILE}")
    print(f"Downloading from HF: {PREFIX_HF}/{TEST_DATASET_NAME} ...")
    try:
        local = hf_hub_download(
            repo_id=f"{PREFIX_HF}/{TEST_DATASET_NAME}",
            filename=f"{TEST_DATASET_NAME}.extxyz",
            repo_type="dataset",
            local_dir=str(DATA_DIR / XYZ_DIR),
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Download failed: {exc}. Run upload_ds_hf.py first."
        ) from exc
    print(f"Dataset ready: {local}")
    return local

# ── Shared config (architecture, training, optimizer) ─────────────────────
SHARED_HYPERPARAMS = dict(
    r_max=6.0,
    max_num_epochs=100,
    batch_size=16,
    patience=999,
    eval_interval=1,
    valid_frac=0.15,
    swa=False,
    num_channels=128,
    num_cutoff_basis=5,
    max_L=1,
    max_ell=3,
    num_interactions=2,
    correlation=3,
    num_radial_basis=8,
    lora=False,
    ema=False,
    lr=5e-4,
)


def _apply_common_args(args, config, seed: int) -> None:
    """Map all shared hyperparams onto the argparse namespace."""
    hp = config.hyperparams
    args.E0s = str(ENERGY_OFFSET)
    args.train_file = str(TRAIN_FILE)
    args.seed = seed
    args.work_dir = str(OUTPUTS_DIR / config.metadata.experiment_name)
    args.checkpoints_dir = str(
        OUTPUTS_DIR / config.metadata.experiment_name / "checkpoints"
    )
    args.results_dir = str(
        OUTPUTS_DIR / config.metadata.experiment_name / "results"
    )
    args.model_dir = str(
        OUTPUTS_DIR / config.metadata.experiment_name / "models"
    )
    args.log_dir = str(
        OUTPUTS_DIR / config.metadata.experiment_name / "logs"
    )
    args.keep_checkpoints = True
    args.save_all_checkpoints = True
    args.foundation_model = str(config.model.path)
    args.default_dtype = config.dtype
    args.device = config.device
    args.model = "MACE"
    args.lora = hp.lora
    args.r_max = hp.r_max
    args.num_channels = hp.num_channels
    args.max_L = hp.max_L
    args.max_ell = hp.max_ell
    args.num_interactions = hp.num_interactions
    args.correlation = hp.correlation
    args.num_radial_basis = hp.num_radial_basis
    args.num_cutoff_basis = hp.num_cutoff_basis
    args.energy_key = hp.energy_key
    args.force_key = hp.force_key
    args.pin_memory = hp.pin_memory
    args.loss = hp.loss
    args.forces_weight = hp.forces_weight
    args.energy_weight = hp.energy_weight
    args.valid_batch_size = hp.valid_batch_size
    args.valid_frac = hp.valid_frac
    args.batch_size = hp.batch_size
    args.max_num_epochs = hp.max_num_epochs
    args.optimizer = hp.optimizer
    args.lr = hp.lr
    args.amsgrad = hp.amsgrad
    args.weight_decay = hp.weight_decay
    args.clip_grad = hp.clip_grad
    args.swa = hp.swa
    args.start_swa = hp.start_swa
    args.swa_lr = hp.swa_lr
    args.ema = hp.ema
    args.ema_decay = hp.ema_decay
    args.patience = hp.patience
    args.eval_interval = hp.eval_interval

    # ── wandb (offline) ──
    args.wandb = True
    args.wandb_project = "mlpdft_poc"
    args.wandb_name = config.metadata.experiment_name
    args.wandb_dir = str(
        OUTPUTS_DIR / config.metadata.experiment_name / "wandb"
    )



def train_one(
    experiment_name: str,
    seed: int,
    scheduler: str,
    **scheduler_kwargs,
) -> None:
    """Run a single MACE training with the shared config + given scheduler."""
    print(f"\n{'=' * 60}")
    print(f"  {experiment_name}  |  seed={seed}  |  scheduler={scheduler}")
    print(f"{'=' * 60}")

    config = Mace_TrainerConfig(
        model_key="mace_omat_medium",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float64",  # must match foundation model dtype (mace_omat_medium is float64)
        hyperparams=MaceTrainingHyperparams(**SHARED_HYPERPARAMS),
        metadata=MaceTrainingMetadata(
            experiment_name=experiment_name,
            seed=seed,
        ),
    )

    parser = build_default_arg_parser()
    args = parser.parse_args(["--name", experiment_name])

    _apply_common_args(args, config, seed)

    # ── Scheduler ──
    args.scheduler = scheduler
    for k, v in scheduler_kwargs.items():
        setattr(args, k, v)

    run(args)
    config.write_config_train()


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    dataset_path = ensure_dataset()
    print(f"Training on: {dataset_path}")

    # ── Common cosine scheduler kwargs (no warm restarts) ──
    cosine_single = dict(
        cosine_T_0=100,      # one full cosine decay across the 100 epochs
        cosine_T_mult=1,
        cosine_eta_min=1e-6,
    )
    # ── Snapshot: warm restarts every 33 epochs → 3 cycles ──
    cosine_cyclic = dict(
        cosine_T_0=33,       # restart every 33 epochs
        cosine_T_mult=1,     # equal-length cycles
        cosine_eta_min=1e-6,
    )

    # 1 ── Baseline (cosine decay, single basin) ──
    train_one(
        "baseline_ft", seed=123,
        scheduler="CosineAnnealingWarmRestarts",
        **cosine_single,
    )

    # 2 ── Committee (3 seeds, each single cosine decay) ──
    for seed in (123, 124, 125):
        train_one(
            f"committee_s{seed}", seed=seed,
            scheduler="CosineAnnealingWarmRestarts",
            **cosine_single,
        )

    # 3 ── Snapshot (warm restarts → 3 basins from 1 run) ──
    train_one(
        "snapshot_warm", seed=123,
        scheduler="CosineAnnealingWarmRestarts",
        **cosine_cyclic,
    )

    print("\nDone — 5 trainings complete.")


if __name__ == "__main__":
    main()
