#!/usr/bin/env python3
"""Train and evaluate UQ ensembles for a fine-tuned MACE foundation model.

Requires curated, block-split train/validation/test extxyz files. Trains an
independent committee, a warm-snapshot run, a multi-head bootstrap model, and
optionally the frozen-body multi-head committee (MHC) of Beck et al.
(arXiv:2508.09907) plus a frozen-body point estimate, then reports DFT test
errors and uncertainty-ranking metrics.

The MHC arm freezes the atomic-environment descriptors (embedding, interaction
and product blocks) and trains only randomly initialised readout heads on
disjoint slices of the training set. Select it with ``--arms mhc``; it does not
use the released ``multiheads_finetuning`` replay path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*not available.*")

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from huggingface_hub import snapshot_download

import mace.cli.run_train as mace_run_train
from mace.calculators.mace import MACECalculator
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser
from mace.tools.checkpoint import CheckpointIO as _CheckpointIO
from mace.tools.scripts_utils import LRScheduler as _LRScheduler
from mace.tools.scripts_utils import remove_pt_head

from mlpdft.config import (
    Mace_TrainerConfig,
    MaceTrainingHyperparams,
    MaceTrainingMetadata,
)
from mlpdft.constants import (
    DATA_DIR,
    ENERGY_KEY,
    ENERGY_OFFSET,
    FORCE_KEY,
    OUTPUTS_DIR,
    PREFIX_HF,
    XYZ_DIR,
)

_original_scheduler_init = _LRScheduler.__init__
_original_scheduler_step = _LRScheduler.step


def _patched_scheduler_init(self, optimizer, args):
    if args.scheduler == "CosineAnnealingWarmRestarts":
        self.scheduler = args.scheduler
        self._optimizer_type = args.optimizer
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        self.lr_scheduler = scheduler_class(
            optimizer=optimizer,
            T_0=getattr(args, "cosine_T_0", 20),
            T_mult=getattr(args, "cosine_T_mult", 1),
            eta_min=getattr(args, "cosine_eta_min", 1e-6),
        )
    else:
        _original_scheduler_init(self, optimizer, args)


def _patched_scheduler_step(self, metrics=None, epoch=None):
    if self.scheduler == "CosineAnnealingWarmRestarts":
        self.lr_scheduler.step(epoch=epoch)
    else:
        _original_scheduler_step(self, metrics=metrics, epoch=epoch)


_LRScheduler.__init__ = _patched_scheduler_init
_LRScheduler.step = _patched_scheduler_step


# --- Checkpoint retention ---------------------------------------------------
# With warm restarts MACE writes a checkpoint at every validation epoch.  Only
# the cycle-end states are needed to build the snapshot ensemble, so skip the
# other writes; the few surviving files are deleted after export.  The active
# set is reset for every run so a snapshot run cannot leak into the next arm.
_ACTIVE_CHECKPOINT_EPOCHS: set[int] | None = None
_original_checkpoint_save = _CheckpointIO.save


def _selective_checkpoint_save(self, checkpoint, epochs, keep_last=False):
    if (
        _ACTIVE_CHECKPOINT_EPOCHS is not None
        and epochs not in _ACTIVE_CHECKPOINT_EPOCHS
    ):
        return
    return _original_checkpoint_save(self, checkpoint, epochs, keep_last)


_CheckpointIO.save = _selective_checkpoint_save


# --- Frozen-body guards -----------------------------------------------------
# ``--freeze`` and ``--foundation_filter_elements`` are consumed inside MACE's
# ``run_train``.  Wrap the two call sites so a frozen-body run cannot silently
# train the descriptors, and cannot copy the foundation readout into the new
# committee heads (which would make every head start identical).
_original_configure_model = mace_run_train.configure_model
_original_get_params_options = mace_run_train.get_params_options

_DESCRIPTOR_PREFIXES = (
    "node_embedding",
    "interactions",
    "products",
    "radial_embedding",
    "spherical_harmonics",
)


def _checked_configure_model(*args, **kwargs):
    model, output_args = _original_configure_model(*args, **kwargs)
    run_args = args[0] if args else kwargs.get("args")
    if getattr(run_args, "_uq_frozen_body", False):
        if run_args.freeze != FROZEN_BODY_LAYERS:
            raise RuntimeError(
                f"Frozen-body run expected freeze={FROZEN_BODY_LAYERS}, "
                f"got {run_args.freeze!r}"
            )
        if run_args.foundation_filter_elements and not getattr(
            run_args, "_uq_warm_start_heads", False
        ):
            raise RuntimeError(
                "Frozen-body run must set foundation_filter_elements=False so "
                "the new heads start from random weights (use "
                "--warm-start-heads to opt in)"
            )
        if getattr(run_args, "multiheads_finetuning", False):
            raise RuntimeError(
                "Frozen-body run must not enable multiheads_finetuning (replay)"
            )
        for name, parameter in model.named_parameters():
            if name.split(".")[0] in _DESCRIPTOR_PREFIXES:
                parameter.requires_grad = False
        foundation = next(
            (
                item
                for item in args
                if hasattr(item, "readouts") and hasattr(item, "heads")
            ),
            None,
        )
        # load_foundations_elements copies the bessel weights and the message
        # passing blocks but not the Agnesi distance transform.  Without this
        # the frozen "body" is not the foundation body, and a warm-started
        # readout still sees different descriptors.
        if foundation is not None:
            source_dt = getattr(foundation.radial_embedding, "distance_transform", None)
            target_dt = getattr(model.radial_embedding, "distance_transform", None)
            if (
                source_dt is not None
                and target_dt is not None
                and type(source_dt) is type(target_dt)
            ):
                missing, unexpected = target_dt.load_state_dict(
                    source_dt.state_dict(), strict=False
                )
                if missing or unexpected:
                    print(
                        "[uq_compare] warning: distance transform not fully "
                        f"copied (missing={missing}, unexpected={unexpected})"
                    )
        if getattr(run_args, "_uq_warm_start_readout", False):
            if foundation is None:
                raise RuntimeError(
                    "warm start requested but no foundation model reached "
                    "configure_model"
                )
            if len(model.heads) != 1:
                raise RuntimeError(
                    "--warm-start-readout is only valid for the single-head "
                    f"probe, got {len(model.heads)} heads: {model.heads}"
                )
            # ScaleShiftMACE keeps a two-stage readout per head (an atomic
            # energy linear followed by the interaction readout), so copying
            # readouts[0] alone leaves the force-producing stage random.
            if len(foundation.readouts) != len(model.readouts):
                raise RuntimeError(
                    "warm start: foundation and probe have different readout "
                    f"stacks ({len(foundation.readouts)} vs "
                    f"{len(model.readouts)})"
                )
            for index, (source, target) in enumerate(
                zip(foundation.readouts, model.readouts)
            ):
                missing, unexpected = target.load_state_dict(
                    source.state_dict(), strict=False
                )
                if missing or unexpected:
                    raise RuntimeError(
                        f"warm-start readout stage {index} does not match the "
                        f"foundation: missing={missing}, "
                        f"unexpected={unexpected}"
                    )
            if hasattr(model, "scale_shift") and hasattr(foundation, "scale_shift"):
                # The foundation has one scalar scale/shift; a fresh multi-head
                # model allocates one entry per head.  Broadcast the scalar.
                target_state = model.scale_shift.state_dict()
                for key, value in foundation.scale_shift.state_dict().items():
                    if key not in target_state:
                        raise RuntimeError(
                            f"warm start: unexpected scale_shift key {key!r}"
                        )
                    target = target_state[key]
                    value = value.to(device=target.device, dtype=target.dtype)
                    if target.shape == value.shape:
                        target.copy_(value)
                    elif value.numel() == 1:
                        target.copy_(value.expand_as(target))
                    else:
                        raise RuntimeError(
                            "warm start: scale_shift shape mismatch for "
                            f"{key!r}: {tuple(value.shape)} vs "
                            f"{tuple(target.shape)}"
                        )
            print(
                "[uq_compare] warm-started the single probe readout from the "
                "foundation model"
            )
        print(
            f"[uq_compare] frozen body: {len(model.heads)} head(s) "
            f"{model.heads}, descriptors frozen, readouts trainable"
        )
    return model, output_args


def _checked_get_params_options(run_args, model):
    options = _original_get_params_options(run_args, model)
    if getattr(run_args, "_uq_frozen_body", False):
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        leaked = [
            name for name in trainable if name.split(".")[0] in _DESCRIPTOR_PREFIXES
        ]
        if leaked:
            raise RuntimeError(
                f"Frozen-body run has trainable descriptor tensors: {leaked[:5]}"
            )
        if not trainable:
            raise RuntimeError("Frozen-body run has no trainable parameters")
        unexpected = [
            name for name in trainable
            if not name.startswith(("readouts", "scale_shift"))
        ]
        if unexpected:
            raise RuntimeError(
                "Frozen-body run has trainable tensors outside the readouts: "
                f"{unexpected[:5]}"
            )
        n_trainable = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        n_total = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"[uq_compare] frozen-body check passed: {n_trainable:,} trainable "
            f"of {n_total:,} parameters ({100 * n_trainable / n_total:.2f}%)"
        )
    return options


mace_run_train.configure_model = _checked_configure_model
mace_run_train.get_params_options = _checked_get_params_options


DEFAULT_SEEDS = (123, 124, 125)
TARGET_HEADS = ("uq_head_123", "uq_head_124", "uq_head_125")
MHC_HEAD_TEMPLATE = "mhc_head_{index}"
FROZEN_BODY_LAYERS = 6
# Frozen-body readouts start from random weights and must absorb a large
# cohesive-energy offset, so the 1:1 energy:force objective is energy-dominated
# and force error grows during training (lif_uq_mhc_001).  The defaults below
# apply only to the frozen arms; the Exp 2 arms keep 1:1.
FROZEN_FORCES_WEIGHT = 100.0
FROZEN_PATIENCE = 50
SMOKE_LEARNING_DROP = 0.10
SMOKE_MAX_REGRESSION_FROM_BEST = 0.05
SMOKE_ALREADY_GOOD_RMSE = 0.05

# Run defaults: the runner needs no flags. The block-split inputs live under
# ``dataset/xyz_files/uq_splits`` and are auto-downloaded from the HF dataset
# repo when missing. The no-flag default is the current experiment: the
# frozen-body MHC plus its point-estimate control. Previous-result merging is
# automatic when the Exp 2 run directory is present.
DEFAULT_SPLIT_DIR = DATA_DIR / XYZ_DIR / "uq_splits"
DEFAULT_RUN_ID = "lif_uq_mhc_001"
PREVIOUS_RUN_ID = "lif_uq_001"

UQ_DATASET_REPO = f"{PREFIX_HF}/li_f_uq_splits_v2"
UQ_SPLIT_FILENAMES = ("uq_train.extxyz", "uq_valid.extxyz", "uq_test.extxyz")


def _ensure_split_files(paths: list[Path]) -> None:
    """Download any missing split file from the UQ dataset repo on the Hub.

    Private repo: requires ``HF_TOKEN`` (or ``huggingface-cli login``). Skips the
    network entirely when every file is already present locally.
    """
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    print(f"Downloading UQ splits from {UQ_DATASET_REPO} ...")
    allow_patterns = [*UQ_SPLIT_FILENAMES, "uq_manifest.json"]
    for parent in {path.parent for path in missing}:
        parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=UQ_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(parent),
            allow_patterns=allow_patterns,
        )
    still_missing = [path for path in paths if not path.is_file()]
    if still_missing:
        raise FileNotFoundError(
            f"Could not obtain split file(s) from {UQ_DATASET_REPO}: {still_missing}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compare foundation-model UQ ensembles on held-out DFT data.")
    )
    parser.add_argument(
        "--train-file", type=Path, default=DEFAULT_SPLIT_DIR / "uq_train.extxyz"
    )
    parser.add_argument(
        "--valid-file", type=Path, default=DEFAULT_SPLIT_DIR / "uq_valid.extxyz"
    )
    parser.add_argument(
        "--test-file", type=Path, default=DEFAULT_SPLIT_DIR / "uq_test.extxyz"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "uq_diffusion_comparison",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=f"Run directory name (default: {DEFAULT_RUN_ID}; 'lif_uq_mhc_smoke' with --smoke-test)",
    )
    parser.add_argument("--model-key", default="mace_omat_medium")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--forces-weight",
        type=float,
        default=None,
        help=(
            "Weight of the force term in the UniversalLoss. Default: 100 for "
            "the frozen-body arms (MHC and probe), 1.0 otherwise."
        ),
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=1.0,
        help="Weight of the energy term in the UniversalLoss (default: 1.0)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Early-stopping patience on the validation loss. Default: 50 for "
            "the frozen-body arms, max_epochs + 1 (disabled) otherwise."
        ),
    )
    parser.add_argument(
        "--warm-start-readout",
        action="store_true",
        help=(
            "Initialise the single frozen-probe head from the foundation "
            "readout instead of random weights. Probe only; the four MHC heads "
            "stay random so they can differ."
        ),
    )
    parser.add_argument(
        "--warm-start-heads",
        action="store_true",
        help=(
            "Initialise every frozen MHC head from the foundation readout "
            "instead of random weights. Diversity then comes from the disjoint "
            "data slices, not from head initialisation."
        ),
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=50,
        help=(
            "Epochs per frozen arm in --smoke-test (default: 50, about 50 min "
            "on one A40). The learning gate aborts if validation force RMSE "
            "does not fall by 10%% or drifts 5%% above its best value."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valid-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--snapshot-count", type=int, default=3)
    parser.add_argument("--bootstrap-block-size", type=int, default=5)
    parser.add_argument(
        "--no-bootstrap-committee",
        action="store_false",
        dest="bootstrap_committee",
        help="Disable per-member training-data bootstrap (default: enabled)",
    )
    parser.add_argument("--pt-train-file", default="omat")
    parser.add_argument("--pt-samples", type=int, default=1000)
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "Comma-separated arms to train/evaluate: committee, snapshots, "
            "multihead-bootstrap, mhc, frozen-probe, zero-shot. Defaults to "
            "mhc,frozen-probe (Experiment 3). Pass committee,snapshots,"
            "multihead-bootstrap to reproduce Experiment 2."
        ),
    )
    parser.add_argument(
        "--mhc-heads",
        type=int,
        default=4,
        help="Number of frozen-body committee heads (default: 4)",
    )
    parser.add_argument(
        "--mhc-epochs",
        type=int,
        default=400,
        help="Epoch budget for the frozen-body multi-head committee",
    )
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=400,
        help="Epoch budget for the frozen-body single-head point estimate",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20251027,
        help="Seed for the disjoint MHC head split",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 2-epoch MHC + probe check before any full run",
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=8,
        help="Test frames used by the smoke-test checks",
    )
    parser.add_argument(
        "--include-summary",
        type=Path,
        action="append",
        default=[],
        help="Merge methods from a previous uq_summary.json (repeatable)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output run directory before starting",
    )
    parser.add_argument("--skip-multihead", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


ALL_ARMS = (
    "committee",
    "snapshots",
    "multihead-bootstrap",
    "mhc",
    "frozen-probe",
    "zero-shot",
)
DEFAULT_ARMS = ("mhc", "frozen-probe")


def parse_arms(value: str | None, smoke_test: bool = False) -> tuple[str, ...]:
    if value is None:
        return ("mhc", "frozen-probe") if smoke_test else DEFAULT_ARMS
    if value.strip().lower() == "all":
        return ALL_ARMS
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(arms) - set(ALL_ARMS))
    if unknown:
        raise ValueError(
            f"Unknown arm(s): {', '.join(unknown)}. Choose from {', '.join(ALL_ARMS)}"
        )
    if not arms:
        raise ValueError("--arms must list at least one arm")
    return arms


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Seeds must be comma-separated integers"
        ) from exc
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Provide exactly three unique committee seeds")
    return seeds


def _read_frames(path: Path, require_labels: bool = True) -> list[Atoms]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    frames = read(path, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    if not frames:
        raise ValueError(f"Dataset file contains no frames: {path}")
    if require_labels:
        for index, atoms in enumerate(frames):
            if ENERGY_KEY not in atoms.info or FORCE_KEY not in atoms.arrays:
                raise ValueError(
                    f"Frame {index} in {path} lacks {ENERGY_KEY} or {FORCE_KEY}"
                )
            forces = np.asarray(atoms.arrays[FORCE_KEY])
            if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
                raise ValueError(f"Invalid force labels in frame {index} of {path}")
            if not np.isfinite(float(atoms.info[ENERGY_KEY])):
                raise ValueError(f"Invalid energy label in frame {index} of {path}")
    return list(frames)


def _isolated_atom_energies(frames: list[Atoms]) -> dict[int, float]:
    """Restrict the isolated-atom reference to species actually in the data.

    ``ENERGY_OFFSET`` carries B as well as Li and F.  The UQ splits are Li/F
    only, and MACE builds one atomic-energy entry per species in the data, so
    passing the unfiltered dictionary fails with a 2-vs-3 shape mismatch.
    """
    present = {int(number) for atoms in frames for number in atoms.numbers}
    missing = sorted(present - set(ENERGY_OFFSET))
    if missing:
        raise ValueError(
            f"No isolated-atom reference energy for Z={missing}; add it to ENERGY_OFFSET"
        )
    return {number: ENERGY_OFFSET[number] for number in sorted(present)}


def _make_bootstrap_files(
    train_frames: list[Atoms],
    output_dir: Path,
    seeds: list[int],
    block_size: int,
    names: list[str] | None = None,
) -> dict[str, Path]:
    by_group: dict[str, list[Atoms]] = defaultdict(list)
    for atoms in train_frames:
        group = str(atoms.info.get("config_type", "all"))
        by_group[group].append(atoms)

    output_dir.mkdir(parents=True, exist_ok=True)
    names = names or list(TARGET_HEADS[: len(seeds)])
    paths: dict[str, Path] = {}
    for head_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        sampled: list[Atoms] = []
        for group_frames in by_group.values():
            size = len(group_frames)
            width = min(max(1, block_size), size)
            group_sample: list[Atoms] = []
            while len(group_sample) < size:
                start = int(rng.integers(0, size - width + 1))
                group_sample.extend(group_frames[start : start + width])
            sampled.extend(group_sample[:size])

        name = names[head_index]
        path = output_dir / f"{name}_bootstrap.extxyz"
        write(path, sampled, format="extxyz")
        paths[name] = path
    return paths


def _make_disjoint_head_files(
    train_frames: list[Atoms],
    output_dir: Path,
    n_heads: int,
    seed: int,
    names: list[str] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Split the training set into disjoint, group-stratified head files.

    Every frame is assigned to exactly one head, and the assignment is
    stratified by ``config_type`` so each environment appears in every head.
    At the project's scale (214 frames) four heads give ~53 frames per head;
    the paper used 1,000 frames per head, so this is a reduced-scale disjoint
    MHC and must be reported as such.
    """
    if n_heads < 2:
        raise ValueError("The multi-head committee needs at least two heads")
    if len(train_frames) < n_heads:
        raise ValueError(
            f"Cannot split {len(train_frames)} training frames into {n_heads} heads"
        )

    names = names or [MHC_HEAD_TEMPLATE.format(index=index) for index in range(n_heads)]
    if len(names) != n_heads or len(set(names)) != n_heads:
        raise ValueError("Head names must be unique and match --mhc-heads")

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, atoms in enumerate(train_frames):
        by_group[str(atoms.info.get("config_type", "all"))].append(index)

    rng = np.random.default_rng(seed)
    assignments: list[list[int]] = [[] for _ in range(n_heads)]
    offset = 0
    for group in sorted(by_group):
        indices = np.asarray(by_group[group], dtype=int)
        order = rng.permutation(len(indices))
        for position, local_index in enumerate(order):
            assignments[(offset + position) % n_heads].append(
                int(indices[local_index])
            )
        offset += len(indices)

    if any(not assigned for assigned in assignments):
        raise ValueError(
            "Disjoint split produced an empty head; lower --mhc-heads or add data"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    split_record: dict[str, Any] = {
        "seed": seed,
        "n_heads": n_heads,
        "n_frames_total": len(train_frames),
        "heads": {},
    }
    for name, assigned in zip(names, assignments):
        selected = sorted(assigned)
        path = output_dir / f"{name}_train.extxyz"
        write(path, [train_frames[index] for index in selected], format="extxyz")
        paths[name] = path
        group_counts: dict[str, int] = defaultdict(int)
        for index in selected:
            group_counts[str(train_frames[index].info.get("config_type", "all"))] += 1
        split_record["heads"][name] = {
            "n_frames": len(selected),
            "frame_indices": selected,
            "by_group": dict(sorted(group_counts.items())),
        }

    split_path = output_dir / "mhc_split.json"
    split_path.write_text(json.dumps(split_record, indent=2), encoding="utf-8")
    counts = {name: len(assigned) for name, assigned in zip(names, assignments)}
    print(f"Disjoint MHC head split ({len(train_frames)} frames): {counts}")
    print(f"Split manifest: {split_path}")
    return paths, split_record


def _make_hyperparams(
    args: argparse.Namespace,
    train_file: Path,
    valid_file: Path,
    epochs: int | None = None,
    forces_weight: float | None = None,
    energy_weight: float | None = None,
    patience: int | None = None,
):
    max_epochs = epochs or args.epochs
    values: dict[str, Any] = {
        "r_max": 6.0,
        "max_num_epochs": max_epochs,
        "batch_size": args.batch_size,
        "valid_batch_size": args.valid_batch_size,
        "patience": patience if patience is not None else max_epochs + 1,
        "eval_interval": 1,
        "valid_frac": 0.1,
        "loss": "universal",
        "forces_weight": forces_weight if forces_weight is not None else 1.0,
        "energy_weight": energy_weight if energy_weight is not None else 1.0,
        "swa": False,
        "num_channels": 128,
        "num_cutoff_basis": 5,
        "max_L": 1,
        "max_ell": 3,
        "num_interactions": 2,
        "correlation": 3,
        "num_radial_basis": 8,
        "lora": False,
        "ema": False,
        "lr": args.learning_rate,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
    }
    return MaceTrainingHyperparams(**values)


def _configure_run(
    args: argparse.Namespace,
    config: Mace_TrainerConfig,
    run_name: str,
    run_dir: Path,
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    seed: int,
    scheduler_t0: int,
    freeze: int | None = None,
    foundation_filter_elements: bool | None = None,
    checkpoint_epochs: set[int] | None = None,
    energy_offset: dict[int, float] | None = None,
) -> None:
    global _ACTIVE_CHECKPOINT_EPOCHS
    _ACTIVE_CHECKPOINT_EPOCHS = checkpoint_epochs
    if energy_offset is None:
        energy_offset = dict(ENERGY_OFFSET)
    hp = config.hyperparams
    args.name = run_name
    args.seed = seed
    args.train_file = str(train_file)
    args.valid_file = str(valid_file)
    args.test_file = str(test_file)
    args.valid_fraction = hp.valid_frac
    args.E0s = repr(energy_offset)
    args.energy_key = ENERGY_KEY
    args.forces_key = FORCE_KEY
    args.foundation_model = str(config.model.path)
    args.default_dtype = config.dtype
    args.device = config.device
    args.model = "MACE"
    args.r_max = hp.r_max
    args.num_channels = hp.num_channels
    args.num_cutoff_basis = hp.num_cutoff_basis
    args.max_L = hp.max_L
    args.max_ell = hp.max_ell
    args.num_interactions = hp.num_interactions
    args.correlation = hp.correlation
    args.num_radial_basis = hp.num_radial_basis
    args.loss = hp.loss
    args.forces_weight = hp.forces_weight
    args.energy_weight = hp.energy_weight
    args.valid_batch_size = hp.valid_batch_size
    args.batch_size = hp.batch_size
    args.max_num_epochs = hp.max_num_epochs
    args.patience = hp.patience
    args.eval_interval = hp.eval_interval
    args.lr = hp.lr
    args.optimizer = hp.optimizer
    args.amsgrad = hp.amsgrad
    args.weight_decay = hp.weight_decay
    args.clip_grad = hp.clip_grad
    args.swa = hp.swa
    args.ema = hp.ema
    args.ema_decay = hp.ema_decay
    args.pin_memory = hp.pin_memory
    args.wandb = False
    args.multiheads_finetuning = False
    args.freeze = freeze
    if foundation_filter_elements is not None:
        args.foundation_filter_elements = foundation_filter_elements
    args._uq_frozen_body = bool(freeze)
    args.work_dir = str(run_dir)
    args.checkpoints_dir = str(run_dir / "checkpoints")
    args.results_dir = str(run_dir / "results")
    args.model_dir = str(run_dir / "models")
    args.log_dir = str(run_dir / "logs")
    save_snapshot_checkpoints = scheduler_t0 < hp.max_num_epochs
    args.keep_checkpoints = save_snapshot_checkpoints
    args.save_all_checkpoints = save_snapshot_checkpoints
    args.scheduler = "CosineAnnealingWarmRestarts"
    args.cosine_T_0 = scheduler_t0
    args.cosine_T_mult = 1
    args.cosine_eta_min = 1e-6


def _run_training(
    options: argparse.Namespace,
    run_name: str,
    run_dir: Path,
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    seed: int,
    scheduler_t0: int,
    multihead_files: dict[str, Path] | None = None,
    head_files: dict[str, Path] | None = None,
    freeze: int | None = None,
    foundation_filter_elements: bool | None = None,
    epochs: int | None = None,
    energy_offset: dict[int, float] | None = None,
    warm_start_readout: bool = False,
    warm_start_heads: bool = False,
) -> Path:
    if multihead_files is not None and head_files is not None:
        raise ValueError("Pass either multihead_files (replay) or head_files (MHC)")

    if energy_offset is None:
        energy_offset = _isolated_atom_energies(
            _read_frames(train_file)
            + _read_frames(valid_file)
            + _read_frames(test_file)
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    frozen = freeze is not None
    force_weight = options.forces_weight
    if force_weight is None:
        force_weight = FROZEN_FORCES_WEIGHT if frozen else 1.0
    patience = options.patience
    if patience is None:
        patience = FROZEN_PATIENCE if frozen else None
    if frozen and patience is not None and patience >= (epochs or options.epochs):
        print(
            f"[uq_compare] warning: patience={patience} is not below the epoch "
            f"budget {epochs or options.epochs}; early stopping will not trigger"
        )
    print(
        f"[uq_compare] {run_name}: forces_weight={force_weight:g}, "
        f"energy_weight={options.energy_weight:g}, patience={patience}"
    )
    hyperparams = _make_hyperparams(
        options,
        train_file,
        valid_file,
        epochs=epochs,
        forces_weight=force_weight,
        energy_weight=options.energy_weight,
        patience=patience,
    )
    config = Mace_TrainerConfig(
        model_key=options.model_key,
        device=options.device,
        dtype=options.dtype,
        hyperparams=hyperparams,
        metadata=MaceTrainingMetadata(experiment_name=run_name, seed=seed),
    )
    config.metadata.config_path = str(run_dir / "mace_config.json")

    total_epochs = hyperparams.max_num_epochs
    checkpoint_epochs: set[int] | None = None
    if scheduler_t0 < total_epochs:
        # Warm-restart arm: keep only the cycle-end states that the snapshot
        # exporter will load.  Everything else is skipped at write time.
        cycles = total_epochs // scheduler_t0
        checkpoint_epochs = {
            min(index * scheduler_t0 - 1, total_epochs - 1)
            for index in range(1, cycles + 1)
        }

    parser = build_default_arg_parser()
    mace_args = parser.parse_args(["--name", run_name])
    _configure_run(
        mace_args,
        config,
        run_name,
        run_dir,
        train_file,
        valid_file,
        test_file,
        seed,
        scheduler_t0,
        freeze=freeze,
        foundation_filter_elements=foundation_filter_elements,
        checkpoint_epochs=checkpoint_epochs,
        energy_offset=energy_offset,
    )
    mace_args.num_workers = options.num_workers
    mace_args._uq_warm_start_readout = bool(warm_start_readout)
    mace_args._uq_warm_start_heads = bool(warm_start_heads)

    if multihead_files is not None:
        mace_args.multiheads_finetuning = True
        mace_args.pt_train_file = options.pt_train_file
        mace_args.pt_valid_file = None
        mace_args.num_samples_pt = options.pt_samples
        mace_args.force_mh_ft_lr = True
        mace_args.heads = repr(
            {
                head: {
                    "train_file": str(bootstrap),
                    "valid_file": str(valid_file),
                    "test_file": str(test_file),
                    "E0s": repr(energy_offset),
                    "valid_fraction": hyperparams.valid_frac,
                }
                for head, bootstrap in multihead_files.items()
            }
        )
    elif head_files is not None:
        # Frozen-body MHC: several randomly initialised readout heads on
        # disjoint data slices sharing the foundation descriptors.  No pt_head,
        # no replay data, and multiheads_finetuning stays False.
        mace_args.multiheads_finetuning = False
        mace_args.heads = repr(
            {
                head: {
                    "train_file": str(slice_file),
                    "valid_file": str(valid_file),
                    "test_file": str(test_file),
                    "E0s": repr(energy_offset),
                    "valid_fraction": hyperparams.valid_frac,
                }
                for head, slice_file in head_files.items()
            }
        )

    args_path = run_dir / "run_config.json"
    args_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "seed": seed,
                "scheduler_t0": scheduler_t0,
                "train_file": str(train_file),
                "valid_file": str(valid_file),
                "test_file": str(test_file),
                "multihead_heads": list(multihead_files or {}),
                "disjoint_heads": list(head_files or {}),
                "freeze": freeze,
                "foundation_filter_elements": foundation_filter_elements,
                "pt_train_file": options.pt_train_file
                if multihead_files is not None
                else None,
                "forces_weight": hyperparams.forces_weight,
                "energy_weight": hyperparams.energy_weight,
                "patience": hyperparams.patience,
                "warm_start_readout": bool(warm_start_readout),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nTraining {run_name} on {train_file}")
    old_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        run(mace_args)
    finally:
        os.chdir(old_cwd)
    config.write_config_train()

    model_path = run_dir / "models" / f"{run_name}.model"
    if not model_path.is_file():
        raise FileNotFoundError(f"MACE did not save the expected model: {model_path}")
    return model_path


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"_epoch-(\d+)(?:_swa)?\.pt$", path.name)
    return int(match.group(1)) if match else None


def _export_snapshot_models(
    run_dir: Path,
    run_name: str,
    final_model_path: Path,
    epochs: int,
    count: int,
) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = [
        (epoch, path)
        for path in checkpoint_dir.glob("*.pt")
        if (epoch := _checkpoint_epoch(path)) is not None
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    model = torch.load(final_model_path, map_location="cpu", weights_only=False)
    exported_dir = run_dir / "snapshot_models"
    exported_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    used_epochs: set[int] = set()

    for index in range(1, count + 1):
        # Collect at the END of each SGDR cycle (one epoch before the next
        # warm restart), where the LR is at its minimum and the member is
        # converged.  Even spacing across the run lands on the restart
        # boundaries and captures the loss-spiked, just-kicked model.
        cycle_length = epochs // count
        target_epoch = min(index * cycle_length - 1, epochs - 1)
        candidates = [item for item in checkpoints if item[0] not in used_epochs]
        epoch, checkpoint_path = min(
            candidates, key=lambda item: abs(item[0] - target_epoch)
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model_path = exported_dir / f"{run_name}_epoch_{epoch}.model"
        torch.save(model, model_path)
        used_epochs.add(epoch)
        exported.append(model_path)
        print(f"Exported snapshot at epoch {epoch}: {model_path}")
    return exported


def _cleanup_checkpoints(run_dir: Path) -> int:
    """Delete MACE checkpoint files after the exported models are written.

    Checkpoints store model *and* optimizer state and are overwritten/kept
    depending on MACE's flags, so a warm-restart run can leave hundreds of
    files. Only the exported ``.model`` files and the final model are needed
    downstream, so the checkpoint directory is removed.
    """
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return 0
    removed = 0
    for path in checkpoint_dir.glob("*.pt"):
        path.unlink()
        removed += 1
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    if removed:
        print(f"Removed {removed} checkpoint file(s): {checkpoint_dir}")
    return removed


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        output = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            output[order[start:end]] = (start + end - 1) / 2 + 1
            start = end
        return output

    corr = np.corrcoef(ranks(left), ranks(right))[0, 1]
    return float(corr) if np.isfinite(corr) else None


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    corr = np.corrcoef(left, right)[0, 1]
    return float(corr) if np.isfinite(corr) else None


def _relative_log_likelihood(
    errors: np.ndarray, sigmas: np.ndarray, rmse: float
) -> float | None:
    """Relative log-likelihood of Kellner et al., as used in the MHC paper.

    Positive values mean the per-point sigma beats a constant-RMSE predictor;
    1 is the oracle sigma = |error|.  A small floor keeps exact zeros finite.
    """
    errors = np.asarray(errors, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if errors.size == 0 or not np.isfinite(rmse) or rmse <= 0:
        return None
    sigma_squared = np.maximum(sigmas**2, 1e-24)
    nll = 0.5 * (errors**2 / sigma_squared + np.log(2 * np.pi * sigma_squared))
    nll_constant = 0.5 * (
        errors**2 / rmse**2 + np.log(2 * np.pi * rmse**2)
    )
    oracle_sigma = np.maximum(np.abs(errors), 1e-8)
    nll_oracle = 0.5 * (
        errors**2 / oracle_sigma**2 + np.log(2 * np.pi * oracle_sigma**2)
    )
    numerator = float(np.sum(nll - nll_constant))
    denominator = float(np.sum(nll_oracle - nll_constant))
    if denominator == 0 or not np.isfinite(denominator):
        return None
    return numerator / denominator


def _coverage(errors: np.ndarray, sigmas: np.ndarray, factor: float) -> float | None:
    errors = np.asarray(errors, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if errors.size == 0:
        return None
    return float(
        np.mean(np.abs(errors) <= factor * np.abs(sigmas) * (1.0 + 1e-9) + 1e-12)
    )


def _summarize_subset(
    indices: list[int],
    energy_errors: np.ndarray,
    force_abs_errors: list[np.ndarray],
    force_frame_rmse: np.ndarray,
    force_uncertainty: np.ndarray,
) -> dict[str, Any]:
    if not indices:
        return {"n_frames": 0}
    energy = energy_errors[indices]
    components = np.concatenate([force_abs_errors[index].ravel() for index in indices])
    uncertainties = force_uncertainty[indices]
    frame_errors = force_frame_rmse[indices]
    # Atom-weighted squared error and component count per frame, so at 100%
    # coverage the risk-coverage RMSE equals the overall force RMSE.  Averaging
    # per-frame RMSEs instead would up-weight frames with few atoms.
    squared_error_sums = np.asarray(
        [float(np.sum(force_abs_errors[index] ** 2)) for index in indices]
    )
    component_counts = np.asarray(
        [force_abs_errors[index].size for index in indices], dtype=float
    )
    order = np.argsort(uncertainties)
    risk = {}
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        selected_count = max(1, int(np.ceil(fraction * len(indices))))
        selected = order[:selected_count]
        risk[str(fraction)] = float(
            np.sqrt(
                squared_error_sums[selected].sum() / component_counts[selected].sum()
            )
        )
    return {
        "n_frames": len(indices),
        "energy_mae_mev_per_atom": float(np.mean(np.abs(energy)) * 1000),
        "energy_rmse_mev_per_atom": float(np.sqrt(np.mean(energy**2)) * 1000),
        "force_mae_ev_per_angstrom": float(np.mean(components)),
        "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(components**2))),
        "spearman_uncertainty_vs_force_error": _spearman_correlation(
            uncertainties, frame_errors
        ),
        "pearson_uncertainty_vs_force_error": _pearson_correlation(
            uncertainties, frame_errors
        ),
        "low_uncertainty_risk_coverage_force_rmse": risk,
    }


def _atom_level_correlation(
    metric: str,
    indices: list[int],
    atom_uncertainties: list[np.ndarray],
    atom_absolute_errors: list[np.ndarray],
) -> float | None:
    if not indices:
        return None
    uncertainty = np.concatenate([atom_uncertainties[index] for index in indices])
    error = np.concatenate([atom_absolute_errors[index] for index in indices])
    if metric == "spearman":
        return _spearman_correlation(uncertainty, error)
    if metric == "pearson":
        return _pearson_correlation(uncertainty, error)
    raise ValueError(f"Unknown correlation metric: {metric}")


def _evaluate_method(
    method: str,
    members: list[tuple[str, Path, str | None]],
    test_frames: list[Atoms],
    options: argparse.Namespace,
    output_dir: Path,
    calibration_frames: list[Atoms] | None = None,
) -> dict[str, Any]:
    member_energies: list[np.ndarray] = []
    member_forces: list[list[np.ndarray]] = []
    member_cal_energies: list[np.ndarray] = []
    member_cal_forces: list[list[np.ndarray]] = []

    calibration_frames = calibration_frames or []
    for member_name, model_path, head_name in members:
        calc_kwargs: dict[str, Any] = {
            "model_paths": [str(model_path)],
            "device": options.device,
            "default_dtype": options.dtype,
        }
        if head_name is not None:
            calc_kwargs["head"] = head_name
        calculator = MACECalculator(**calc_kwargs)
        energies: list[float] = []
        forces: list[np.ndarray] = []
        cal_energies: list[float] = []
        cal_forces: list[np.ndarray] = []
        for atoms in test_frames:
            prediction = atoms.copy()
            prediction.calc = calculator
            energies.append(float(prediction.get_potential_energy()))
            forces.append(np.asarray(prediction.get_forces(), dtype=float))
        for atoms in calibration_frames:
            prediction = atoms.copy()
            prediction.calc = calculator
            cal_energies.append(float(prediction.get_potential_energy()))
            cal_forces.append(np.asarray(prediction.get_forces(), dtype=float))
        member_energies.append(np.asarray(energies))
        member_forces.append(forces)
        if calibration_frames:
            member_cal_energies.append(np.asarray(cal_energies))
            member_cal_forces.append(cal_forces)
        del calculator
        if options.device == "cuda":
            torch.cuda.empty_cache()
        print(f"Evaluated {method} member {member_name}")

    energy_matrix = np.stack(member_energies, axis=0)
    energy_mean = np.mean(energy_matrix, axis=0)
    energy_std_per_atom = np.std(energy_matrix, axis=0) / np.asarray(
        [len(atoms) for atoms in test_frames]
    )
    n_frames = len(test_frames)
    force_uncertainty = np.zeros(n_frames)
    force_frame_rmse = np.zeros(n_frames)
    energy_errors = np.zeros(n_frames)
    force_abs_errors: list[np.ndarray] = []
    force_signed_errors: list[np.ndarray] = []
    force_component_sigmas: list[np.ndarray] = []
    atom_uncertainties: list[np.ndarray] = []
    atom_absolute_errors: list[np.ndarray] = []
    groups: list[str] = []
    csv_path = output_dir / f"{method}_predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "frame_index",
                "group",
                "n_atoms",
                "reference_energy_ev",
                "predicted_energy_mean_ev",
                "predicted_energy_std_ev_per_atom",
                "energy_abs_error_mev_per_atom",
                "force_rmse_ev_per_angstrom",
                "force_uncertainty_ev_per_angstrom",
            ),
        )
        writer.writeheader()
        for index, atoms in enumerate(test_frames):
            predicted_forces = np.stack(
                [member[index] for member in member_forces], axis=0
            )
            mean_forces = np.mean(predicted_forces, axis=0)
            component_std = np.std(predicted_forces, axis=0)
            force_uncertainty[index] = float(np.sqrt(np.mean(component_std**2)))
            ref_forces = np.asarray(atoms.arrays[FORCE_KEY], dtype=float)
            frame_force_error = mean_forces - ref_forces
            force_abs_errors.append(np.abs(frame_force_error))
            force_signed_errors.append(frame_force_error)
            force_component_sigmas.append(component_std)
            # Per-atom sigma as in the MHC paper (Eq. 4): mean over the three
            # force components of the per-component committee std.
            atom_uncertainties.append(np.mean(component_std, axis=1))
            atom_absolute_errors.append(np.mean(np.abs(frame_force_error), axis=1))
            force_frame_rmse[index] = float(np.sqrt(np.mean(frame_force_error**2)))
            ref_energy = float(atoms.info[ENERGY_KEY])
            energy_errors[index] = (energy_mean[index] - ref_energy) / len(atoms)
            group = str(atoms.info.get("config_type", "unknown"))
            groups.append(group)
            writer.writerow(
                {
                    "frame_index": index,
                    "group": group,
                    "n_atoms": len(atoms),
                    "reference_energy_ev": ref_energy,
                    "predicted_energy_mean_ev": energy_mean[index],
                    "predicted_energy_std_ev_per_atom": (energy_std_per_atom[index]),
                    "energy_abs_error_mev_per_atom": (abs(energy_errors[index]) * 1000),
                    "force_rmse_ev_per_angstrom": force_frame_rmse[index],
                    "force_uncertainty_ev_per_angstrom": (force_uncertainty[index]),
                }
            )

    all_indices = list(range(n_frames))
    group_indices = {
        group: [index for index, value in enumerate(groups) if value == group]
        for group in sorted(set(groups))
    }
    by_group = {
        group: _summarize_subset(
            indices,
            energy_errors,
            force_abs_errors,
            force_frame_rmse,
            force_uncertainty,
        )
        for group, indices in group_indices.items()
    }
    for group, indices in group_indices.items():
        by_group[group]["atom_level_spearman_uncertainty_vs_force_error"] = (
            _atom_level_correlation(
                "spearman", indices, atom_uncertainties, atom_absolute_errors
            )
        )
        by_group[group]["atom_level_pearson_uncertainty_vs_force_error"] = (
            _atom_level_correlation(
                "pearson", indices, atom_uncertainties, atom_absolute_errors
            )
        )

    calibration = None
    if calibration_frames and len(members) > 1:
        atom_counts = np.asarray([len(atoms) for atoms in calibration_frames])
        cal_energy_matrix = np.stack(member_cal_energies, axis=0)
        cal_energy_std = np.std(cal_energy_matrix, axis=0) / atom_counts
        cal_energy_error = (
            np.mean(cal_energy_matrix, axis=0)
            - np.asarray(
                [float(atoms.info[ENERGY_KEY]) for atoms in calibration_frames]
            )
        ) / atom_counts
        alpha_energy = float(
            np.sqrt(np.mean(cal_energy_error**2 / (cal_energy_std**2 + 1e-12)))
        )
        force_ratio_sum = 0.0
        component_count = 0
        for frame_index, atoms in enumerate(calibration_frames):
            stack = np.stack(
                [member[frame_index] for member in member_cal_forces], axis=0
            )
            sigma = np.std(stack, axis=0)
            error = np.mean(stack, axis=0) - np.asarray(
                atoms.arrays[FORCE_KEY], dtype=float
            )
            force_ratio_sum += float(np.sum(error**2 / (sigma**2 + 1e-12)))
            component_count += error.size
        calibration = {
            "n_frames": len(calibration_frames),
            "alpha_force": float(np.sqrt(force_ratio_sum / component_count)),
            "alpha_energy": alpha_energy,
            "definition": (
                "alpha^2 = mean((prediction - reference)^2 / disagreement^2) "
                "on the calibration (validation) frames"
            ),
        }
    member_metrics = []
    for member_index, (member_name, _, head_name) in enumerate(members):
        single_force_errors = [
            member_forces[member_index][index]
            - np.asarray(test_frames[index].arrays[FORCE_KEY], dtype=float)
            for index in all_indices
        ]
        single_energy_errors = np.asarray(
            [
                (member_energies[member_index][index] - float(atoms.info[ENERGY_KEY]))
                / len(atoms)
                for index, atoms in enumerate(test_frames)
            ]
        )
        member_metrics.append(
            {
                "member": member_name,
                "head": head_name,
                "energy_rmse_mev_per_atom": float(
                    np.sqrt(np.mean(single_energy_errors**2)) * 1000
                ),
                "force_rmse_ev_per_angstrom": float(
                    np.sqrt(
                        np.mean(
                            np.concatenate(
                                [error.ravel() for error in single_force_errors]
                            )
                            ** 2
                        )
                    )
                ),
            }
        )

    summary = {
        "method": method,
        "member_count": len(members),
        "prediction_file": str(csv_path),
        "overall": _summarize_subset(
            all_indices,
            energy_errors,
            force_abs_errors,
            force_frame_rmse,
            force_uncertainty,
        ),
        "by_group": by_group,
        "atom_level_spearman_uncertainty_vs_force_error": _atom_level_correlation(
            "spearman", all_indices, atom_uncertainties, atom_absolute_errors
        ),
        "atom_level_pearson_uncertainty_vs_force_error": _atom_level_correlation(
            "pearson", all_indices, atom_uncertainties, atom_absolute_errors
        ),
        "calibration": calibration,
        "member_accuracy": member_metrics,
    }
    if calibration is not None:
        alpha_force = float(calibration["alpha_force"])
        alpha_energy = float(calibration["alpha_energy"])

        def _add_calibration_metrics(
            subset: dict[str, Any], indices: list[int]
        ) -> None:
            signed_errors = np.concatenate(
                [force_signed_errors[index].ravel() for index in indices]
            )
            scaled_sigmas = np.concatenate(
                [force_component_sigmas[index].ravel() for index in indices]
            ) * alpha_force
            subset["rll_force"] = _relative_log_likelihood(
                signed_errors,
                scaled_sigmas,
                subset["force_rmse_ev_per_angstrom"],
            )
            subset["coverage_1sigma_force"] = _coverage(
                signed_errors, scaled_sigmas, 1.0
            )
            subset["coverage_2sigma_force"] = _coverage(
                signed_errors, scaled_sigmas, 2.0
            )
            energy_sigmas = energy_std_per_atom[indices] * alpha_energy
            subset["coverage_1sigma_energy"] = _coverage(
                energy_errors[indices], energy_sigmas, 1.0
            )
            subset["coverage_2sigma_energy"] = _coverage(
                energy_errors[indices], energy_sigmas, 2.0
            )

        _add_calibration_metrics(summary["overall"], all_indices)
        for group, indices in group_indices.items():
            _add_calibration_metrics(summary["by_group"][group], indices)
        calibration.update(
            {
                "rll_force": summary["overall"]["rll_force"],
                "coverage_1sigma_force": summary["overall"][
                    "coverage_1sigma_force"
                ],
                "coverage_2sigma_force": summary["overall"][
                    "coverage_2sigma_force"
                ],
                "coverage_1sigma_energy": summary["overall"][
                    "coverage_1sigma_energy"
                ],
                "coverage_2sigma_energy": summary["overall"][
                    "coverage_2sigma_energy"
                ],
            }
        )
    if len(members) < 2:
        # A single point estimate has no disagreement, so ranking metrics are
        # undefined rather than zero.
        summary["overall"]["low_uncertainty_risk_coverage_force_rmse"] = None
        for group_summary in summary["by_group"].values():
            group_summary["low_uncertainty_risk_coverage_force_rmse"] = None

    atom_level_path = output_dir / f"{method}_atom_level.npz"
    np.savez_compressed(
        atom_level_path,
        sigma=np.concatenate(atom_uncertainties),
        abs_error=np.concatenate(atom_absolute_errors),
        frame_index=np.concatenate(
            [
                np.full(len(values), index, dtype=np.int32)
                for index, values in enumerate(atom_uncertainties)
            ]
        ),
        group=np.concatenate(
            [
                np.full(len(values), groups[index])
                for index, values in enumerate(atom_uncertainties)
            ]
        ),
    )
    summary["atom_level_file"] = str(atom_level_path)
    return summary


def _load_prior_summaries(paths: list[Path]) -> dict[str, Any]:
    """Load method summaries produced by an earlier run of this script."""
    imported: dict[str, Any] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"--include-summary file not found: {resolved}")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Expected a non-empty method mapping in {resolved}")
        for method, summary in payload.items():
            if method in imported:
                print(f"Skipping duplicate imported method {method} from {resolved}")
                continue
            if not isinstance(summary, dict):
                raise ValueError(f"Malformed summary for {method} in {resolved}")
            entry = deepcopy(summary)
            entry["imported_from"] = str(resolved)
            imported[method] = entry
            print(f"Imported method {method} from {resolved}")
    return imported


def _remove_run_directory(run_root: Path, base_dir: Path) -> None:
    resolved_root = run_root.resolve()
    resolved_base = base_dir.resolve()
    if resolved_root == resolved_base or resolved_base not in resolved_root.parents:
        raise ValueError(
            f"Refusing to remove {resolved_root}: it is not a child of {resolved_base}"
        )
    print(f"Removing existing run directory: {resolved_root}")
    shutil.rmtree(resolved_root)


def _smoke_check_mhc(
    model_path: Path,
    head_names: list[str],
    test_frames: list[Atoms],
    options: argparse.Namespace,
) -> dict[str, Any]:
    """Fail fast if the frozen-body heads are missing, identical, or unreadable."""
    from itertools import combinations

    forces: dict[str, list[np.ndarray]] = {}
    available: list[str] = []
    for head in head_names:
        calculator = MACECalculator(
            model_paths=[str(model_path)],
            device=options.device,
            default_dtype=options.dtype,
            head=head,
        )
        available = list(getattr(calculator, "available_heads", []))
        if sorted(available) != sorted(head_names):
            raise RuntimeError(
                f"Model heads {available} do not match expected {head_names}"
            )
        per_frame: list[np.ndarray] = []
        for atoms in test_frames:
            prediction = atoms.copy()
            prediction.calc = calculator
            per_frame.append(np.asarray(prediction.get_forces(), dtype=float))
        forces[head] = per_frame
        del calculator
        if options.device == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(np.concatenate([f.ravel() for f in forces[head_names[0]]])).all():
        raise RuntimeError("Frozen-head predictions contain non-finite forces")

    spreads = {}
    for left, right in combinations(head_names, 2):
        per_frame_rms = [
            float(np.sqrt(np.mean((left_force - right_force) ** 2)))
            for left_force, right_force in zip(forces[left], forces[right])
        ]
        spreads[f"{left}|{right}"] = float(np.mean(per_frame_rms))
    min_spread = min(spreads.values())
    rms_force = float(
        np.sqrt(
            np.mean(
                [np.mean(frame**2) for frames in forces.values() for frame in frames]
            )
        )
    )
    if min_spread <= 1e-8:
        raise RuntimeError(
            "Frozen-body heads produce identical forces; check that "
            "foundation_filter_elements=False and the heads are independent"
        )
    result = {
        "heads": head_names,
        "n_frames": len(test_frames),
        "pairwise_force_rms_difference": spreads,
        "min_pairwise_force_rms_difference": min_spread,
        "relative_to_rms_force": min_spread / rms_force if rms_force else None,
    }
    relative = result["relative_to_rms_force"]
    relative_text = f"{relative:.3e}" if relative is not None else "n/a"
    print(
        "[uq_compare] smoke check passed: heads differ, min pairwise force RMS "
        f"difference {min_spread:.6f} eV/A (relative {relative_text})"
    )
    return result



def _learning_gate(
    run_dir: Path,
    run_name: str,
    min_drop: float = SMOKE_LEARNING_DROP,
    max_head_worsen: float = 0.10,
    max_final_rmse: float | None = None,
) -> dict[str, Any]:
    """Check that training actually improves validation force RMSE.

    The lif_uq_mhc_001 run passed every structural check and still trained the
    readouts to be worse than their initialisation: energy fell while force
    error rose.  A smoke run that only checks head diversity cannot catch that,
    so this gate reads the per-epoch validation log and aborts if the mean
    force error does not fall by ``min_drop`` or if any head gets materially
    worse.
    """

    matches = sorted((run_dir / "results").glob(f"{run_name}*_train.txt"))
    if not matches:
        raise RuntimeError(
            f"learning gate: no training log matching {run_name}*_train.txt "
            f"under {run_dir / 'results'}"
        )
    path = matches[0]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("{")
    ]
    evals = [
        record
        for record in records
        if record.get("mode") == "eval"
        and record.get("head") is not None
        and "rmse_f" in record
    ]
    if not evals:
        raise RuntimeError(f"learning gate: no eval records with rmse_f in {path}")

    by_head: dict[str, list[dict[str, Any]]] = {}
    for record in evals:
        by_head.setdefault(str(record["head"]), []).append(record)

    per_head: dict[str, dict[str, float]] = {}
    for head, head_records in by_head.items():
        head_records.sort(
            key=lambda item: item.get("epoch")
            if isinstance(item.get("epoch"), int)
            else -1
        )
        forces = [float(item["rmse_f"]) for item in head_records]
        first, last, best = forces[0], forces[-1], min(forces)
        per_head[head] = {
            "first_rmse_f": first,
            "last_rmse_f": last,
            "best_rmse_f": best,
            "relative_drop": (first - last) / first if first > 0 else 0.0,
            "last_over_best": last / best if best > 0 else float("inf"),
        }

    drops = [item["relative_drop"] for item in per_head.values()]
    regressions = [item["last_over_best"] for item in per_head.values()]
    mean_drop = float(np.mean(drops))
    worst = float(min(drops))
    worst_regression = float(max(regressions))
    start_rmse = float(np.mean([item["first_rmse_f"] for item in per_head.values()]))
    final_rmse = float(np.mean([item["last_rmse_f"] for item in per_head.values()]))
    # A warm-started probe can begin at foundation quality; demanding another
    # 10% there would fail a correct run.  Require the drop only when the arm
    # starts away from the zero-shot accuracy scale.
    required_drop = 0.0 if start_rmse <= SMOKE_ALREADY_GOOD_RMSE else min_drop
    passed = (
        mean_drop >= required_drop
        and worst >= -max_head_worsen
        and worst_regression <= 1.0 + SMOKE_MAX_REGRESSION_FROM_BEST
        and (max_final_rmse is None or final_rmse <= max_final_rmse)
    )
    result = {
        "run": run_name,
        "epochs": max(
            item.get("epoch")
            for item in evals
            if isinstance(item.get("epoch"), int)
        ),
        "mean_relative_drop": mean_drop,
        "worst_relative_drop": worst,
        "worst_last_over_best": worst_regression,
        "per_head": per_head,
        "start_force_rmse": start_rmse,
        "final_force_rmse": final_rmse,
        "max_final_rmse": max_final_rmse,
        "required_drop": required_drop,
        "min_required_drop": min_drop,
        "max_allowed_head_worsening": max_head_worsen,
        "max_allowed_regression_from_best": SMOKE_MAX_REGRESSION_FROM_BEST,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(
            f"learning gate failed for {run_name}: validation force RMSE did "
            f"not improve or drifted above its best value (mean drop "
            f"{mean_drop:.1%}, worst {worst:.1%}, worst last/best "
            f"{worst_regression:.2f}; required mean >= {required_drop:.0%}, worst "
            f">= {-max_head_worsen:.0%}, last/best <= "
            f"{1.0 + SMOKE_MAX_REGRESSION_FROM_BEST:.2f}, final <= "
            f"{max_final_rmse if max_final_rmse is not None else 'n/a'}). "
            f"Per head: "
            f"{per_head}. Check the force/energy loss weights and early "
            "stopping."
        )
    print(
        f"[uq_compare] learning gate passed for {run_name}: mean validation "
        f"force RMSE drop {mean_drop:.1%}, worst head {worst:.1%}"
    )
    return result


def main() -> None:
    options = parse_args()
    options.seeds = parse_seeds(options.seeds)
    try:
        arms = parse_arms(options.arms, smoke_test=options.smoke_test)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    options.run_id = options.run_id or (
        "lif_uq_mhc_smoke" if options.smoke_test else DEFAULT_RUN_ID
    )
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this node")
    print(f"UQ run '{options.run_id}': arms = {','.join(arms)}")
    if options.warm_start_readout and "frozen-probe" not in arms:
        raise SystemExit("--warm-start-readout requires the frozen-probe arm")
    if options.smoke_test:
        print(
            f"Smoke test: {options.smoke_epochs} epochs per frozen arm; checks "
            f"run before evaluation on {options.smoke_frames} test frames"
        )

    train_file = options.train_file.expanduser().resolve()
    valid_file = options.valid_file.expanduser().resolve()
    test_file = options.test_file.expanduser().resolve()
    _ensure_split_files([train_file, valid_file, test_file])
    train_frames = _read_frames(train_file)
    valid_frames = _read_frames(valid_file)
    test_frames = _read_frames(test_file)

    base_dir = options.output_dir.expanduser().resolve()
    run_root = base_dir / options.run_id
    if run_root.exists():
        if not options.overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_root}. "
                "Use --overwrite or a new --run-id."
            )
        _remove_run_directory(run_root, base_dir)
    run_root.mkdir(parents=True, exist_ok=False)

    include_summaries = list(options.include_summary)
    if not include_summaries:
        previous_summary = (
            base_dir / PREVIOUS_RUN_ID / "evaluation" / "uq_summary.json"
        )
        if previous_summary.is_file():
            include_summaries = [previous_summary]
            print(f"Auto-including previous results: {previous_summary}")
        else:
            print(
                "No previous uq_summary.json found; evaluating only this run's "
                "arms (no committee/snapshot comparison will be merged)"
            )

    mhc_epochs = options.smoke_epochs if options.smoke_test else options.mhc_epochs
    probe_epochs = (
        options.smoke_epochs if options.smoke_test else options.probe_epochs
    )
    eval_frames = (
        test_frames[: options.smoke_frames] if options.smoke_test else test_frames
    )
    calibration_frames = (
        valid_frames[: options.smoke_frames]
        if options.smoke_test
        else valid_frames
    )

    run_records: list[dict[str, Any]] = []
    methods: dict[str, list[tuple[str, Path, str | None]]] = {}

    if "committee" in arms:
        committee_members = []
        committee_bootstrap: dict[str, Path] = {}
        if options.bootstrap_committee:
            bootstrap_dir = run_root / "committee_bootstrap_data"
            committee_bootstrap = _make_bootstrap_files(
                train_frames,
                bootstrap_dir,
                options.seeds,
                options.bootstrap_block_size,
                names=[f"committee_seed_{seed}" for seed in options.seeds],
            )
        for seed in options.seeds:
            name = f"independent_seed_{seed}"
            run_dir = run_root / name
            member_train_file = (
                committee_bootstrap[f"committee_seed_{seed}"]
                if options.bootstrap_committee
                else train_file
            )
            started = time.perf_counter()
            model_path = _run_training(
                options,
                name,
                run_dir,
                member_train_file,
                valid_file,
                test_file,
                seed,
                options.epochs,
            )
            elapsed = time.perf_counter() - started
            _cleanup_checkpoints(run_dir)
            committee_members.append((name, model_path, None))
            run_records.append(
                {
                    "name": name,
                    "arm": "committee",
                    "seconds": elapsed,
                    "model": str(model_path),
                    "train_file": str(member_train_file),
                }
            )
        methods["independent_committee"] = committee_members

    if "snapshots" in arms:
        snapshot_name = "warm_snapshots"
        snapshot_dir = run_root / snapshot_name
        started = time.perf_counter()
        snapshot_final = _run_training(
            options,
            snapshot_name,
            snapshot_dir,
            train_file,
            valid_file,
            test_file,
            options.seeds[0],
            max(2, options.epochs // options.snapshot_count),
        )
        snapshot_paths = _export_snapshot_models(
            snapshot_dir,
            snapshot_name,
            snapshot_final,
            options.epochs,
            options.snapshot_count,
        )
        elapsed = time.perf_counter() - started
        _cleanup_checkpoints(snapshot_dir)
        snapshot_members = [(path.stem, path, None) for path in snapshot_paths]
        methods["warm_snapshots"] = snapshot_members
        run_records.append(
            {
                "name": snapshot_name,
                "arm": "snapshots",
                "seconds": elapsed,
                "model": str(snapshot_final),
                "snapshot_models": [str(path) for path in snapshot_paths],
            }
        )

    if "multihead-bootstrap" in arms and not options.skip_multihead:
        bootstrap_dir = run_root / "multihead_bootstrap_data"
        bootstrap_files = _make_bootstrap_files(
            train_frames,
            bootstrap_dir,
            options.seeds[:3],
            options.bootstrap_block_size,
        )
        multihead_name = "multihead_bootstrap"
        multihead_dir = run_root / multihead_name
        started = time.perf_counter()
        multihead_model = _run_training(
            options,
            multihead_name,
            multihead_dir,
            train_file,
            valid_file,
            test_file,
            options.seeds[0],
            options.epochs,
            multihead_files=bootstrap_files,
        )
        elapsed = time.perf_counter() - started
        multihead_model_object = torch.load(
            multihead_model, map_location="cpu", weights_only=False
        )
        multihead_export_dir = multihead_dir / "target_head_models"
        multihead_export_dir.mkdir(parents=True, exist_ok=True)
        multihead_members: list[tuple[str, Path, str | None]] = []
        for head_name in bootstrap_files:
            single_head = remove_pt_head(deepcopy(multihead_model_object), head_name)
            head_path = multihead_export_dir / f"{head_name}.model"
            torch.save(single_head, head_path)
            multihead_members.append((head_name, head_path, None))
        _cleanup_checkpoints(multihead_dir)
        methods["multihead_bootstrap"] = multihead_members
        run_records.append(
            {
                "name": multihead_name,
                "arm": "multihead-bootstrap",
                "seconds": elapsed,
                "model": str(multihead_model),
                "target_head_models": [str(member[1]) for member in multihead_members],
                "pt_train_file": options.pt_train_file,
                "pt_samples": options.pt_samples,
            }
        )

    smoke_check: dict[str, Any] = {}
    if "mhc" in arms:
        head_names = [
            MHC_HEAD_TEMPLATE.format(index=index)
            for index in range(options.mhc_heads)
        ]
        head_files, split_record = _make_disjoint_head_files(
            train_frames,
            run_root / "mhc_head_data",
            options.mhc_heads,
            options.split_seed,
            names=head_names,
        )
        mhc_name = "mhc_frozen_readouts"
        mhc_dir = run_root / mhc_name
        started = time.perf_counter()
        mhc_model = _run_training(
            options,
            mhc_name,
            mhc_dir,
            train_file,
            valid_file,
            test_file,
            options.seeds[0],
            mhc_epochs,
            head_files=head_files,
            freeze=FROZEN_BODY_LAYERS,
            foundation_filter_elements=options.warm_start_heads,
            epochs=mhc_epochs,
            warm_start_heads=options.warm_start_heads,
        )
        elapsed = time.perf_counter() - started
        _cleanup_checkpoints(mhc_dir)
        methods["frozen_mhc"] = [
            (head_name, mhc_model, head_name) for head_name in head_names
        ]
        run_records.append(
            {
                "name": mhc_name,
                "arm": "mhc",
                "seconds": elapsed,
                "model": str(mhc_model),
                "heads": head_names,
                "freeze": FROZEN_BODY_LAYERS,
                "foundation_filter_elements": False,
                "epochs": mhc_epochs,
                "head_split": split_record,
            }
        )
        if options.smoke_test:
            smoke_check["head_check"] = _smoke_check_mhc(
                mhc_model, head_names, eval_frames, options
            )
            smoke_check["learning_gate_mhc"] = _learning_gate(
                mhc_dir, mhc_name, max_final_rmse=0.10
            )

    if "frozen-probe" in arms:
        probe_name = "frozen_probe"
        probe_dir = run_root / probe_name
        started = time.perf_counter()
        probe_model = _run_training(
            options,
            probe_name,
            probe_dir,
            train_file,
            valid_file,
            test_file,
            options.seeds[0],
            probe_epochs,
            freeze=FROZEN_BODY_LAYERS,
            foundation_filter_elements=False,
            epochs=probe_epochs,
            warm_start_readout=options.warm_start_readout,
        )
        elapsed = time.perf_counter() - started
        if options.smoke_test:
            smoke_check["learning_gate_probe"] = _learning_gate(
                probe_dir, probe_name, max_final_rmse=0.05
            )
        _cleanup_checkpoints(probe_dir)
        methods["frozen_probe"] = [(probe_name, probe_model, None)]
        run_records.append(
            {
                "name": probe_name,
                "arm": "frozen-probe",
                "seconds": elapsed,
                "model": str(probe_model),
                "freeze": FROZEN_BODY_LAYERS,
                "foundation_filter_elements": False,
                "epochs": probe_epochs,
            }
        )

    if not options.skip_evaluation:
        foundation_config = Mace_TrainerConfig(
            model_key=options.model_key,
            device=options.device,
            dtype=options.dtype,
            hyperparams=_make_hyperparams(options, train_file, valid_file),
            metadata=MaceTrainingMetadata(
                experiment_name="uq_zero_shot_reference", seed=0
            ),
        )
        evaluation_dir = run_root / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        methods_to_evaluate: dict[str, list[tuple[str, Path, str | None]]] = {
            "zero_shot": [("foundation", foundation_config.model.path, None)],
        }
        methods_to_evaluate.update(methods)

        summaries = {}
        for method, members in methods_to_evaluate.items():
            if not members:
                continue
            summaries[method] = _evaluate_method(
                method,
                members,
                eval_frames,
                options,
                evaluation_dir,
                calibration_frames=calibration_frames,
            )
        for method, summary in _load_prior_summaries(
            include_summaries
        ).items():
            if method in summaries:
                print(f"Not importing {method}: it was evaluated in this run")
                continue
            summaries[method] = summary
        (evaluation_dir / "uq_summary.json").write_text(
            json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8"
        )
        if smoke_check:
            (evaluation_dir / "smoke_check.json").write_text(
                json.dumps(smoke_check, indent=2), encoding="utf-8"
            )

    manifest = {
        "run_id": options.run_id,
        "arms": list(arms),
        "smoke_test": options.smoke_test,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "test_file": str(test_file),
        "model_key": options.model_key,
        "device": options.device,
        "dtype": options.dtype,
        "seeds": options.seeds,
        "epochs": options.epochs,
        "mhc_epochs": mhc_epochs,
        "probe_epochs": probe_epochs,
        "mhc_heads": options.mhc_heads,
        "frozen_body_layers": FROZEN_BODY_LAYERS,
        "split_seed": options.split_seed,
        "snapshot_count": options.snapshot_count,
        "bootstrap_block_size": options.bootstrap_block_size,
        "committee_bootstrapped": options.bootstrap_committee,
        "multihead_enabled": "multihead-bootstrap" in arms
        and not options.skip_multihead,
        "include_summary": [str(path) for path in include_summaries],
        "runs": run_records,
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"All runs and results saved under: {run_root}")


if __name__ == "__main__":
    main()
