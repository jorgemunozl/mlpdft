#!/usr/bin/env python3
"""Train and evaluate UQ ensembles for a fine-tuned MACE foundation model.

Requires curated, block-split train/validation/test extxyz files. Trains an
independent committee, a warm-snapshot run, and a multi-head bootstrap model,
then reports DFT test errors and uncertainty-ranking metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from mace.calculators.mace import MACECalculator
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser
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


DEFAULT_SEEDS = (123, 124, 125)
TARGET_HEADS = ("uq_head_123", "uq_head_124", "uq_head_125")

# Run defaults: the runner needs no flags. The block-split inputs live under
# ``dataset/xyz_files/uq_splits`` and are auto-downloaded from the HF dataset
# repo when missing. The run id is fixed so the runbook's paths are literal.
# Override with flags if needed.
DEFAULT_SPLIT_DIR = DATA_DIR / XYZ_DIR / "uq_splits"
DEFAULT_RUN_ID = "lif_uq_001"

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
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model-key", default="mace_omat_medium")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--epochs", type=int, default=150)
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
    parser.add_argument("--skip-multihead", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


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


def _make_hyperparams(args: argparse.Namespace, train_file: Path, valid_file: Path):
    values: dict[str, Any] = {
        "r_max": 6.0,
        "max_num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "valid_batch_size": args.valid_batch_size,
        "patience": args.epochs + 1,
        "eval_interval": 1,
        "valid_frac": 0.1,
        "loss": "universal",
        "forces_weight": 1.0,
        "energy_weight": 1.0,
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
) -> None:
    hp = config.hyperparams
    args.name = run_name
    args.seed = seed
    args.train_file = str(train_file)
    args.valid_file = str(valid_file)
    args.test_file = str(test_file)
    args.valid_fraction = hp.valid_frac
    args.E0s = repr(ENERGY_OFFSET)
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
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)
    hyperparams = _make_hyperparams(options, train_file, valid_file)
    config = Mace_TrainerConfig(
        model_key=options.model_key,
        device=options.device,
        dtype=options.dtype,
        hyperparams=hyperparams,
        metadata=MaceTrainingMetadata(experiment_name=run_name, seed=seed),
    )
    config.metadata.config_path = str(run_dir / "mace_config.json")

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
    )
    mace_args.num_workers = options.num_workers

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
                    "E0s": repr(ENERGY_OFFSET),
                    "valid_fraction": hyperparams.valid_frac,
                }
                for head, bootstrap in multihead_files.items()
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
                "pt_train_file": options.pt_train_file
                if multihead_files is not None
                else None,
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
    order = np.argsort(uncertainties)
    risk = {}
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        selected_count = max(1, int(np.ceil(fraction * len(indices))))
        selected = order[:selected_count]
        risk[str(fraction)] = float(np.sqrt(np.mean(frame_errors[selected] ** 2)))
    return {
        "n_frames": len(indices),
        "energy_mae_mev_per_atom": float(np.mean(np.abs(energy)) * 1000),
        "energy_rmse_mev_per_atom": float(np.sqrt(np.mean(energy**2)) * 1000),
        "force_mae_ev_per_angstrom": float(np.mean(components)),
        "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(components**2))),
        "spearman_uncertainty_vs_force_error": _spearman_correlation(
            uncertainties, frame_errors
        ),
        "low_uncertainty_risk_coverage_force_rmse": risk,
    }


def _evaluate_method(
    method: str,
    members: list[tuple[str, Path, str | None]],
    test_frames: list[Atoms],
    options: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    member_energies: list[np.ndarray] = []
    member_forces: list[list[np.ndarray]] = []

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
        for atoms in test_frames:
            prediction = atoms.copy()
            prediction.calc = calculator
            energies.append(float(prediction.get_potential_energy()))
            forces.append(np.asarray(prediction.get_forces(), dtype=float))
        member_energies.append(np.asarray(energies))
        member_forces.append(forces)
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
    by_group = {
        group: _summarize_subset(
            [index for index, value in enumerate(groups) if value == group],
            energy_errors,
            force_abs_errors,
            force_frame_rmse,
            force_uncertainty,
        )
        for group in sorted(set(groups))
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
        "member_accuracy": member_metrics,
    }
    return summary


def main() -> None:
    options = parse_args()
    options.seeds = parse_seeds(options.seeds)
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this node")

    train_file = options.train_file.expanduser().resolve()
    valid_file = options.valid_file.expanduser().resolve()
    test_file = options.test_file.expanduser().resolve()
    _ensure_split_files([train_file, valid_file, test_file])
    train_frames = _read_frames(train_file)
    _read_frames(valid_file)
    test_frames = _read_frames(test_file)

    run_root = options.output_dir.expanduser().resolve() / options.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    run_records: list[dict[str, Any]] = []
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
        committee_members.append((name, model_path, None))
        run_records.append(
            {
                "name": name,
                "seconds": elapsed,
                "model": str(model_path),
                "train_file": str(member_train_file),
            }
        )

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
    snapshot_members = [(path.stem, path, None) for path in snapshot_paths]
    run_records.append(
        {
            "name": snapshot_name,
            "seconds": elapsed,
            "model": str(snapshot_final),
            "snapshot_models": [str(path) for path in snapshot_paths],
        }
    )

    multihead_members: list[tuple[str, Path, str | None]] = []
    if not options.skip_multihead:
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
        for head_name in bootstrap_files:
            single_head = remove_pt_head(deepcopy(multihead_model_object), head_name)
            head_path = multihead_export_dir / f"{head_name}.model"
            torch.save(single_head, head_path)
            multihead_members.append((head_name, head_path, None))
        run_records.append(
            {
                "name": multihead_name,
                "seconds": elapsed,
                "model": str(multihead_model),
                "target_head_models": [str(member[1]) for member in multihead_members],
                "pt_train_file": options.pt_train_file,
                "pt_samples": options.pt_samples,
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
        methods = {
            "zero_shot": [("foundation", foundation_config.model.path, None)],
            "independent_committee": committee_members,
            "warm_snapshots": snapshot_members,
        }
        if multihead_members:
            methods["multihead_bootstrap"] = multihead_members

        summaries = {}
        for method, members in methods.items():
            summaries[method] = _evaluate_method(
                method, members, test_frames, options, evaluation_dir
            )
        (evaluation_dir / "uq_summary.json").write_text(
            json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8"
        )

    manifest = {
        "run_id": options.run_id,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "test_file": str(test_file),
        "model_key": options.model_key,
        "device": options.device,
        "dtype": options.dtype,
        "seeds": options.seeds,
        "epochs": options.epochs,
        "snapshot_count": options.snapshot_count,
        "bootstrap_block_size": options.bootstrap_block_size,
        "committee_bootstrapped": options.bootstrap_committee,
        "multihead_enabled": not options.skip_multihead,
        "runs": run_records,
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"All runs and results saved under: {run_root}")


if __name__ == "__main__":
    main()
