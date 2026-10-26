#!/usr/bin/env python3
"""Active-learning loop for the three-way UQ comparison.

Implements the pool-based label-replay protocol in
``artifacts/methodology-review/active-learning-protocol.md``:

  * pool  = the training split (labels stored on disk, "revealed" on acquisition)
  * seed  = balanced, evenly-spaced start (``--seed-per-group`` frames per group)
  * loop  = train ensemble -> score unlabeled pool by acquisition uncertainty
            -> diversity guard -> reveal top-b -> evaluate the locked test set
  * stop  = pool exhausted, or test force RMSE plateau (``--min-delta`` over
            ``--patience-iterations``)

Every "acquisition" reveals an already-stored ``REF_forces`` / ``REF_energy``
label; no new DFT is generated and there is no human in the loop.

The fine-tune core is imported from :mod:`mlpdft.uq_compare`, so each AL
iteration uses the exact same training recipe (epochs, scheduler, seeds,
bootstrap) as the calibration run. Costs are measured per iteration, enabling
labels-to-target (LTT) and compute-to-target (CTT) against a random baseline.

Usage:
    uv run python src/mlpdft/uq_active_learning.py \
        --strategies committee multihead snapshot random \
        --ceiling-rmse 0.0150
"""

from __future__ import annotations

import argparse
import json
import os
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
from ase.io import write
from mace.calculators.mace import MACECalculator
from mace.tools.scripts_utils import remove_pt_head

from mlpdft.constants import DATA_DIR, FORCE_KEY, OUTPUTS_DIR, XYZ_DIR
from mlpdft.uq_compare import (
    DEFAULT_SEEDS,
    _ensure_split_files,
    _export_snapshot_models,
    _make_bootstrap_files,
    _read_frames,
    _run_training,
)

DEFAULT_SPLIT_DIR = DATA_DIR / XYZ_DIR / "uq_splits"
DEFAULT_RUN_ID = "lif_al_001"

# Acquisition score regularizer (matches the paper's relative-force-error form).
ACQ_EPS = 0.2  # eV / Å


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Active-learning loop over the Li-F UQ split, comparing committee / "
            "multi-head / snapshot uncertainty against a random baseline."
        )
    )
    # --- data (shared with uq_compare.py) ---
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
        "--output-dir", type=Path, default=OUTPUTS_DIR / "uq_active_learning"
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    # --- training (must match uq_compare.py defaults) ---
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
    # --- active-learning loop ---
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["committee", "multihead", "snapshot", "random"],
        choices=("committee", "multihead", "snapshot", "random"),
        help="Acquisition strategies to run. 'random' uses committee training.",
    )
    parser.add_argument(
        "--seed-per-group",
        type=int,
        default=10,
        help="Initial labeled frames per group (evenly-spaced in trajectory order)",
    )
    parser.add_argument(
        "--acquire-batch",
        type=int,
        default=10,
        help="Frames acquired (revealed) per iteration",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="Diversity guard: minimum within-group pool distance between acquired frames",
    )
    parser.add_argument(
        "--ceiling-rmse",
        type=float,
        default=None,
        help="Full-data (run-2) test force RMSE; target = 1.05 * this value",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Relative RMSE change below which the loop is considered plateaued",
    )
    parser.add_argument(
        "--patience-iterations",
        type=int,
        default=3,
        help="Consecutive iterations to compare for the plateau rule",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override the acquisition count (default: exhaust the pool)",
    )
    parser.add_argument("--random-seed", type=int, default=12345)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Provide exactly three unique committee seeds")
    return seeds


def _select_seed(pool_frames: list, per_group: int) -> list[int]:
    """Return evenly-spaced, per-group-balanced seed indices into ``pool_frames``."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, atoms in enumerate(pool_frames):
        by_group[str(atoms.info.get("config_type", "all"))].append(index)

    seed: list[int] = []
    for indices in by_group.values():
        n = len(indices)
        k = min(per_group, n)
        if k == 1:
            picks = [indices[0]]
        else:
            positions = np.linspace(0, n - 1, k).round().astype(int)
            picks = [indices[int(p)] for p in positions]
        seed.extend(picks)
    return sorted(seed)


def _group_position(pool_frames: list) -> tuple[list[str], dict[int, int]]:
    """Return each frame's config_type and its trajectory-order position within it."""
    group_of: list[str] = []
    position_of: dict[int, int] = {}
    counter: dict[str, int] = defaultdict(int)
    for index, atoms in enumerate(pool_frames):
        group = str(atoms.info.get("config_type", "all"))
        group_of.append(group)
        position_of[index] = counter[group]
        counter[group] += 1
    return group_of, position_of


def _predict(
    members: list[tuple[str, Path, str | None]],
    frames: list,
    options: argparse.Namespace,
) -> tuple[list[list], list[list]]:
    """Evaluate ensemble members over ``frames``; return per-member forces and energies."""
    member_forces: list[list] = []
    member_energies: list[list] = []
    for _, model_path, head_name in members:
        calc_kwargs: dict[str, Any] = {
            "model_paths": [str(model_path)],
            "device": options.device,
            "default_dtype": options.dtype,
        }
        if head_name is not None:
            calc_kwargs["head"] = head_name
        calculator = MACECalculator(**calc_kwargs)
        forces: list[np.ndarray] = []
        energies: list[float] = []
        for atoms in frames:
            prediction = atoms.copy()
            prediction.calc = calculator
            forces.append(np.asarray(prediction.get_forces(), dtype=float))
            energies.append(float(prediction.get_potential_energy()))
        member_forces.append(forces)
        member_energies.append(energies)
        del calculator
        if options.device == "cuda":
            torch.cuda.empty_cache()
    return member_forces, member_energies


def _acquisition_scores(member_forces: list[list]) -> np.ndarray:
    """Per-frame acquisition score: max over atoms of sigma_i / (|F_bar_i| + eps)."""
    n_frames = len(member_forces[0])
    scores = np.empty(n_frames)
    for index in range(n_frames):
        stacked = np.stack([member[index] for member in member_forces], axis=0)  # [M, A, 3]
        mean_forces = np.mean(stacked, axis=0)  # [A, 3]
        component_std = np.std(stacked, axis=0)  # [A, 3]
        sigma_i = np.sqrt(np.mean(component_std**2, axis=1))  # [A]
        force_mag = np.linalg.norm(mean_forces, axis=1)  # [A]
        scores[index] = float(np.max(sigma_i / (force_mag + ACQ_EPS)))
    return scores


def _test_force_rmse(member_forces: list[list], test_frames: list) -> float:
    """Atom-weighted ensemble-mean force RMSE over the locked test set."""
    errors: list[np.ndarray] = []
    for index, atoms in enumerate(test_frames):
        stacked = np.stack([member[index] for member in member_forces], axis=0)
        mean_forces = np.mean(stacked, axis=0)
        ref_forces = np.asarray(atoms.arrays[FORCE_KEY], dtype=float)
        errors.append(mean_forces - ref_forces)
    components = np.concatenate([error.ravel() for error in errors])
    return float(np.sqrt(np.mean(components**2)))


def _select_with_diversity(
    scores: np.ndarray,
    unlabeled: list[int],
    group_of: list[str],
    position_of: dict[int, int],
    gap: int,
    batch: int,
) -> list[int]:
    """Greedily take the highest-scoring frames, skipping within-group neighbours."""
    order = sorted(range(len(unlabeled)), key=lambda j: -scores[j])
    selected: list[int] = []
    for j in order:
        index = unlabeled[j]
        if any(
            group_of[index] == group_of[other]
            and abs(position_of[index] - position_of[other]) < gap
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= batch:
            break
    return selected


def _train_ensemble(
    strategy: str,
    labeled_frames: list,
    valid_file: Path,
    test_file: Path,
    options: argparse.Namespace,
    iteration_dir: Path,
) -> tuple[list[tuple[str, Path, str | None]], float]:
    """Train one ensemble on ``labeled_frames`` and return (members, elapsed_seconds)."""
    iteration_dir.mkdir(parents=True, exist_ok=True)
    train_path = iteration_dir / "labeled_train.extxyz"
    write(train_path, labeled_frames, format="extxyz")
    started = time.perf_counter()

    if strategy == "committee":
        bootstrap = (
            _make_bootstrap_files(
                labeled_frames,
                iteration_dir / "bootstrap_data",
                options.seeds,
                options.bootstrap_block_size,
                names=[f"committee_seed_{seed}" for seed in options.seeds],
            )
            if options.bootstrap_committee
            else None
        )
        members: list[tuple[str, Path, str | None]] = []
        for seed in options.seeds:
            name = f"independent_seed_{seed}"
            member_train = (
                bootstrap[f"committee_seed_{seed}"] if bootstrap is not None else train_path
            )
            model_path = _run_training(
                options,
                name,
                iteration_dir / name,
                member_train,
                valid_file,
                test_file,
                seed,
                options.epochs,
            )
            members.append((name, model_path, None))
    elif strategy == "snapshot":
        name = "warm_snapshots"
        run_dir = iteration_dir / name
        final_model = _run_training(
            options,
            name,
            run_dir,
            train_path,
            valid_file,
            test_file,
            options.seeds[0],
            max(2, options.epochs // options.snapshot_count),
        )
        snapshots = _export_snapshot_models(
            run_dir, name, final_model, options.epochs, options.snapshot_count
        )
        members = [(path.stem, path, None) for path in snapshots]
    elif strategy == "multihead":
        bootstrap = _make_bootstrap_files(
            labeled_frames,
            iteration_dir / "bootstrap_data",
            options.seeds[:3],
            options.bootstrap_block_size,
        )
        name = "multihead_bootstrap"
        run_dir = iteration_dir / name
        model = _run_training(
            options,
            name,
            run_dir,
            train_path,
            valid_file,
            test_file,
            options.seeds[0],
            options.epochs,
            multihead_files=bootstrap,
        )
        model_object = torch.load(model, map_location="cpu", weights_only=False)
        export_dir = run_dir / "target_head_models"
        export_dir.mkdir(parents=True, exist_ok=True)
        members = []
        for head_name in bootstrap:
            single_head = remove_pt_head(deepcopy(model_object), head_name)
            head_path = export_dir / f"{head_name}.model"
            torch.save(single_head, head_path)
            members.append((head_name, head_path, None))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return members, time.perf_counter() - started


def _run_strategy(
    strategy: str,
    pool_frames: list,
    valid_file: Path,
    test_frames: list,
    seed_indices: list[int],
    options: argparse.Namespace,
    strategy_dir: Path,
) -> dict[str, Any]:
    """Run one full acquisition loop and return its data-efficiency records."""
    labeled = list(seed_indices)
    labeled_set = set(labeled)
    group_of, position_of = _group_position(pool_frames)
    rng = np.random.default_rng(options.random_seed)

    train_strategy = "committee" if strategy == "random" else strategy
    n_pool = len(pool_frames)

    records: list[dict[str, Any]] = []
    cumulative_seconds = 0.0
    iteration = 0

    while True:
        labeled_frames = [pool_frames[index] for index in labeled]
        iteration_dir = strategy_dir / f"iter_{iteration:02d}"
        members, train_seconds = _train_ensemble(
            train_strategy, labeled_frames, valid_file, test_file, options, iteration_dir
        )
        cumulative_seconds += train_seconds

        member_forces, _ = _predict(members, test_frames, options)
        rmse = _test_force_rmse(member_forces, test_frames)

        records.append(
            {
                "iteration": iteration,
                "labeled_count": len(labeled),
                "labeled_fraction": len(labeled) / n_pool,
                "test_force_rmse_ev_per_angstrom": rmse,
                "train_seconds": train_seconds,
                "cumulative_seconds": cumulative_seconds,
                "labeled_group_counts": {
                    group: sum(1 for index in labeled if group_of[index] == group)
                    for group in sorted(set(group_of))
                },
            }
        )
        print(
            f"[{strategy:>10}] iter {iteration:02d}  "
            f"labeled {len(labeled):3d}/{n_pool}  "
            f"test force RMSE {rmse:.4f} eV/Å  "
            f"cost {cumulative_seconds:8.1f} s"
        )

        unlabeled = [index for index in range(n_pool) if index not in labeled_set]
        if not unlabeled:
            break
        if options.max_iterations is not None and iteration >= options.max_iterations:
            break
        if len(records) > options.patience_iterations:
            prev = records[-1 - options.patience_iterations]["test_force_rmse_ev_per_angstrom"]
            if prev > 0 and abs(rmse - prev) / prev < options.min_delta:
                print(f"[{strategy:>10}] plateau reached (ΔRMSE < {options.min_delta:.0%}); stopping")
                break

        batch_size = min(options.acquire_batch, len(unlabeled))
        if strategy == "random":
            batch = rng.choice(unlabeled, size=batch_size, replace=False).tolist()
        else:
            pool_subset = [pool_frames[index] for index in unlabeled]
            subset_forces, _ = _predict(members, pool_subset, options)
            scores = _acquisition_scores(subset_forces)
            batch = _select_with_diversity(
                scores, unlabeled, group_of, position_of, options.gap, batch_size
            )

        labeled.extend(batch)
        labeled_set.update(batch)
        iteration += 1

    return {
        "strategy": strategy,
        "training_strategy": train_strategy,
        "records": records,
        "seed_indices": list(seed_indices),
    }


def main() -> None:
    options = parse_args()
    options.seeds = parse_seeds(options.seeds)
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this node")
    if options.ceiling_rmse is not None:
        options.target_rmse = 1.05 * options.ceiling_rmse
    else:
        options.target_rmse = None

    train_file = options.train_file.expanduser().resolve()
    valid_file = options.valid_file.expanduser().resolve()
    test_file = options.test_file.expanduser().resolve()
    _ensure_split_files([train_file, valid_file, test_file])
    pool_frames = _read_frames(train_file)
    _read_frames(valid_file)
    test_frames = _read_frames(test_file)

    seed_indices = _select_seed(pool_frames, options.seed_per_group)
    print(
        f"Pool {len(pool_frames)} frames; seed {len(seed_indices)} "
        f"({options.seed_per_group}/group); batch {options.acquire_batch}"
    )

    run_root = options.output_dir.expanduser().resolve() / options.run_id
    run_root.mkdir(parents=True, exist_ok=False)

    results: dict[str, Any] = {}
    for strategy in options.strategies:
        strategy_dir = run_root / strategy
        results[strategy] = _run_strategy(
            strategy,
            pool_frames,
            valid_file,
            test_frames,
            seed_indices,
            options,
            strategy_dir,
        )
        if options.target_rmse is not None:
            ltt = ctt = None
            for record in results[strategy]["records"]:
                if record["test_force_rmse_ev_per_angstrom"] <= options.target_rmse:
                    ltt = record["labeled_count"]
                    ctt = record["cumulative_seconds"]
                    break
            results[strategy]["labels_to_target"] = ltt
            results[strategy]["compute_to_target"] = ctt

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
        "strategies": options.strategies,
        "seed_per_group": options.seed_per_group,
        "seed_indices": seed_indices,
        "acquire_batch": options.acquire_batch,
        "gap": options.gap,
        "ceiling_rmse": options.ceiling_rmse,
        "target_rmse": options.target_rmse,
        "min_delta": options.min_delta,
        "patience_iterations": options.patience_iterations,
        "random_seed": options.random_seed,
        "results": results,
    }
    (run_root / "al_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\nActive-learning run saved under: {run_root}")


if __name__ == "__main__":
    main()
