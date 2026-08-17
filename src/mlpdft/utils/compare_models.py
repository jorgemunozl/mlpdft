#!/usr/bin/env python3
"""
Compare the parameters of two models that share the same architecture.

Loads the named tensors of both models and accumulates squared / absolute
errors to compute the overall MSE, RMSE, MAE and max-absolute-difference.

Both MACE epoch checkpoints (*_epoch-*.pt) and full .model files are
supported. Note: .model files embed e3nn TorchScript that requires an
NVIDIA driver to unpickle; epoch checkpoints are plain tensors and load
on any machine.

Usage:
    python compare_models.py [model_a] [model_b] [--verbose]

Defaults (both under outputs/):
    model_a = <outputs>/baseline_ft/checkpoints/baseline_ft_run-123_epoch-99.pt
    model_b = <outputs>/committee_s123/checkpoints/committee_s123_run-123_epoch-99.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TypedDict, cast

import torch

from mlpdft.constants import OUTPUTS_DIR


# DEFAULT_MODEL_A = ( OUTPUTS_DIR / "committee_s125" / "checkpoints" / "committee_s125_run-125_epoch-99.pt" )
DEFAULT_MODEL_A = ( OUTPUTS_DIR / "baseline_ft" / "checkpoints" / "baseline_ft_run-123_epoch-99.pt" )
# DEFAULT_MODEL_B = ( OUTPUTS_DIR / "committee_s123" / "checkpoints" / "committee_s123_run-123_epoch-99.pt" )
# DEFAULT_MODEL_B = ( OUTPUTS_DIR / "committee_s124" / "checkpoints" / "committee_s124_run-124_epoch-99.pt" )
DEFAULT_MODEL_B = ( OUTPUTS_DIR / "snapshot_warm" / "checkpoints" / "snapshot_warm_run-123_epoch-97.pt" )

class TensorStat(TypedDict):
    name: str
    numel: int
    mse: float
    max_abs: float


class CompareStats(TypedDict):
    numel: int
    mse: float
    rmse: float
    mae: float
    max_abs: float
    max_abs_name: str
    per_tensor: list[TensorStat]


def _as_parameter_dict(obj: object, model_path: Path) -> dict[str, torch.Tensor]:
    """Normalize a loaded object into a dict of named tensors."""
    if isinstance(obj, torch.nn.Module):
        return dict(obj.state_dict())
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        # MACE epoch checkpoint: {"model": state_dict, "optimizer": ..., ...}
        return cast(dict[str, torch.Tensor], obj["model"])
    if isinstance(obj, dict) and obj and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    ):
        return cast(dict[str, torch.Tensor], obj)
    raise ValueError(f"Unrecognized file format: {model_path}")


def load_parameters(model_path: Path) -> dict[str, torch.Tensor]:
    """Load the named tensors of a model or checkpoint file onto the CPU."""
    # Plain tensor pickles (epoch checkpoints) load without e3nn / CUDA.
    try:
        obj = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        return _as_parameter_dict(obj, model_path)
    except Exception:
        pass

    # Full module pickles (.model) require e3nn's TorchScript -> NVIDIA driver.
    try:
        obj = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{model_path}'. Full .model files embed e3nn TorchScript that requires an NVIDIA driver; use an epoch checkpoint (*_epoch-*.pt) instead."
        ) from exc
    return _as_parameter_dict(obj, model_path)


def compare_models(
    model_a: dict[str, torch.Tensor], model_b: dict[str, torch.Tensor]
) -> CompareStats:
    """Accumulate per-tensor differences between two same-shaped models."""
    if model_a.keys() != model_b.keys():
        missing = sorted(set(model_a) - set(model_b))
        extra = sorted(set(model_b) - set(model_a))
        raise ValueError(
            f"Tensor names do not match: missing in B: {missing}, extra in B: {extra}"
        )

    # Accumulators over all tensors
    total_sq = 0.0
    total_abs = 0.0
    total_numel = 0
    max_abs_diff = 0.0
    max_abs_name = ""
    per_tensor: list[TensorStat] = []

    for name in sorted(model_a):
        tensor_a = model_a[name].detach().float()
        tensor_b = model_b[name].detach().float()

        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}"
            )

        diff = tensor_a - tensor_b
        numel = diff.numel()
        if numel == 0:
            # Empty placeholder tensors contribute nothing to the error.
            continue
        sq = (diff * diff).sum().item()
        abs_ = diff.abs().sum().item()
        max_abs = diff.abs().max().item()

        total_sq += sq
        total_abs += abs_
        total_numel += numel
        if max_abs > max_abs_diff:
            max_abs_diff = max_abs
            max_abs_name = name

        per_tensor.append(
            {
                "name": name,
                "numel": numel,
                "mse": sq / numel,
                "max_abs": max_abs,
            }
        )

    mse = total_sq / total_numel
    return {
        "numel": total_numel,
        "mse": mse,
        "rmse": mse**0.5,
        "mae": total_abs / total_numel,
        "max_abs": max_abs_diff,
        "max_abs_name": max_abs_name,
        "per_tensor": per_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSE / RMSE between the parameters of two models."
    )
    parser.add_argument(
        "model_a",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_A,
        help=f"Path to the first model/checkpoint (default: {DEFAULT_MODEL_A})",
    )
    parser.add_argument(
        "model_b",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_B,
        help=f"Path to the second model/checkpoint (default: {DEFAULT_MODEL_B})",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a per-parameter breakdown"
    )
    args = parser.parse_args()

    if not args.model_a.exists():
        parser.error(f"model_a not found: {args.model_a}")
    if not args.model_b.exists():
        parser.error(f"model_b not found: {args.model_b}")

    model_a = load_parameters(args.model_a)
    model_b = load_parameters(args.model_b)

    print("=" * 70)
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print("=" * 70)

    try:
        stats = compare_models(model_a, model_b)
    except ValueError as exc:
        parser.error(f"Models are not comparable: {exc}")

    mse_log = math.log10(stats["mse"]) if stats["mse"] > 0 else float("-inf")
    print(f"  Tensors compared    : {stats['numel']:,}")
    print(f"  MSE                 : {stats['mse']:.6e}   (log10 = {mse_log:.2f})")
    print(f"  RMSE                : {stats['rmse']:.6e}")
    print(f"  MAE                 : {stats['mae']:.6e}")
    print(f"  Max |diff|          : {stats['max_abs']:.6e}  ({stats['max_abs_name']})")
    print("=" * 70)

    if args.verbose:
        print("\nPer-parameter breakdown (sorted by MSE):")
        print(
            f"{'parameter':<58} {'numel':>10} {'MSE':>13} {'log10 MSE':>10} {'max |diff|':>13}"
        )
        print("-" * 108)
        for entry in sorted(stats["per_tensor"], key=lambda e: e["mse"], reverse=True):
            log_mse = math.log10(entry["mse"]) if entry["mse"] > 0 else float("-inf")
            print(
                f"{entry['name']:<58} {entry['numel']:>10,} {entry['mse']:>13.6e} {log_mse:>10.2f} {entry['max_abs']:>13.6e}"
            )


if __name__ == "__main__":
    main()
