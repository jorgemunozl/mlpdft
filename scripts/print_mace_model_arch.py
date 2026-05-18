#!/usr/bin/env python3
"""
Print architecture hyperparameters from a MACE .model checkpoint.

Uses MACE's extract_load + extract_config_mace_model (same as training introspection).

Default search: --model-path if set; else first match under ~/.cache/mace for
the MACE-MP-0a small naming pattern (128-L0 / mace-128-L0).

Example:
  uv run python scripts/print_mace_model_arch.py
  uv run python scripts/print_mace_model_arch.py --model-path ~/.cache/mace/foo.model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_KEYS_SLIDE = (
    "r_max",
    "num_bessel",
    "num_polynomial_cutoff",
    "max_ell",
    "num_interactions",
    "correlation",
    "hidden_irreps",
    "edge_irreps",
    "MLP_irreps",
    "num_elements",
    "radial_type",
    "distance_transform",
    "pair_repulsion",
    "apply_cutoff",
    "radial_MLP",
    "heads",
    "use_reduced_cg",
    "use_so3",
    "avg_num_neighbors",
)


def _fmt(val: Any) -> str:
    if val is None:
        return "None"
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    if hasattr(val, "item") and callable(getattr(val, "item")):
        try:
            x = val.item()
            if isinstance(x, float) and x == int(x):
                return str(int(x))
            return str(x)
        except Exception:
            pass
    if hasattr(val, "tolist"):
        return str(val.tolist())
    if hasattr(val, "__class__") and val.__class__.__name__ == "type":
        return val.__name__
    if callable(val):
        return getattr(val, "__name__", repr(val))
    return str(val)


def _default_cache_model() -> Path | None:
    cache = Path.home() / ".cache" / "mace"
    if not cache.is_dir():
        return None
    patterns = (
        "*128*L0*.model",
        "*128*L0*",  # mace cache uses extensionless names, e.g. ...epoch249model
        "*mace*128*.model",
        "*.model",
    )
    for pat in patterns:
        matches = sorted(cache.glob(pat))
        if matches:
            return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to .model checkpoint (default: first candidate under ~/.cache/mace)",
    )
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Print full extract_config_mace_model dict (noisy)",
    )
    args = parser.parse_args()

    path = args.model_path
    if path is None:
        path = _default_cache_model()
    if path is None or not path.is_file():
        print(
            "Error: no checkpoint found. Download MACE-MP (e.g. mace_mp small) or pass --model-path.",
            file=sys.stderr,
        )
        return 1

    import torch

    from mace.tools.scripts_utils import extract_config_mace_model

    # Released checkpoints are often raw ScaleShiftMACE modules. extract_load() rebuilds
    # from config + state_dict and can fail across mace-torch versions; loading raw works.
    obj = torch.load(str(path.resolve()), map_location="cpu", weights_only=False)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    cfg = extract_config_mace_model(model)
    if "error" in cfg:
        print(cfg["error"], file=sys.stderr)
        return 1

    print(f"# {path}")
    print(f"# class: {model.__class__.__name__}\n")

    if args.all_keys:
        for k in sorted(cfg.keys()):
            print(f"{k}: {_fmt(cfg[k])}")
        return 0

    for k in _KEYS_SLIDE:
        if k in cfg:
            print(f"{k}: {_fmt(cfg[k])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
