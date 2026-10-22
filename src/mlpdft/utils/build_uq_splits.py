#!/usr/bin/env python3
"""Build block-based train/valid/test extxyz splits for the UQ comparison.

Splits each group's trajectory into *contiguous time blocks* (never a random
neighboring-frame split). For every group, frames are assigned in trajectory
order:

    [ -------- train -------- | -- valid -- | -------------- test -------------- ]
       first (1 - test_frac) of the trajectory   tail of the remainder   last test_frac

The three splits are the concatenation of the per-group blocks, so every group
contributes to every split and `config_type` is preserved for per-group
reporting. A JSON manifest records the source file and frame range of each
group's contribution so every frame is traceable back to the DFT trajectory.

Labels are validated the same way `uq_compare.py` does: each frame must carry
`REF_energy` in ``atoms.info`` and `REF_forces` in ``atoms.arrays``.

Usage:
    uv run python src/mlpdft/utils/build_uq_splits.py --dry-run
    uv run python src/mlpdft/utils/build_uq_splits.py \
        --groups LIF64_KJPAW_FINAL LIWITHF_KJPAW_FINAL \
        --output-dir dataset/xyz_files/uq_splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from mlpdft.constants import DATA_DIR, ENERGY_KEY, FORCE_KEY, XYZ_DIR

# Final Li/F groups from assets/research/UQ_EXPERIMENT.md (primary scope:
# LiF / Li-rich / Li|LiF interface, NVT only, no boron chemistry). The NPT
# variants are excluded: the downstream diffusion validation runs in fixed-cell
# NVT, and NPT carries a density/barostat confound (non-uniform settings).
PROTOCOL_GROUPS = [
    "LIF64_KJPAW_FINAL",
    "LIWITHF_KJPAW_FINAL",
    "LIFINTERFACE_KJPAW_FINAL",
]

DEFAULT_STRIDE = 30


def pick_extxyz(group: str, stride: int | None) -> Path | None:
    """Return the extxyz file for a group, preferring the ``_<stride>_`` tag."""
    files = sorted((DATA_DIR / group / XYZ_DIR).glob("*.extxyz"))
    if not files:
        return None
    if stride is not None:
        tagged = [p for p in files if f"_{stride}_" in p.name]
        if tagged:
            return max(tagged, key=lambda p: p.stat().st_size)
    return max(files, key=lambda p: p.stat().st_size)


def _validate_labels(frames: list[Atoms], source: Path) -> None:
    for index, atoms in enumerate(frames):
        if ENERGY_KEY not in atoms.info or FORCE_KEY not in atoms.arrays:
            raise ValueError(
                f"Frame {index} of {source} lacks {ENERGY_KEY} or {FORCE_KEY}"
            )


def split_group(
    frames: list[Atoms], test_frac: float, valid_frac: float
) -> tuple[list[Atoms], list[Atoms], list[Atoms]]:
    """Contiguous train/valid/test split of one trajectory, in file order.

    ``valid_frac`` is the fraction of the *non-test* remainder used for
    validation, matching the protocol ("keep a separate validation block from
    the training portion").
    """
    n = len(frames)
    n_test = int(round(n * test_frac))
    n_rest = n - n_test
    n_valid = int(round(n_rest * valid_frac))
    test = frames[n - n_test :] if n_test else []
    rest = frames[:n_rest]
    valid = rest[n_rest - n_valid :] if n_valid else []
    train = rest[: n_rest - n_valid] if n_valid else rest
    return train, valid, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=PROTOCOL_GROUPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / XYZ_DIR / "uq_splits",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Prefer the per-group extxyz tagged with this stride (0 = largest file)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.4, help="Locked test fraction per group"
    )
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=0.2,
        help="Validation fraction of the non-test remainder per group",
    )
    parser.add_argument(
        "--run-id", default="uq", help="Manifest / filename prefix"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping groups with no extxyz file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the resulting split sizes without writing any files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.test_frac < 1.0:
        raise ValueError("--test-frac must be in [0, 1)")
    if not 0.0 <= args.valid_frac < 1.0:
        raise ValueError("--valid-frac must be in [0, 1)")

    stride = args.stride or None
    train_all: list[Atoms] = []
    valid_all: list[Atoms] = []
    test_all: list[Atoms] = []
    manifest: dict[str, object] = {
        "run_id": args.run_id,
        "test_frac": args.test_frac,
        "valid_frac": args.valid_frac,
        "stride_preference": stride,
        "groups": {},
    }
    missing: list[str] = []

    for group in args.groups:
        path = pick_extxyz(group, stride)
        if path is None:
            missing.append(group)
            if args.strict:
                raise FileNotFoundError(f"No extxyz file found for group {group}")
            print(f"[skip] {group}: no converted extxyz")
            continue

        frames = read(str(path), index=":")
        if isinstance(frames, Atoms):
            frames = [frames]
        frames = list(frames)
        _validate_labels(frames, path)

        train, valid, test = split_group(frames, args.test_frac, args.valid_frac)
        train_all.extend(train)
        valid_all.extend(valid)
        test_all.extend(test)
        n_rest = len(frames) - len(test)
        manifest["groups"][group] = {
            "source_file": str(path),
            "total_frames": len(frames),
            "train": {"count": len(train), "range": [0, len(train)]},
            "valid": {
                "count": len(valid),
                "range": [n_rest - len(valid), n_rest],
            },
            "test": {
                "count": len(test),
                "range": [len(frames) - len(test), len(frames)],
            },
        }
        print(
            f"[ok]   {group}: {len(frames)} frames "
            f"-> train {len(train)} / valid {len(valid)} / test {len(test)}  ({path.name})"
        )

    if missing:
        print(f"\n[warn] {len(missing)} group(s) had no extxyz: {', '.join(missing)}")
        print("       Convert them first with DataSet.convert_qe_out_to_extxyz().")

    if not train_all:
        raise SystemExit("No frames collected; nothing to write.")

    manifest["totals"] = {
        "train": len(train_all),
        "valid": len(valid_all),
        "test": len(test_all),
        "groups_included": len(manifest["groups"]),  # type: ignore[arg-type]
        "groups_missing": missing,
    }
    print(
        f"\nTotal: train {len(train_all)} / valid {len(valid_all)} / test {len(test_all)}"
    )

    if args.dry_run:
        print("[dry-run] no files written")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": args.output_dir / f"{args.run_id}_train.extxyz",
        "valid": args.output_dir / f"{args.run_id}_valid.extxyz",
        "test": args.output_dir / f"{args.run_id}_test.extxyz",
    }
    write(outputs["train"], train_all, format="extxyz")
    write(outputs["valid"], valid_all, format="extxyz")
    write(outputs["test"], test_all, format="extxyz")

    manifest["files"] = {name: str(p) for name, p in outputs.items()}
    manifest_path = args.output_dir / f"{args.run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
