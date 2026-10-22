#!/usr/bin/env python3
"""Upload the UQ train/valid/test splits to a private Hugging Face dataset repo.

Uploads the three block-split extxyz files, the split manifest, the per-group
source extxyz files (for provenance), and a generated dataset card.

Usage:
    uv run python src/mlpdft/utils/upload_uq_splits_hf.py
    uv run python src/mlpdft/utils/upload_uq_splits_hf.py --repo-id user/name
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from mlpdft.constants import DATA_DIR, PREFIX_HF, XYZ_DIR

DEFAULT_REPO_ID = f"{PREFIX_HF}/li_f_uq_splits_v2"
SPLIT_DIR = DATA_DIR / XYZ_DIR / "uq_splits"


def build_readme(manifest: dict, repo_id: str) -> str:
    groups = manifest["groups"]
    totals = manifest["totals"]
    rows = "\n".join(
        f"| `{group}` | {info['total_frames']} | {info['train']['count']} | "
        f"{info['valid']['count']} | {info['test']['count']} |"
        for group, info in groups.items()
    )
    return f"""---
license: mit
tags:
  - mace
  - li-f
  - dft
  - qe
  - uncertainty-quantification
---

# Li–F UQ splits ({manifest["run_id"]})

Block-split Li–F DFT data for uncertainty-quantification benchmarking of
fine-tuned MACE foundation models (`uq_compare.py`).

## Composition

| Group | Frames | Train | Valid | Test |
|---|---:|---:|---:|---:|
{rows}

**Totals:** train {totals["train"]} / valid {totals["valid"]} / test {totals["test"]}

## Generation parameters

- Equilibration cutoff: `100` frames discarded
- Frame stride: `30` (all groups, to decorrelate adjacent AIMD frames)
- Cap: `200` frames per group
- Energy key: `REF_energy`, Force key: `REF_forces`
- `config_type` = source group name

## Split method

Per group, contiguous time blocks in trajectory order:
train (first) → valid (middle) → test (last {int(manifest["test_frac"] * 100)}%).
No random neighboring-frame split. See `uq_manifest.json` for each group's
source file and frame ranges.

## Files

- `uq_train.extxyz`, `uq_valid.extxyz`, `uq_test.extxyz`
- `uq_manifest.json`
- `group_files/` — per-group source extxyz

## Usage

```bash
huggingface-cli download {repo_id} --repo-type dataset --local-dir uq_splits

uv run python src/mlpdft/uq_compare.py \\
  --train-file uq_splits/uq_train.extxyz \\
  --valid-file uq_splits/uq_valid.extxyz \\
  --test-file  uq_splits/uq_test.extxyz \\
  --run-id lif_uq_001
```
"""


def upload_file(api: HfApi, repo_id: str, local_path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    size_mb = local_path.stat().st_size / 1e6
    print(f"[hf] uploaded {path_in_repo:32s} ({size_mb:.2f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repo public (default: private, matching the other Li–F datasets)",
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.split_dir / "uq_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing {manifest_path}. Run build_uq_splits.py before uploading."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_files = [
        args.split_dir / "uq_train.extxyz",
        args.split_dir / "uq_valid.extxyz",
        args.split_dir / "uq_test.extxyz",
        manifest_path,
    ]
    for path in split_files:
        if not path.is_file():
            raise SystemExit(f"Missing split file: {path}")

    api = HfApi()
    print(f"[hf] ensuring repo {args.repo_id} (private={not args.public})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    for path in split_files:
        upload_file(api, args.repo_id, path, path.name)

    # Per-group source files, for provenance/traceability.
    for group, info in manifest["groups"].items():
        source = Path(info["source_file"])
        if source.is_file():
            upload_file(api, args.repo_id, source, f"group_files/{source.name}")
        else:
            print(f"[warn]           group source missing: {source}")

    readme = build_readme(manifest, args.repo_id)
    readme_local = args.split_dir / "README.md"
    readme_local.write_text(readme, encoding="utf-8")
    upload_file(api, args.repo_id, readme_local, "README.md")

    print(f"\nDone. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
