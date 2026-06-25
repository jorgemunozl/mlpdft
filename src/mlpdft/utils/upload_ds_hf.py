#!/usr/bin/env python3
"""Merge per-group .extxyz files into one dataset and upload to Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ase.io import read, write
from huggingface_hub import HfApi, create_repo

from mlpdft.config import Mace_TrainerConfig
from mlpdft.constants import (
    DATA_DIR,
    DATASET_NAME,
    FRAME_STRIDE,
    GROUPS_LIF,
    MAX_FRAMES,
    MERGED_FILENAME,
    PREFIX_HF,
    TEMPLATE_PATH,
    XYZ_DIR,
)


def merge_extxyz():
    all_frames: list = []
    for group, path in group_paths.items():
        frames = read(str(path), index=":")
        print(f"[read]     {group}: {len(frames)} frames  ←  {path.name}")
        all_frames.extend(frames)

    print(f"\nTotal frames across all groups: {len(all_frames)}")

    merged_path = DATA_DIR / XYZ_DIR / MERGED_FILENAME
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    for i, atoms in enumerate(all_frames):
        write(
            str(merged_path),
            atoms,
            format="extxyz",
            append=i > 0,
            write_results=False,
        )
    print(f"[write]    Merged extxyz → {merged_path}")

    # Build the group table rows
    group_rows = ""
    for idx, group in enumerate(GROUPS_LIF, 1):
        n = len(read(str(group_paths[group]), index=":"))
        group_rows += f"| {idx} | `{group}` | {n} |\n"

    # Read template and substitute placeholders
    template = TEMPLATE_PATH.read_text()
    readme = (
        template.replace("{{GROUP_TABLE}}", group_rows.rstrip("\n"))
        .replace("{{TOTAL_FRAMES}}", str(len(all_frames)))
        .replace("{{FRAME_STRIDE}}", str(FRAME_STRIDE))
        .replace("{{MERGED_FILENAME}}", MERGED_FILENAME)
    )

    # Write README
    readme_path = merged_path.with_name("README.md")
    readme_path.write_text(readme)
    print(f"[write]    Dataset card → {readme_path}")


def create_repo():
    repo_url = create_repo(
        repo_id=PREFIX_HF + "/" + DATASET_NAME,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"[hf]       Repo ready → {repo_url}")


def upload_files(name, path_in_repo: Path):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(path_in_repo),
        path_in_repo=name,
        repo_id=PREFIX_HF + "/" + DATASET_NAME,
        repo_type="dataset",
    )
    print(f"[hf]       Uploaded {name}")


def upload_qe_output():
    group_paths: List[Path] = []
    for group in GROUPS_LIF:
        print(f"[upload]   {group}")
        conf = Mace_TrainerConfig(
            group=group,
        )
        group_paths.append(conf.data_in_path)

    for file in group_paths:
        upload_files(file.name, file)


def main():
    upload_qe_output()


if __name__ == "__main__":
    main()
