#!/usr/bin/env python3
"""Merge per-group .extxyz files into one dataset and upload to Hugging Face."""

from __future__ import annotations

import re
from pathlib import Path

from ase.io import read, write
from huggingface_hub import HfApi, create_repo

from mlpdft.constants import (
    DATA_DIR,
    DATASET_NAME_2,
    FRAME_STRIDE,
    GROUPS_LIF,
    MERGED_FILENAME_DS_2,
    PREFIX_HF,
    TEMPLATE_PATH,
    XYZ_DIR,
)

REPO_ID = f"{PREFIX_HF}/{DATASET_NAME_2}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_extxyz(group: str) -> Path | None:
    """Return the newest stride‑3 extxyz file for *group*, or None."""
    xyz_dir = DATA_DIR / group / XYZ_DIR
    if not xyz_dir.is_dir():
        print(f"[warn]     {group}: no xyz_files directory, skipping")
        return None

    # Match files named  {group}_{stride}_{max}.extxyz  with stride == FRAME_STRIDE
    pattern = re.compile(rf"^{re.escape(group)}_(\d+)_(\d+)\.extxyz$")
    candidates: list[tuple[int, Path]] = []
    for p in xyz_dir.iterdir():
        m = pattern.match(p.name)
        if m and int(m.group(1)) == FRAME_STRIDE:
            candidates.append((int(m.group(2)), p))

    if not candidates:
        print(f"[warn]     {group}: no stride‑{FRAME_STRIDE} extxyz found, skipping")
        return None

    # Pick the file with the largest frame count (most complete)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_group_paths() -> dict[str, Path]:
    """Map every GROUPS_LIF entry to its stride‑3 extxyz file."""
    paths: dict[str, Path] = {}
    for group in GROUPS_LIF:
        p = find_latest_extxyz(group)
        if p is not None:
            paths[group] = p
    return paths


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_extxyz(group_paths: dict[str, Path]) -> Path:
    """Concatenate per‑group extxyz files into a single merged file.

    Returns the path to the merged file.
    """
    all_frames: list = []
    for group in GROUPS_LIF:
        if group not in group_paths:
            continue
        path = group_paths[group]
        frames = read(str(path), index=":")
        print(f"[read]     {group}: {len(frames)} frames  ←  {path.name}")
        all_frames.extend(frames)

    total = len(all_frames)
    print(f"\nTotal frames across all groups: {total}")

    merged_path = DATA_DIR / XYZ_DIR / MERGED_FILENAME_DS_2
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

    # ── Build README ──
    group_rows = ""
    for idx, group in enumerate(GROUPS_LIF, 1):
        if group in group_paths:
            n = len(read(str(group_paths[group]), index=":"))
        else:
            n = 0
        group_rows += f"| {idx} | `{group}` | {n} |\n"

    template = TEMPLATE_PATH.read_text()
    readme = (
        template.replace("{{GROUP_TABLE}}", group_rows.rstrip("\n"))
        .replace("{{TOTAL_FRAMES}}", str(total))
        .replace("{{FRAME_STRIDE}}", str(FRAME_STRIDE))
        .replace("{{MERGED_FILENAME}}", MERGED_FILENAME_DS_2)
    )

    readme_path = DATA_DIR / XYZ_DIR / "README.md"
    readme_path.write_text(readme)
    print(f"[write]    Dataset card → {readme_path}")

    return merged_path


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def ensure_repo() -> None:
    """Create the HF dataset repo if it doesn't exist."""
    repo_url = create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"[hf]       Repo ready → {repo_url}")


def upload_file(local_path: Path, path_in_repo: str) -> None:
    """Upload a single file to the HF dataset repo."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"[hf]       Uploaded {path_in_repo}")


def upload_qe_output() -> None:
    """Upload the raw QE .out files to the dataset repo (optional)."""
    for group in GROUPS_LIF:
        out_path = DATA_DIR / group / f"{group}.out"
        if out_path.exists():
            upload_file(out_path, f"qe_outputs/{out_path.name}")
        else:
            print(f"[warn]     {group}: .out file not found, skipping")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Locate per‑group extxyz files
    group_paths = build_group_paths()
    if not group_paths:
        print("[error]    No extxyz files found. Run qe_out_to_extxyz first.")
        return

    # 2. Merge them into a single file + README
    merged_path = merge_extxyz(group_paths)
    readme_path = DATA_DIR / XYZ_DIR / "README.md"

    # 3. Create / ensure HF repo
    ensure_repo()

    # 4. Upload the merged dataset + README
    upload_file(merged_path, MERGED_FILENAME_DS_2)
    upload_file(readme_path, "README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
