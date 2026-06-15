#!/usr/bin/env python3
"""Merge per-group .extxyz files into one dataset and upload to Hugging Face."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
from huggingface_hub import HfApi, create_repo

from mlpdft.config import Mace_TrainerConfig
from mlpdft.constants import DATA_DIR, GROUPS_LIF, XYZ_DIR

# ---------- settings ----------
HF_REPO_ID = "jorgemunozl/minimal_li_f_mace_dataset"
FRAME_STRIDE = 5
MAX_FRAMES = None  # use all frames after striding
MERGED_FILENAME = "minimal_li_f_mace_dataset.extxyz"
# ------------------------------

# Template path (sibling of this script)
TEMPLATE_PATH = Path(__file__).resolve().parent / "dataset_readme_template.md"

# ---------------------------------------------------------------------------
# 1. Resolve expected .extxyz path per group
# ---------------------------------------------------------------------------
group_paths: dict[str, Path] = {}
for group in GROUPS_LIF:
    conf = Mace_TrainerConfig(
        group=group,
        frame_stride=FRAME_STRIDE,
        max_frames=MAX_FRAMES,
    )
    group_paths[group] = conf.data_out_path

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

readme_path = merged_path.with_name("README.md")
readme_path.write_text(readme)
print(f"[write]    Dataset card → {readme_path}")

# ---------------------------------------------------------------------------
# 5. Upload to Hugging Face Hub
# ---------------------------------------------------------------------------
api = HfApi()

repo_url = create_repo(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    exist_ok=True,
)
print(f"[hf]       Repo ready → {repo_url}")

api.upload_file(
    path_or_fileobj=str(merged_path),
    path_in_repo=MERGED_FILENAME,
    repo_id=HF_REPO_ID,
    repo_type="dataset",
)
print(f"[hf]       Uploaded {MERGED_FILENAME}")

api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
)
print(f"[hf]       Uploaded README.md")

print("\n✅ Done.")
