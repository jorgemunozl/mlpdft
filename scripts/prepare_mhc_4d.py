#!/usr/bin/env python3
"""Download and validate the exact inputs for the four-head MHC run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write
from huggingface_hub import hf_hub_download, snapshot_download

DATA_REPO = "jorgemunozl/li-f-uq-interim-two-environments"
DATA_REV = "c2699e8e52889c18f6d29f657eb996ac3f78524f"
MODEL_REPO = "jorgemunozl/mace_omat_medium"
MODEL_REV = "d64b6faf564bd682ba1e9ca337408f6925aa908d"
MODEL_SHA = "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
SHA = {
    "uq_train.extxyz": "9c5f36c3fae1a0eb530d5e71a8d419db2a978c93a474a2b15ad6163563078a3a",
    "uq_valid.extxyz": "b9e50870a1ca72b98b924548a3f20f7a12f3fc23aedd7ed702c936406160d60a",
    "uq_calibration.extxyz": "591d56968f3a1c70fa97198dcd0cc6f3ee79ce4075d6e913f591ca3bfb54c233",
    "uq_test.extxyz": "539ade4f0c58bc3c1a99d6abb5eec68141fc98483aa478c83164ac82ec4ea72e",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def verify_dataset(data_dir: Path) -> dict:
    results = {}
    for name, digest in SHA.items():
        path = data_dir / name
        if sha256(path) != digest:
            raise ValueError(f"Dataset SHA-256 mismatch: {path}")
        frames = read(path, index=":")
        expected = 500 if name == "uq_train.extxyz" else 100
        counts = Counter(a.info.get("config_type") for a in frames)
        if counts != {"bulk_LiF": expected, "F_doped_Li": expected}:
            raise ValueError(f"Unexpected environments in {name}: {counts}")
        for atoms in frames:
            f = atoms.arrays.get("REF_forces")
            e = atoms.info.get("REF_energy")
            if (f is None or np.asarray(f).shape != (len(atoms), 3) or e is None
                    or not np.isfinite(f).all() or not np.isfinite(e)
                    or not atoms.pbc.all() or atoms.get_volume() <= 0):
                raise ValueError(f"Invalid labels or cell in {name}")
        results[name] = {"sha256": digest, "frames": len(frames), "by_group": dict(counts)}
    return results


def verify_head_split(train_path: Path, scratch: Path) -> dict:
    from mlpdft.uq_compare import _make_disjoint_head_files

    frames = read(train_path, index=":")
    _, record = _make_disjoint_head_files(frames, scratch, 4, 20251027)
    used = []
    for head in record["heads"].values():
        if head["n_frames"] != 250 or head["by_group"] != {"bulk_LiF": 125, "F_doped_Li": 125}:
            raise ValueError(f"Incorrect head assignment: {head['by_group']}")
        used.extend(head["frame_indices"])
    if sorted(used) != list(range(1000)):
        raise ValueError("Head split is not an exact disjoint partition")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Check already-present files without downloading")
    parser.add_argument("--allow-cpu", action="store_true", help="Local preflight test only")
    args = parser.parse_args()

    if not args.offline:
        data_ready = all((args.data_dir / name).is_file() and sha256(args.data_dir / name) == digest
                         for name, digest in SHA.items())
        if not data_ready:
            snapshot_download(repo_id=DATA_REPO, repo_type="dataset", revision=DATA_REV,
                              local_dir=str(args.data_dir), allow_patterns=[*SHA, "frame_manifest.csv", "dataset_manifest.json"])
        model_ready = args.model_file.is_file() and sha256(args.model_file) == MODEL_SHA
        if not model_ready:
            model_path = Path(hf_hub_download(repo_id=MODEL_REPO, repo_type="model", revision=MODEL_REV,
                                              filename="mace_omat_medium.model"))
            args.model_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_path, args.model_file)
    if sha256(args.model_file) != MODEL_SHA:
        raise ValueError("Foundation checkpoint SHA-256 mismatch")
    verified = verify_dataset(args.data_dir)
    if not args.allow_cpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        gpu = torch.cuda.get_device_name(0)
        if "A40" not in gpu.upper():
            raise RuntimeError(f"Expected an A40, found {gpu}")
    else:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU preflight"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        head_record = verify_head_split(args.data_dir / "uq_train.extxyz", Path(tmp))
    validation = read(args.data_dir / "uq_valid.extxyz", index=":")
    smoke = [a for group in ("bulk_LiF", "F_doped_Li")
             for a in [x for x in validation if x.info["config_type"] == group][:4]]
    write(args.output_dir / "smoke_validation.extxyz", smoke, format="extxyz")
    summary = {"dataset_repo": DATA_REPO, "dataset_revision": DATA_REV, "files": verified,
               "model_repo": MODEL_REPO, "model_revision": MODEL_REV, "model_sha256": MODEL_SHA,
               "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
               "head_split": head_record, "dataset_caveat": "development only; F-doped burn-in and dependence unresolved"}
    (args.output_dir / "preflight.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Preflight passed: pinned model and data, four disjoint 125/125 heads,", gpu, flush=True)


if __name__ == "__main__":
    main()
