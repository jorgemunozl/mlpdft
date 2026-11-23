#!/usr/bin/env python3
"""Build the 40 Ry shift-ladder dataset (ADR-0007, ADR-0008) from QE MD logs.

Every set comes from one 40/320 Ry run; the recipe is the table ``SETS``.
Training frames use a fixed stride (ADR-0003). The last 3,000 steps of the
training trajectory are held out, after a 120-step guard, as a time-block test
set: it reproduces the earlier single-trajectory split, so the gap to the
independent replicas measures how much that split inflates the results.
Calibration and in-distribution test sets come later from the replicas of
the 2026-11 QE campaign (``--replica`` adds them).

Usage
-----
    PYTHONPATH=src python3 src/mlpdft/utils/build_shift_ladder_dataset.py \
        --dataset-root /home/jorge/project/mlpdft/dataset \
        --output-dir outputs/shift_ladder_40ry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from ase.io import write

from extract_qe_trajectory import extract

TRAIN_RUN = "LIF64_KJPAW_FINAL"
HOLDOUT_START = 15_980  # last 3,000 SCF steps of the 18,980-step run
GUARD = 120  # the gap used between time blocks in the earlier build

# name: (role, source run, env label, first step, stop step or None, stride)
SETS = {
    "train_pool": ("train", TRAIN_RUN, "bulk_LiF", 500, HOLDOUT_START - GUARD, 10),
    "monitor": ("monitor", TRAIN_RUN, "bulk_LiF", 505, HOLDOUT_START - GUARD, 150),
    "test_timeblock": ("test_timeblock", TRAIN_RUN, "bulk_LiF", HOLDOUT_START, None, 10),
    # variable cell, no thermostat: kinetic T ~160 K, V 531 +- 4.5 A^3, yet larger forces than the 300 K run
    "shift_bulk_npt": ("shift_state", "LIF64_KJPAW_NPT_FINAL", "bulk_LiF_NPH", 500, None, 25),
    "shift_fdoped": ("shift_chemistry", "LIWITHF_KJPAW_FINAL", "F_doped_Li", 500, None, 20),
    "shift_li_metal": ("shift_chemistry", "BCC_54_KJPAW_FINAL", "Li_metal", 500, None, 25),
    "shift_interface": ("shift_structure", "LIFINTERFACE_KJPAW_V2", "LiF_Li_film", 586, None, 5),
    "far_libf4": ("shift_far", "LIBF4_FINAL", "LiBF4", 500, None, 50),
    "far_bli3": ("shift_far", "BLI3_KJPAW_FINAL", "BLi3", 500, None, 25),
}


def log_header(path: Path) -> dict:
    """Cutoffs and pseudopotential checksums printed in a pw.x log."""
    info: dict = {"md5": []}
    with path.open(errors="ignore") as fh:
        for line in fh:
            if "kinetic-energy cutoff" in line:
                info["ecutwfc_Ry"] = float(re.search(r"(\d+\.\d+)", line).group(1))
            elif "charge density cutoff" in line:
                info["ecutrho_Ry"] = float(re.search(r"(\d+\.\d+)", line).group(1))
            elif "MD5 check sum" in line:
                info["md5"].append(line.split()[-1])
            elif line.startswith("!"):
                break
    return info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="build only these sets")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sets": {}}
    manifest["recipe"] = {
        "builder": "src/mlpdft/utils/build_shift_ladder_dataset.py",
        "holdout_start_step": HOLDOUT_START, "guard_steps": GUARD,
        "pairing": "SCF block k with the positions printed in MD iteration k (as ASE)",
        "units": "QE CODATA 2006 constants",
    }
    seen: dict[tuple[str, int], str] = {}
    for name, (role, run, env, first, stop, stride) in SETS.items():
        if args.only and name not in args.only:
            continue
        log = args.dataset_root / run / f"{run}.out"
        if not log.exists():  # a few runs keep their log under another name
            candidates = sorted((args.dataset_root / run).glob("*.out"))
            if len(candidates) != 1:
                raise SystemExit(f"{run}: cannot find a unique .out log")
            log = candidates[0]
        header = log_header(log)
        if header.get("ecutwfc_Ry") != 40.0 or header.get("ecutrho_Ry") != 320.0:
            raise SystemExit(f"{run}: not a 40/320 Ry run ({header})")
        frames = extract(log, env=env, run=run, burn_in=first, stride=stride, stop=stop)
        for atoms in frames:
            key = (run, atoms.info["scf_step"])
            if key in seen:
                raise SystemExit(f"{name}: step {key} already used by {seen[key]}")
            seen[key] = name
            atoms.info["split"] = name
        out = args.output_dir / f"{name}.extxyz"
        write(out, frames, format="extxyz")
        steps = [a.info["scf_step"] for a in frames]
        manifest["sets"][name] = {
            "role": role, "source_run": run, "source_log": log.name, "config_type": env, "frames": len(frames),
            "atoms_per_frame": len(frames[0]), "scf_steps": [steps[0], steps[-1]], "stride": stride,
            "ecutwfc_Ry": header["ecutwfc_Ry"], "ecutrho_Ry": header["ecutrho_Ry"],
            "pseudopotential_md5": header["md5"], "sha256": sha256(out),
        }
        print(f"{name:18s} {len(frames):5d} frames  {run} steps {steps[0]}-{steps[-1]} stride {stride}")
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
