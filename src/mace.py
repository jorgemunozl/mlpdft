from __future__ import annotations

from config import MaceConfig
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from mace.calculators import mace_mp
from ase.atoms import Atoms

class MACE(nn.Module):
    def __init__(self, config: MaceConfig):
        self.config = config

    def build_calculator(self):
        return mace_mp(model=self.config.model,
                        device=self.config.device,
                        default_dtype=self.config.dtype
            )

    def evaluate(self, atoms: Atoms, calculator) -> dict:



    def print_summary(self, results: list, config: MaceEvalConfig, summary_title: str, model_label: str) -> None:
        _print_and_write_summary(results, config, summary_title, model_label)


def _eval_extxyz(config: MaceConfig) -> int:
    xyz_path = config.test_extxyz.resolve()
    if not xyz_path.is_file():
        print(f"Error: test_extxyz not found: {xyz_path}", file=sys.stderr)
        return 1

    calc, model_label = config.build_calculator()

    from ase.io import read

    frames = read(str(xyz_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if config.max_frames is not None:
        frames = frames[: config.max_frames]

    results = []
    for i, atoms in enumerate(frames):
        atoms = attach_reference_from_calc(atoms)
        metrics = evaluate_mace_on_atoms(atoms, calc)
        metrics["filename"] = f"{xyz_path.name}#{i}"
        metrics["filepath"] = str(xyz_path)
        results.append(metrics)

    _print_and_write_summary(
        results,
        config,
        summary_title=f"extxyz: {xyz_path.name}",
        model_label=model_label,
    )
    return 0


def _collect_test_json_paths(config: MaceConfig) -> List[Path]:
    json_root = config.resolve_json_root()
    if not json_root.exists():
        raise FileNotFoundError(f"JSON root not found: {json_root}")

    if config.perconfig is not None:
        print(f"Reading perconfig from: {config.perconfig}")
        rows = parse_perconfig(config.perconfig)
        test_rows = [r for r in rows if r.testing_bool]
        print(f"Found {len(rows)} total configs, {len(test_rows)} test configs")

        test_files: List[Path] = []
        for row in test_rows:
            json_path = json_root / row.group / row.filename
            if not json_path.exists():
                json_path = json_root / row.filename
            if json_path.exists():
                test_files.append(json_path)
            else:
                print(f"Warning: JSON not found: {json_path}", file=sys.stderr)
        return test_files

    print(f"Computing train/test split from: {json_root}")
    all_json = sorted(json_root.glob("*.json"))
    if not all_json:
        raise FileNotFoundError(f"No JSON files found in {json_root}")

    train_files, test_files = compute_fitsnap_split(
        all_json, config.training_frac, config.testing_frac
    )
    print(f"Total: {len(all_json)}, Train: {len(train_files)}, Test: {len(test_files)}")
    return test_files


 def build_calculator(self):
        """Return (calculator, human-readable model label)."""
        if self.mace_model is not None:
            from mace.calculators import MACECalculator

            path = self.mace_model.resolve()
            calc = MACECalculator(
                model_paths=str(path),
                device=self.device,
                default_dtype=self.dtype,
            )
            return calc, str(path)

        from mace.calculators import mace_mp

        calc = mace_mp(
            model=self.model, device=self.device, default_dtype=self.dtype
        )
        return calc, f"mace_mp:{self.model}"


def _eval_json(config: MaceConfig) -> int:
    """

    """
    try:
        test_files = _collect_test_json_paths(config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not test_files:
        print("Error: No test files to process", file=sys.stderr)
        return 1

    if config.max_frames is not None:
        test_files = test_files[: config.max_frames]
        print(f"Limited to {len(test_files)} test frames")

    calc, model_label = config.build_calculator()
    print(f"Processing {len(test_files)} test configurations...")

    results = []
    for i, json_path in enumerate(test_files):
        try:
            atoms = load_json_as_atoms(json_path)
            metrics = evaluate_mace_on_atoms(atoms, calc)
            metrics["filename"] = json_path.name
            metrics["filepath"] = str(json_path)
            results.append(metrics)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processed {i + 1}/{len(test_files)}: {json_path.name}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}", file=sys.stderr)
            continue

    _print_and_write_summary(
        results,
        config,
        summary_title="JSON test split",
        model_label=model_label,
    )
    return 0


def run(config: MaceConfig) -> int:
    """Run MACE evaluation with the given configuration."""
    if config.test_extxyz is not None:
        return _eval_extxyz(config)
    return _eval_json(config)


if __name__ == "__main__":
    path_to_test_xyz = Path("../../model_LiF/test.xyz")

    cfg = MaceEvalConfig(
        test_extxyz=path_to_test_xyz,
        mace_model=None,
        out_csv=Path("outputs/mace_eval_results.csv"),
    )

    raise SystemExit(run(cfg))
