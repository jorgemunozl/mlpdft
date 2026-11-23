"""Evaluate trained UQ arms on every set of a dataset and compute the metrics.

Reads ``arms_manifest.json`` from a ``uq_compare`` run directory:

    {"<arm>": {"kind": "ensemble", "members": [{"name", "model_path", "head"}, ...]},
     "llpr":  {"kind": "llpr", "model_path", "head", "covariance"}}

(paths relative to the run directory) and the dataset ``manifest.json`` written
by ``build_shift_ladder_dataset.py`` (set name -> role and file). For each arm
and set it saves per-atom errors and uncertainties, then computes
``uq_metrics.summarize``:

* calibration (``alpha`` by NLL, split-conformal ``q``) is fitted on the set
  whose role is ``calibration`` and applied unchanged to every other set;
* shift-detection AUROC is computed against the in-distribution test set
  (role ``test``; until the replicas exist, ``test_timeblock`` is used and the
  output says so);
* bootstrap blocks are replicas (``run``) when a set has several runs,
  otherwise time blocks of ``--block-steps`` SCF steps.

Usage
-----
    python -m mlpdft.uq_evaluate --run-dir <run> --dataset-dir <dataset> --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from mlpdft import uq_metrics

FORCES_KEY = "REF_forces"


def member_forces(model_path: Path, head: str | None, frames: list, device: str, dtype: str) -> np.ndarray:
    """Forces of one model (one head) on every frame, concatenated over atoms."""
    from mace.calculators import MACECalculator

    kwargs = {"model_paths": [str(model_path)], "device": device, "default_dtype": dtype}
    if head is not None:
        kwargs["head"] = head
    calc = MACECalculator(**kwargs)
    out = []
    for atoms in frames:
        a = atoms.copy()
        a.calc = calc
        out.append(a.get_forces())
    return np.concatenate(out)


def llpr_sigma(spec: dict, run_dir: Path, frames: list) -> np.ndarray:
    from mlpdft.llpr import predict_force_sigma

    sig = predict_force_sigma(run_dir / spec["model_path"], spec.get("head"),
                              run_dir / spec["covariance"], frames)
    return np.concatenate(sig)


def predict(arm: dict, run_dir: Path, frames: list, device: str, dtype: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean forces and their uncertainty (None for a single model without UQ)."""
    if arm["kind"] == "llpr":
        mean = member_forces(run_dir / arm["model_path"], arm.get("head"), frames, device, dtype)
        return mean, llpr_sigma(arm, run_dir, frames)
    members = np.stack([member_forces(run_dir / m["model_path"], m.get("head"), frames, device, dtype)
                        for m in arm["members"]])
    if len(members) == 1:
        return members[0], None
    return members.mean(axis=0), members.std(axis=0, ddof=1)


def blocks_for(frames: list, block_steps: int) -> tuple[np.ndarray, str]:
    runs = [a.info.get("run", "") for a in frames]
    if len(set(runs)) > 1:
        labels = {r: i for i, r in enumerate(sorted(set(runs)))}
        return np.array([labels[r] for r in runs]), "replica"
    steps = np.array([int(a.info.get("scf_step", i)) for i, a in enumerate(frames)])
    return steps // block_steps, f"time blocks of {block_steps} steps"


def evaluate(run_dir: Path, dataset_dir: Path, *, device: str, dtype: str, block_steps: int,
             n_boot: int, target: float, predictor=predict) -> dict:
    arms = json.loads((run_dir / "arms_manifest.json").read_text())
    sets = json.loads((dataset_dir / "manifest.json").read_text())["sets"]
    eval_sets = {name: s for name, s in sets.items() if s["role"] not in ("train", "monitor")}
    calibration = next((n for n, s in eval_sets.items() if s["role"] == "calibration"), None)
    id_test = next((n for n, s in eval_sets.items() if s["role"] == "test"), None)
    id_note = "replica test set"
    if id_test is None and "test_timeblock" in eval_sets:
        id_test, id_note = "test_timeblock", "time-block holdout (replicas not yet available)"

    frames = {name: read(dataset_dir / f"{name}.extxyz", index=":") for name in eval_sets}
    out_dir = run_dir / "evaluation"
    report: dict = {"calibration_set": calibration, "id_reference_set": id_test, "id_reference_note": id_note,
                    "conformal_target": target, "arms": {}}
    for arm_name, arm in arms.items():
        preds = {}
        for name, fr in frames.items():
            ref = np.concatenate([a.arrays[FORCES_KEY] for a in fr])
            mean, sigma = predictor(arm, run_dir, fr, device, dtype)
            frame = np.repeat(np.arange(len(fr)), [len(a) for a in fr])
            preds[name] = (mean - ref, sigma, frame)
            (out_dir / arm_name).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / arm_name / f"{name}.npz", err=mean - ref, frame=frame,
                                sigma=np.full_like(mean, np.nan) if sigma is None else sigma,
                                scf_step=np.array([a.info.get("scf_step", -1) for a in fr]),
                                run=np.array([a.info.get("run", "") for a in fr]))
        alpha = q = None
        if calibration is not None and preds[calibration][1] is not None:
            e_cal, s_cal, _ = preds[calibration]
            alpha = uq_metrics.nll_scale(e_cal, s_cal)
            q = uq_metrics.conformal_multiplier(e_cal, s_cal, target)
        arm_report = {"alpha": alpha, "conformal_q": q, "sets": {}}
        for name, (err, sigma, frame) in preds.items():
            block, block_kind = blocks_for(frames[name], block_steps)
            if sigma is None:
                row = {"force_rmse": float(np.sqrt(np.mean(err ** 2))), "force_mae": float(np.mean(np.abs(err))),
                       "note": "single model: accuracy only"}
            else:
                row = uq_metrics.summarize(err, sigma, frame, block, alpha=alpha, conformal_q=q,
                                           target=target, n_boot=n_boot)
                if id_test is not None and name != id_test:
                    _, s_id, f_id = preds[id_test]
                    row["shift_auroc_vs_id"] = uq_metrics.shift_auroc(s_id, f_id, sigma, frame)
            row.update({"role": eval_sets[name]["role"], "blocks": block_kind})
            arm_report["sets"][name] = row
        report["arms"][arm_name] = arm_report
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def print_summary(report: dict) -> None:
    print(f"calibration set: {report['calibration_set']}; in-distribution reference: "
          f"{report['id_reference_set']} ({report['id_reference_note']})")
    for arm, ar in report["arms"].items():
        for name, row in ar["sets"].items():
            p = row.get("point", row)
            line = f"{arm:20s} {name:16s} force RMSE {1000 * p['force_rmse']:6.1f} meV/A"
            if "spearman_frame" in p:
                line += f"  rho_frame {p['spearman_frame']:.2f}  miscal {p.get('miscal_area_cal', p['miscal_area_raw']):.3f}"
            if "shift_auroc_vs_id" in row:
                line += f"  AUROC {row['shift_auroc_vs_id']:.2f}"
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--block-steps", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--conformal-target", type=float, default=0.9)
    args = ap.parse_args()
    report = evaluate(args.run_dir, args.dataset_dir, device=args.device, dtype=args.dtype,
                      block_steps=args.block_steps, n_boot=args.n_boot, target=args.conformal_target)
    print_summary(report)


if __name__ == "__main__":
    main()
