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


def llpr_sigma(spec: dict, run_dir: Path, frames: list, device: str, dtype: str) -> np.ndarray:
    from mlpdft.llpr import predict_force_sigma

    sig = predict_force_sigma(run_dir / spec["model_path"], spec.get("head"),
                              run_dir / spec["covariance"], frames, device=device, dtype=dtype)
    return np.concatenate(sig)


def predict(arm: dict, run_dir: Path, frames: list, device: str, dtype: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean forces and their uncertainty (None for a single model without UQ)."""
    if arm["kind"] == "llpr":
        mean = member_forces(run_dir / arm["model_path"], arm.get("head"), frames, device, dtype)
        return mean, llpr_sigma(arm, run_dir, frames, device, dtype)
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


def load_arms(run_dir: Path) -> dict:
    """Arms of one run directory, or of every sub-run below it that has a manifest."""
    manifests = [run_dir / "arms_manifest.json"] if (run_dir / "arms_manifest.json").exists() \
        else sorted(run_dir.glob("*/arms_manifest.json"))
    if not manifests:
        raise SystemExit(f"no arms_manifest.json under {run_dir}")
    arms = {}
    foundation = None
    for manifest in manifests:
        content = json.loads(manifest.read_text())
        # uq_compare writes {"format_version", "foundation_model", "arms": {...}}
        foundation = foundation or content.get("foundation_model")
        for name, arm in content.get("arms", content).items():
            if not isinstance(arm, dict) or "kind" not in arm:
                continue
            if name in arms:
                raise SystemExit(f"arm {name} appears in two runs under {run_dir}")
            arms[name] = {**arm, "_base": str(manifest.parent)}
    if foundation and "zero-shot" not in arms:  # accuracy reference on every set
        arms["zero-shot"] = {"kind": "ensemble", "_base": str(run_dir),
                             "members": [{"name": "foundation", "model_path": foundation, "head": None}]}
    return arms


def orx_report(payload: dict) -> None:
    print("::orx:: " + json.dumps(payload), flush=True)


def evaluate(run_dir: Path, dataset_dir: Path, *, device: str, dtype: str, block_steps: int,
             n_boot: int, target: float, predictor=predict, report_orx: bool = False,
             max_frames: int | None = None) -> dict:
    arms = load_arms(run_dir)
    sets = json.loads((dataset_dir / "manifest.json").read_text())["sets"]
    eval_sets = {name: s for name, s in sets.items() if s["role"] not in ("train", "monitor")}
    calibration = next((n for n, s in eval_sets.items() if s["role"] == "calibration"), None)
    id_test = next((n for n, s in eval_sets.items() if s["role"] == "test"), None)
    id_note = "replica test set"
    if id_test is None and "test_timeblock" in eval_sets:
        id_test, id_note = "test_timeblock", "time-block holdout (replicas not yet available)"

    frames = {name: read(dataset_dir / f"{name}.extxyz", index=":") for name in eval_sets}
    if max_frames:  # evenly spaced subset per set (cheap smoke runs)
        frames = {name: [fr[i] for i in np.unique(np.linspace(0, len(fr) - 1, max_frames).round().astype(int))]
                  if len(fr) > max_frames else fr for name, fr in frames.items()}
    out_dir = run_dir / "evaluation"
    report: dict = {"calibration_set": calibration, "id_reference_set": id_test, "id_reference_note": id_note,
                    "conformal_target": target, "arms": {}}
    total, done = len(arms) * len(frames), 0
    if report_orx:
        orx_report({"phase": "evaluate", "step": 0, "of": total, "unit": "arm x set"})
    for arm_name, arm in arms.items():
        preds = {}
        for name, fr in frames.items():
            ref = np.concatenate([a.arrays[FORCES_KEY] for a in fr])
            mean, sigma = predictor(arm, Path(arm.get("_base", run_dir)), fr, device, dtype)
            done += 1
            if report_orx:
                orx_report({"step": done})
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
            if report_orx:
                p = row.get("point", row)
                m = {"arm": arm_name, "set": name, "force_rmse_meV": round(1000 * p["force_rmse"], 3)}
                if "spearman_frame" in p:
                    m.update({"rho_frame": round(p["spearman_frame"], 4), "rho_atom": round(p["spearman_atom"], 4),
                              "miscal": round(p.get("miscal_area_cal", p["miscal_area_raw"]), 4)})
                if "shift_auroc_vs_id" in row:
                    m["auroc"] = round(row["shift_auroc_vs_id"], 4)
                orx_report({"row": "results", "m": m})
                if name == id_test:
                    key = arm_name.replace("-", "_")
                    signals = {f"force_rmse_{key}": m["force_rmse_meV"]}
                    if "rho_frame" in m:
                        signals[f"rho_frame_{key}"] = m["rho_frame"]
                    orx_report({"m": signals})
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
    ap.add_argument("--report-orx", action="store_true", help="print ::orx:: progress and result rows")
    ap.add_argument("--max-frames-per-set", type=int, default=None,
                    help="evaluate an evenly spaced subset of each set (smoke runs)")
    args = ap.parse_args()
    report = evaluate(args.run_dir, args.dataset_dir, device=args.device, dtype=args.dtype,
                      block_steps=args.block_steps, n_boot=args.n_boot, target=args.conformal_target,
                      report_orx=args.report_orx, max_frames=args.max_frames_per_set)
    print_summary(report)


if __name__ == "__main__":
    main()
