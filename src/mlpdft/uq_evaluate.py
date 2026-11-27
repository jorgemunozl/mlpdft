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
* for every ensemble, each member's force errors are saved (``member_err``) and
  ``uq_metrics.ensemble_diversity`` is reported per set; for every arm with an
  uncertainty, ``uq_metrics.shift_growth`` compares each set with the
  in-distribution test set (``assets/research/methods/ensemble-diversity.md``);
* bootstrap blocks are replicas (``run``) when a set has several runs,
  otherwise time blocks of ``--block-steps`` SCF steps.

Models are evaluated in batches of frames on the GPU; the bootstrap of the
different sets runs in parallel CPU processes.

Usage
-----
    python -m mlpdft.uq_evaluate --run-dir <run> --dataset-dir <dataset> --device cuda
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase.io import read

from mlpdft import timing, uq_metrics

FORCES_KEY = "REF_forces"
BATCH_SIZE = 16  # frames per forward pass when evaluating a model


_MODELS: dict[tuple, object] = {}  # a few loaded models, reused across the sets of an arm
_MAX_MODELS = 4


def _model(path: Path, device: str, dtype: str):
    from mlpdft.mace_batches import load_model

    key = (str(path), device, dtype)
    if key not in _MODELS:
        if len(_MODELS) >= _MAX_MODELS:
            _MODELS.pop(next(iter(_MODELS)))
        model = load_model(path, device, dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _MODELS[key] = model
    return _MODELS[key]


def _forces(model, data: list, head_index: int, device: str, batch_size: int) -> np.ndarray:
    from mlpdft.mace_batches import collate, with_head

    out = []
    with timing.timed({"kind": "eval_member", "frames": len(data),
                       "atoms": int(sum(int(d.num_nodes) for d in data))}):
        for start in range(0, len(data), batch_size):
            batch = with_head(collate(data[start:start + batch_size], device), head_index)
            prediction = model(batch, training=False, compute_force=True)
            out.append(prediction["forces"].detach().cpu().numpy())
    return np.concatenate(out).astype(float)


def ensemble_forces(members: list[dict], run_dir: Path, frames: list, device: str, dtype: str,
                    batch_size: int | None = None) -> np.ndarray:
    """Forces of every member on every frame, shape (members, atoms, 3).

    Builds the same graphs as ``MACECalculator`` (``mlpdft.mace_batches``),
    once per model architecture rather than once per member, evaluates
    ``batch_size`` frames per pass (default ``BATCH_SIZE``), and resolves each
    head explicitly instead of falling back silently to another one.
    """
    from mlpdft.mace_batches import frames_to_data, model_heads, resolve_head

    batch_size = batch_size or BATCH_SIZE
    graphs: dict[tuple, list] = {}
    out = []
    for member in members:
        model = _model(run_dir / member["model_path"], device, dtype)
        name, index = resolve_head(model, member.get("head"))
        key = (tuple(int(z) for z in model.atomic_numbers), float(model.r_max), tuple(model_heads(model)))
        if key not in graphs:  # the head index is set per batch, so heads share graphs
            with timing.timed({"kind": "eval_graphs", "frames": len(frames),
                               "atoms": int(sum(len(a) for a in frames))}):
                graphs[key] = frames_to_data(model, frames, name)
        out.append(_forces(model, graphs[key], index, device, batch_size))
    return np.stack(out)


def member_forces(model_path: Path, head: str | None, frames: list, device: str, dtype: str,
                  batch_size: int | None = None) -> np.ndarray:
    """Forces of one model (one head) on every frame, concatenated over atoms."""
    member = {"model_path": str(Path(model_path).resolve()), "head": head}
    return ensemble_forces([member], Path("/"), frames, device, dtype, batch_size)[0]


def llpr_forces_and_sigma(spec: dict, run_dir: Path, frames: list, device: str,
                          dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Model forces and LLPR force sigmas from one pass over the frames."""
    from mlpdft.llpr import predict as llpr_predict

    results = llpr_predict(run_dir / spec["model_path"], spec.get("head"),
                           run_dir / spec["covariance"], frames, device=device, dtype=dtype)
    return (np.concatenate([r["forces"] for r in results]),
            np.concatenate([r["force_sigma"] for r in results]))


def predict(arm: dict, run_dir: Path, frames: list, device: str, dtype: str) -> tuple:
    """Mean forces and their uncertainty (None for a single model without UQ).

    An ensemble also returns every member's forces, shape (members, atoms, 3),
    as a third element. A custom ``predictor`` may return only the first two.
    """
    if arm["kind"] == "llpr":
        return llpr_forces_and_sigma(arm, run_dir, frames, device, dtype)
    members = ensemble_forces(arm["members"], run_dir, frames, device, dtype)
    if len(members) == 1:
        return members[0], None
    return members.mean(axis=0), members.std(axis=0, ddof=1), members


def blocks_for(frames: list, block_steps: int) -> tuple[np.ndarray, str]:
    runs = [a.info.get("run", "") for a in frames]
    if len(set(runs)) > 1:
        labels = {r: i for i, r in enumerate(sorted(set(runs)))}
        return np.array([labels[r] for r in runs]), "replica"
    steps = np.array([int(a.info.get("scf_step", i)) for i, a in enumerate(frames)])
    return steps // block_steps, f"time blocks of {block_steps} steps"


def load_arms(run_dir: Path, zero_shot: bool = True) -> dict:
    """Arms of one run directory, or of every sub-run below it that has a manifest.

    ``zero_shot`` adds the foundation model as an accuracy reference when no
    run trained it; pass False when evaluating one arm group of a larger run.
    """
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
    if zero_shot and foundation and "zero-shot" not in arms:  # accuracy reference on every set
        arms["zero-shot"] = {"kind": "ensemble", "_base": str(run_dir),
                             "members": [{"name": "foundation", "model_path": foundation, "head": None}]}
    return arms


def orx_report(payload: dict) -> None:
    print("::orx:: " + json.dumps(payload), flush=True)


def _summarize_task(args: tuple) -> dict:
    err, sigma, frame, block, alpha, q, target, n_boot = args
    return uq_metrics.summarize(err, sigma, frame, block, alpha=alpha, conformal_q=q,
                                target=target, n_boot=n_boot)


def _summaries(tasks: dict[str, tuple], workers: int) -> dict[str, dict]:
    """``uq_metrics.summarize`` for every set, in parallel processes when ``workers`` > 1."""
    if workers <= 1 or len(tasks) <= 1:
        return {name: _summarize_task(args) for name, args in tasks.items()}
    context = multiprocessing.get_context("spawn")  # never fork a process holding a CUDA context
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context) as pool:
        futures = {name: pool.submit(_summarize_task, args) for name, args in tasks.items()}
        return {name: future.result() for name, future in futures.items()}


def evaluate(run_dir: Path, dataset_dir: Path, *, device: str, dtype: str, block_steps: int,
             n_boot: int, target: float, predictor=predict, report_orx: bool = False,
             max_frames: int | None = None, zero_shot: bool = True, workers: int = 1,
             orx_rows_only: bool = False) -> dict:
    arms = load_arms(run_dir, zero_shot=zero_shot)
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
                    "conformal_target": target, "dtype": dtype, "arms": {}}
    total, done = len(arms) * len(frames), 0
    progress = report_orx and not orx_rows_only  # the run script reports its own progress
    if progress:
        orx_report({"phase": "evaluate", "step": 0, "of": total, "unit": "arm x set"})
    for arm_name, arm in arms.items():
        preds = {}
        diversity = {}
        for name, fr in frames.items():
            ref = np.concatenate([a.arrays[FORCES_KEY] for a in fr])
            out = predictor(arm, Path(arm.get("_base", run_dir)), fr, device, dtype)
            mean, sigma = out[0], out[1]
            members = out[2] if len(out) > 2 else None
            done += 1
            if progress:
                orx_report({"step": done})
            frame = np.repeat(np.arange(len(fr)), [len(a) for a in fr])
            preds[name] = (mean - ref, sigma, frame)
            arrays = {"err": mean - ref, "frame": frame,
                      "sigma": np.full_like(mean, np.nan) if sigma is None else sigma,
                      "scf_step": np.array([a.info.get("scf_step", -1) for a in fr]),
                      "run": np.array([a.info.get("run", "") for a in fr])}
            if members is not None:
                member_err = members - ref[None]
                arrays["member_err"] = member_err.astype(np.float32)
                diversity[name] = uq_metrics.ensemble_diversity(member_err)
            (out_dir / arm_name).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / arm_name / f"{name}.npz", **arrays)
        alpha = q = None
        if calibration is not None and preds[calibration][1] is not None:
            e_cal, s_cal, _ = preds[calibration]
            alpha = uq_metrics.nll_scale(e_cal, s_cal)
            q = uq_metrics.conformal_multiplier(e_cal, s_cal, target)
        blocks = {name: blocks_for(frames[name], block_steps) for name in preds}
        tasks = {name: (err, sigma, frame, blocks[name][0], alpha, q, target, n_boot)
                 for name, (err, sigma, frame) in preds.items() if sigma is not None}
        with timing.timed({"kind": "bootstrap", "arm": arm_name, "sets": len(tasks), "n_boot": n_boot,
                           "workers": workers}):
            summaries = _summaries(tasks, workers)
        arm_report = {"alpha": alpha, "conformal_q": q, "sets": {}}
        for name, (err, sigma, frame) in preds.items():
            if sigma is None:
                row = {"force_rmse": float(np.sqrt(np.mean(err ** 2))), "force_mae": float(np.mean(np.abs(err))),
                       "note": "single model: accuracy only"}
            else:
                row = summaries[name]
                if id_test is not None and name != id_test:
                    e_id, s_id, f_id = preds[id_test]
                    row["shift_auroc_vs_id"] = uq_metrics.shift_auroc(s_id, f_id, sigma, frame)
                    row["shift_growth"] = uq_metrics.shift_growth(err, sigma, e_id, s_id)
                if name in diversity:
                    row["diversity"] = diversity[name]
            row.update({"role": eval_sets[name]["role"], "blocks": blocks[name][1]})
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
    out_dir.mkdir(parents=True, exist_ok=True)
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
            if "diversity" in row:
                d = row["diversity"]
                line += f"  err-corr {d['error_correlation']:.2f}  mean/best {d['mean_gain']:.2f}"
            print(line)


def main() -> None:
    global BATCH_SIZE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--block-steps", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--conformal-target", type=float, default=0.9)
    ap.add_argument("--report-orx", action="store_true", help="print ::orx:: progress and result rows")
    ap.add_argument("--orx-rows-only", action="store_true",
                    help="with --report-orx: result rows only, no phase or step progress")
    ap.add_argument("--max-frames-per-set", type=int, default=None,
                    help="evaluate an evenly spaced subset of each set (smoke runs)")
    ap.add_argument("--no-zero-shot", action="store_true",
                    help="do not add the foundation model (evaluating one arm group of a larger run)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="frames per forward pass")
    ap.add_argument("--llpr-jacobian-chunk", type=int, default=1,
                    help="feature rows per LLPR backward pass (see mlpdft.llpr)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="CPU processes for the bootstrap (1 = in process)")
    args = ap.parse_args()
    if args.device == "cuda":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    BATCH_SIZE = max(1, args.batch_size)
    from mlpdft import llpr

    llpr.JACOBIAN_CHUNK = max(1, args.llpr_jacobian_chunk)
    report = evaluate(args.run_dir, args.dataset_dir, device=args.device, dtype=args.dtype,
                      block_steps=args.block_steps, n_boot=args.n_boot, target=args.conformal_target,
                      report_orx=args.report_orx, max_frames=args.max_frames_per_set,
                      zero_shot=not args.no_zero_shot, workers=args.workers,
                      orx_rows_only=args.orx_rows_only)
    print_summary(report)


if __name__ == "__main__":
    main()
