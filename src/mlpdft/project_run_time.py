"""Project the time of a shift-ladder run from a smoke run's timing lines.

The smoke run (``configs/shift_ladder/smoke.env``) trains every arm for a few
epochs on a small rung and evaluates a few frames per set. Its ``[timing]``
lines (``mlpdft.timing``) give the training cost per frame and epoch for each
kind of model, the cost per atom of LLPR and of evaluation, and the fixed cost
of every step. This scales them to the settings of another node file (the
baseline, ``configs/shift_ladder/node.env``) and prints hours per arm group.

Epoch 0 of each model is left out of the rates (it includes warm-up).

Usage
-----
    python -m mlpdft.project_run_time --log run_outputs/<smoke>/run.log \
        --measured configs/shift_ladder/smoke.env --target configs/shift_ladder/node.env \
        --dataset-dir run_outputs/<smoke>/data --deadline-minutes 630
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlpdft import timing
from mlpdft.snapshot_schedule import snapshot_schedule

CONTINGENCY = 0.10  # added to the projected total
FINAL_SECONDS = 300  # merging the metrics and the last upload


def read_env(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of a node file (comments and quotes handled by shlex)."""
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        values[key.strip()] = " ".join(shlex.split(raw, comments=True))
    return values


@dataclass
class Settings:
    labels: int
    groups: list[str]
    epochs: int
    mhc_epochs: int
    probe_epochs: int
    shallow_epochs: int
    mhc_heads: int
    shallow_heads: int
    snapshot_count: int
    llpr_train_frames: int
    n_boot: int
    max_frames_per_set: int | None
    cal_frames: int  # 0 = every monitor frame
    snapshot_burn_in_cycles: int = 0
    snapshot_cycle_epochs: int | None = None


def settings(env: dict[str, str]) -> Settings:
    train = argparse.ArgumentParser(add_help=False)
    for flag, default in (("--epochs", 150), ("--mhc-epochs", 120), ("--probe-epochs", 600),
                          ("--shallow-epochs", 120), ("--mhc-heads", 8), ("--shallow-heads", 4),
                          ("--snapshot-count", 3), ("--llpr-max-train-frames", 300)):
        train.add_argument(flag, type=int, default=default)
    train.add_argument("--snapshot-burn-in-cycles", type=int, default=0)
    train.add_argument("--snapshot-cycle-epochs", type=int, default=None)
    t, _ = train.parse_known_args(shlex.split(env.get("EXTRA_ARGS", "")))
    evaluation = argparse.ArgumentParser(add_help=False)
    evaluation.add_argument("--n-boot", type=int, default=1000)
    evaluation.add_argument("--max-frames-per-set", type=int, default=None)
    e, _ = evaluation.parse_known_args(shlex.split(env.get("EVAL_ARGS", "")))
    labels = int(env["LABELS"])
    return Settings(labels=labels, groups=env.get("ARM_GROUPS", "").split(), epochs=t.epochs,
                    mhc_epochs=t.mhc_epochs, probe_epochs=t.probe_epochs, shallow_epochs=t.shallow_epochs,
                    mhc_heads=t.mhc_heads, shallow_heads=t.shallow_heads, snapshot_count=t.snapshot_count,
                    llpr_train_frames=min(t.llpr_max_train_frames, labels), n_boot=e.n_boot,
                    max_frames_per_set=e.max_frames_per_set, cal_frames=int(env.get("UQC_CAL_FRAMES", "0") or 0),
                    snapshot_burn_in_cycles=t.snapshot_burn_in_cycles, snapshot_cycle_epochs=t.snapshot_cycle_epochs)


@dataclass
class Data:
    train_atoms: int  # atoms per training frame
    monitor_frames: int
    monitor_atoms: int
    eval_sets: list[tuple[int, int]]  # (frames, atoms per frame)

    def eval_atoms(self, max_frames: int | None) -> int:
        return sum(min(n, max_frames) * a if max_frames else n * a for n, a in self.eval_sets)

    def eval_frames(self, max_frames: int | None) -> int:
        return sum(min(n, max_frames) if max_frames else n for n, _ in self.eval_sets)


def dataset(manifest_path: Path) -> Data:
    sets = json.loads(manifest_path.read_text())["sets"]
    by_role = defaultdict(list)
    for meta in sets.values():
        by_role[meta["role"]].append((int(meta["frames"]), int(meta["atoms_per_frame"])))
    train = by_role["train"][0]
    monitor = by_role["monitor"][0] if by_role["monitor"] else (0, train[1])
    evaluated = [s for role, items in by_role.items() if role not in ("train", "monitor") for s in items]
    return Data(train_atoms=train[1], monitor_frames=monitor[0], monitor_atoms=monitor[1], eval_sets=evaluated)


def _model_class(run: str) -> str | None:
    if run.startswith(("independent_seed", "warm_snapshots")):
        return "full"
    if run.startswith("mhc"):
        return "mhc"
    if run.startswith("frozen_probe"):
        return "probe"
    return None


def measure(records: list[dict]) -> dict:
    """Rates and per-step overheads of a smoke run."""
    per_sample: dict[str, list[tuple[int, float]]] = defaultdict(list)
    shallow_heads = None
    atoms: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # kind -> [seconds, atoms]
    steps: list[dict] = []
    segment: list[dict] = []
    groups: dict[str, float] = {}
    setup = 0.0
    for record in records:
        kind = record.get("kind")
        if kind == "mace_epoch" and record.get("samples"):
            cls = _model_class(record.get("run") or "")
            if cls:
                per_sample[cls].append((record.get("epoch") or 0, record["seconds"] / record["samples"]))
        elif kind == "shallow_epoch" and record.get("samples"):
            per_sample["shallow"].append((record["epoch"], record["seconds"] / record["samples"]))
            shallow_heads = record.get("heads", shallow_heads)
        elif kind in ("llpr_frames", "eval_member", "eval_graphs") and record.get("atoms"):
            atoms[kind][0] += record["seconds"]
            atoms[kind][1] += record["atoms"]
        if kind == "step":
            steps.append({**record, "records": segment})
            segment = []
        elif kind == "group":
            groups[record["group"]] = record["seconds"]
        elif kind == "setup":
            setup = record["seconds"]
        else:
            segment.append(record)
    rates = {}
    for cls, items in per_sample.items():
        warm = [v for epoch, v in items if epoch > 0] or [v for _, v in items]
        rates[cls] = float(np.median(warm))
    rates.setdefault("probe", rates.get("mhc"))
    rates.setdefault("mhc", rates.get("probe"))
    per_atom = {kind: (s / n if n else None) for kind, (s, n) in atoms.items()}
    overhead: dict[str, dict[str, float]] = defaultdict(dict)
    for step in steps:
        phase, _, group = step["step"].partition(" ")
        inner = sum(r["seconds"] for r in step["records"]
                    if r.get("kind") in ("mace_epoch", "shallow_epoch", "llpr_frames", "eval_member",
                                         "eval_graphs", "bootstrap"))
        overhead[group][phase] = max(0.0, step["seconds"] - inner)
        if phase == "evaluate":
            overhead[group]["bootstrap"] = sum(r["seconds"] for r in step["records"] if r.get("kind") == "bootstrap")
        overhead[group][f"{phase}_seconds"] = step["seconds"]
    for group, seconds in groups.items():
        known = overhead[group].get("train_seconds", 0.0) + overhead[group].get("evaluate_seconds", 0.0)
        overhead[group]["upload"] = max(0.0, seconds - known)
    return {"per_sample": rates, "shallow_heads": shallow_heads, "per_atom": per_atom,
            "overhead": dict(overhead), "setup": setup}


TRAIN_JOBS = {"committee": 3, "snapshots": 1, "mhc": 1, "frozen-probe": 1, "shallow-nll": 1}


def project(measured: dict, smoke: Settings, target: Settings, data: Data) -> dict[str, dict[str, float]]:
    """Projected seconds per arm group and part: train, llpr, evaluate, other."""
    rate = measured["per_sample"]
    per_atom = measured["per_atom"]
    llpr_rate = per_atom.get("llpr_frames") or 0.0
    member_rate = per_atom.get("eval_member") or 0.0
    graph_rate = per_atom.get("eval_graphs") or 0.0
    eval_atoms = data.eval_atoms(target.max_frames_per_set)
    smoke_eval_atoms = max(1, data.eval_atoms(smoke.max_frames_per_set))
    cal_frames = target.cal_frames or data.monitor_frames
    shallow_scale = target.shallow_heads / (measured["shallow_heads"] or smoke.shallow_heads)
    rows: dict[str, dict[str, float]] = {}
    for group in target.groups:
        arms = group.split(",")
        train = llpr = forces = 0.0
        evaluations = graph_builds = 0
        for arm in arms:
            if arm == "committee":
                train += 3 * target.epochs * target.labels * (rate.get("full") or 0.0)
                evaluations, graph_builds = evaluations + 3, graph_builds + 1
            elif arm == "snapshots":
                snapshot_epochs = snapshot_schedule(target.epochs, target.snapshot_count,
                                                    target.snapshot_burn_in_cycles,
                                                    target.snapshot_cycle_epochs).total_epochs
                train += snapshot_epochs * target.labels * (rate.get("full") or 0.0)
                evaluations, graph_builds = evaluations + target.snapshot_count, graph_builds + 1
            elif arm == "mhc":
                train += target.mhc_epochs * target.labels * (rate.get("mhc") or 0.0)
                evaluations, graph_builds = evaluations + target.mhc_heads, graph_builds + 1
            elif arm == "frozen-probe":
                train += target.probe_epochs * target.labels * (rate.get("probe") or 0.0)
                evaluations, graph_builds = evaluations + 1, graph_builds + 1
            elif arm == "shallow-nll":
                train += target.shallow_epochs * target.labels * (rate.get("shallow") or 0.0) * shallow_scale
                evaluations, graph_builds = evaluations + target.shallow_heads, graph_builds + 1
            elif arm == "llpr":
                llpr += (target.llpr_train_frames * data.train_atoms + cal_frames * data.monitor_atoms
                         + eval_atoms) * llpr_rate
            elif arm == "zero-shot":
                evaluations, graph_builds = evaluations + 1, graph_builds + 1
        forces = eval_atoms * (evaluations * member_rate + graph_builds * graph_rate)
        measured_overhead = measured["overhead"].get(group, {})
        jobs = sum(TRAIN_JOBS.get(arm, 0) for arm in arms)
        # MACE re-reads and re-evaluates the training set at the end of every job.
        train_extra = jobs * max(0, target.labels - smoke.labels) * data.train_atoms * (member_rate + graph_rate)
        bootstrap = (measured_overhead.get("bootstrap", 0.0) * target.n_boot / max(1, smoke.n_boot)
                     * eval_atoms / smoke_eval_atoms)
        other = (measured_overhead.get("train", 0.0) + train_extra + measured_overhead.get("evaluate", 0.0)
                 + bootstrap + 1.5 * measured_overhead.get("upload", 0.0))
        rows[group] = {"train": train, "llpr": llpr, "evaluate": forces, "other": other}
    return rows


def report(measured: dict, smoke: Settings, target: Settings, data: Data, target_path: Path,
           deadline_minutes: float | None) -> str:
    rows = project(measured, smoke, target, data)
    rate = measured["per_sample"]
    per_atom = measured["per_atom"]
    lines = [f"Projection for {target_path} ({target.labels} labels) from the smoke run's timings",
             "measured per training frame and epoch: " + ", ".join(
                 f"{name} {1000 * value:.1f} ms" for name, value in sorted(rate.items()) if value),
             "measured per atom: " + ", ".join(
                 f"{name} {1e3 * value:.2f} ms" for name, value in sorted(per_atom.items()) if value),
             "",
             f"{'arm group':22s} {'train':>7s} {'LLPR':>7s} {'eval':>7s} {'other':>7s} {'total':>7s}  hours"]
    total = 0.0
    for group, parts in rows.items():
        group_total = sum(parts.values())
        total += group_total
        lines.append(f"{group:22s} " + " ".join(f"{parts[k] / 3600:7.2f}" for k in ("train", "llpr", "evaluate", "other"))
                     + f" {group_total / 3600:7.2f}")
    fixed = measured["setup"] + FINAL_SECONDS
    total += fixed
    total *= 1 + CONTINGENCY
    lines.append(f"{'setup and final upload':22s} {'':31s} {fixed / 3600:7.2f}")
    lines.append(f"{'TOTAL (+' + str(int(100 * CONTINGENCY)) + '% margin)':22s} {'':31s} {total / 3600:7.2f}")
    if deadline_minutes:
        verdict = "fits" if total <= 60 * deadline_minutes else "DOES NOT FIT"
        lines.append(f"deadline {deadline_minutes:.0f} min ({deadline_minutes / 60:.1f} h): {verdict}, "
                     f"margin {(60 * deadline_minutes - total) / 3600:+.2f} h")
    missing = [name for name in ("full", "mhc", "probe", "shallow") if not rate.get(name)]
    missing += [name for name in ("llpr_frames", "eval_member") if not per_atom.get(name)]
    if missing:
        lines.append(f"WARNING: not measured in the smoke run: {', '.join(missing)} (counted as zero)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, type=Path, help="smoke run log with [timing] lines")
    ap.add_argument("--measured", required=True, type=Path, help="node file the smoke run used")
    ap.add_argument("--target", required=True, type=Path, help="node file to project")
    ap.add_argument("--dataset-dir", required=True, type=Path, help="folder with the dataset manifest.json")
    ap.add_argument("--deadline-minutes", type=float, default=None)
    args = ap.parse_args()
    records = timing.parse(args.log.read_text(errors="ignore").splitlines())
    print(report(measure(records), settings(read_env(args.measured)), settings(read_env(args.target)),
                 dataset(args.dataset_dir / "manifest.json"), args.target, args.deadline_minutes))


if __name__ == "__main__":
    main()
