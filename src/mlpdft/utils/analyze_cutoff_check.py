#!/usr/bin/env python3
"""Compare forces, relative energies and stress across the cutoff-check single points.

Reads the ``outputs/*.out`` files written by stage 2 of the 2026-11 campaign
(``f<i>_step<s>_ec<cutoff>_k<k>.out``, see ``make_qe_campaign.py``) and reports,
per environment and setting, the deviation from the reference (80 Ry, 3x3x3):

* force RMSE and max |dF| over all atoms and components (meV/A);
* relative-energy error: energies are referenced to the mean over frames, so a
  constant offset between cutoffs cancels (meV/atom);
* pressure difference (kbar).

The k-point check compares 40 Ry 3x3x3 against 40 Ry 4x4x4 on the same frames.

With ``--dataset-root``, the 40 Ry 3x3x3 single points are also compared with
the source trajectory's own forces at the same SCF step. Agreement within
1 meV/A shows that the new jobs reproduce the settings of the existing data.

Usage
-----
    python3 src/mlpdft/utils/analyze_cutoff_check.py \
        --stage-dir assets/quantum_expresso/campaign-2026-11/stage2_cutoff \
        --dataset-root /home/jorge/project/mlpdft/dataset --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RY_TO_EV = 13.605693122994
RY_BOHR_TO_EV_A = RY_TO_EV / 0.529177210903
NAME_RE = re.compile(r"f(?P<frame>\d+)_step(?P<step>\d+)_ec(?P<ec>\d+)_k(?P<k>\d+)")
FORCE_TOLERANCE_MEV_A = 10.0  # well below the ~30 meV/A force error of the fine-tuned models
REPRODUCTION_TOLERANCE_MEV_A = 1.0


def trajectory_reproduction(env: str, recs: dict, dataset_root: Path) -> dict | None:
    """40 Ry 3x3x3 single points against the source run's forces at the same SCF step."""
    from estimate_qe_correlation import parse_forces
    from make_qe_campaign import ENVS

    if env not in ENVS or not recs:
        return None
    source = ENVS[env]["source"]
    traj = parse_forces(dataset_root / source / f"{source}.out") * RY_BOHR_TO_EV_A
    pairs = [(rec["forces"], traj[rec["step"]]) for rec in recs.values() if rec["step"] < len(traj)]
    if not pairs:
        return None
    df = np.concatenate([(a - b).ravel() for a, b in pairs])
    rmse = float(np.sqrt(np.mean(df ** 2)) * 1000)
    return {"source": source, "frames": len(pairs), "force_rmse_meV_A": rmse,
            "force_max_abs_meV_A": float(np.max(np.abs(df)) * 1000),
            "reproduces_source": rmse <= REPRODUCTION_TOLERANCE_MEV_A}


def read_scf(path: Path) -> dict | None:
    """Energy (eV), forces (eV/A) and pressure (kbar) of a finished pw.x single point."""
    text = path.read_text(errors="ignore")
    if "JOB DONE" not in text or "convergence has been achieved" not in text:
        return None
    energy = float(re.search(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text).group(1)) * RY_TO_EV
    block = text.split("Forces acting on atoms", 1)[1].split("\n\n", 2)[1]
    forces = [[float(v) for v in line.split("=", 1)[1].split()[:3]]
              for line in block.splitlines() if "force =" in line]
    pressure = re.search(r"P=\s*(-?\d+\.\d+)", text)
    return {"energy": energy, "forces": np.asarray(forces) * RY_BOHR_TO_EV_A,
            "pressure": float(pressure.group(1)) if pressure else None}


def compare(results: dict, settings: list[tuple[int, int]], ref: tuple[int, int]) -> dict:
    """results[(ec, k)][frame] -> record. Deviation of each setting from ``ref`` on shared frames."""
    rows = {}
    for setting in settings:
        if setting == ref:
            continue
        frames = sorted(set(results.get(setting, {})) & set(results.get(ref, {})))
        if not frames:
            continue
        df = np.concatenate([(results[setting][f]["forces"] - results[ref][f]["forces"]).ravel()
                             for f in frames])
        n_atoms = len(results[ref][frames[0]]["forces"])
        e = np.array([results[setting][f]["energy"] for f in frames])
        e_ref = np.array([results[ref][f]["energy"] for f in frames])
        de_rel = ((e - e.mean()) - (e_ref - e_ref.mean())) / n_atoms if len(frames) > 1 else None
        dp = [results[setting][f]["pressure"] - results[ref][f]["pressure"] for f in frames
              if results[setting][f]["pressure"] is not None and results[ref][f]["pressure"] is not None]
        rmse = float(np.sqrt(np.mean(df ** 2)) * 1000)
        rows[f"ec{setting[0]}_k{setting[1]}"] = {
            "reference": f"ec{ref[0]}_k{ref[1]}",
            "frames": len(frames),
            "force_rmse_meV_A": rmse,
            "force_max_abs_meV_A": float(np.max(np.abs(df)) * 1000),
            "relative_energy_max_abs_meV_atom": (float(np.max(np.abs(de_rel)) * 1000)
                                                 if de_rel is not None else None),
            "pressure_diff_kbar_mean": float(np.mean(dp)) if dp else None,
            "within_force_tolerance": rmse <= FORCE_TOLERANCE_MEV_A,
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="also compare 40 Ry single points with the source trajectories")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {}
    for env_dir in sorted(p for p in args.stage_dir.iterdir() if (p / "outputs").is_dir()):
        results: dict = defaultdict(dict)
        missing = []
        for out in sorted((env_dir / "outputs").glob("*.out")):
            m = NAME_RE.search(out.stem)
            if not m:
                continue
            rec = read_scf(out)
            if rec is None:
                missing.append(out.name)
                continue
            rec["step"] = int(m["step"])
            results[(int(m["ec"]), int(m["k"]))][int(m["frame"])] = rec
        settings = sorted(results)
        cutoff_rows = compare(results, [s for s in settings if s[1] == 3], (80, 3))
        k_rows = compare(results, [(40, 3)], (40, 4))
        repro = (trajectory_reproduction(env_dir.name, results.get((40, 3), {}), args.dataset_root)
                 if args.dataset_root else None)
        report[env_dir.name] = {"cutoff_vs_80Ry": cutoff_rows, "kpoints_vs_4x4x4": k_rows,
                                "trajectory_reproduction": repro, "unfinished": missing}
        print(f"== {env_dir.name}" + (f"  (unfinished: {len(missing)})" if missing else ""))
        if repro:
            print(f"  40 Ry vs source {repro['source']} at the same steps: force RMSE "
                  f"{repro['force_rmse_meV_A']:.3f} meV/A, max {repro['force_max_abs_meV_A']:.3f}"
                  f"{'' if repro['reproduces_source'] else '  <-- settings do not reproduce the source'}")
        def fmt(value: float | None, spec: str) -> str:
            return "n/a" if value is None else format(value, spec)

        for label, row in {**cutoff_rows, **{f"{k} vs k4": v for k, v in k_rows.items()}}.items():
            print(f"  {label:14s} vs {row['reference']}: force RMSE {row['force_rmse_meV_A']:7.2f} meV/A, "
                  f"max {row['force_max_abs_meV_A']:7.1f}, rel. energy "
                  f"{fmt(row['relative_energy_max_abs_meV_atom'], '6.2f')} meV/atom, "
                  f"dP {fmt(row['pressure_diff_kbar_mean'], '7.1f')} kbar"
                  f"{'' if row['within_force_tolerance'] else '  <-- above tolerance'}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
