"""Seed-pooled, paired intervals and the ADR-0009 verdicts across design-C runs.

Each design-C run holds one seed set and writes, through ``uq_evaluate``,
``<run>/evaluation/metrics.json`` (set roles) and
``<run>/evaluation/<arm>/<set>.npz`` (per-atom ``err`` and ``sigma``, per-atom
``frame``, per-frame ``run`` and ``scf_step``). This module reads the runs of
one label rung and applies ``assets/research/decisions/ADR-0009``:

* **cells:** a metric is computed per data unit (a replica, or a time block of
  one trajectory) and per seed set; a comparison uses the per-cell difference
  between two arms on the same frames;
* **interval:** a two-way t interval over data units x seed sets with variance
  (MS_D + MS_S) / (a b) and Satterthwaite degrees of freedom; with one data
  unit, a t interval over the seed sets only, labelled conditional;
* **verdicts:** rules 2 (accuracy), 3 (ranking), 4 (calibration), 5 (shift
  detection), 7 (seed consistency), 9 (Holm on the primary family) and 10
  (leakage). Cost (rule 6) and label-rung trends (rule 8) are not handled here.

Usage
-----
    python -m mlpdft.uq_inference --runs seed1/ seed2/ seed3/ --out inference.json \
        --block-steps shift_fdoped=1000 --block-steps shift_li_metal=1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from mlpdft import uq_metrics as um

REFERENCE_ARM = "committee"
UNCERTAINTY_ARMS = ("snapshots", "mhc", "shallow-nll", "llpr")
ACCURACY_MARGIN = 1.25
CONFORMAL_TARGET = 0.9
CONFORMAL_BAND = (0.85, 0.95)
MIN_UNITS = 3


# --------------------------------------------------------------------------- intervals

def two_way_interval(cells: np.ndarray, level: float = 0.95) -> dict:
    """Mean of an (a data units, b seed sets) grid and its ADR-0009 t interval.

    The variance of the mean is (MS_D + MS_S) / (a b), the conservative form of the
    two-way random-effects decomposition without replication. With a = 1 (or b = 1)
    it reduces to a t interval over the other factor.
    """
    cells = np.asarray(cells, dtype=float)
    if cells.ndim != 2 or not np.isfinite(cells).all():
        raise ValueError("cells must be a complete, finite (units, seeds) grid")
    a, b = cells.shape
    mean = float(cells.mean())
    terms = []  # (mean square, its degrees of freedom)
    if a > 1:
        terms.append((b * np.var(cells.mean(axis=1), ddof=1), a - 1))
    if b > 1:
        terms.append((a * np.var(cells.mean(axis=0), ddof=1), b - 1))
    if not terms:
        raise ValueError("an interval needs at least two cells along one factor")
    total = sum(ms for ms, _ in terms)
    se = float(np.sqrt(total / (a * b)))
    df = float(total ** 2 / sum(ms ** 2 / d for ms, d in terms)) if total > 0 else float(min(d for _, d in terms))
    half = float(stats.t.ppf(0.5 + level / 2, df) * se)
    t = mean / se if se > 0 else (np.inf if mean != 0 else 0.0)
    p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else 0.0
    per_seed = cells.mean(axis=0)
    return {"mean": mean, "low": mean - half, "high": mean + half, "se": se, "df": df, "p": p,
            "units": int(a), "seeds": int(b), "per_seed": [float(v) for v in per_seed],
            "conditional": a == 1}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment; returns the adjusted p and the rejection per key."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out, still = len(order), 0.0, {}, True
    for rank, key in enumerate(order):
        adjusted = min(1.0, max(running, (m - rank) * pvalues[key]))
        running = adjusted
        still = still and pvalues[key] <= alpha / (m - rank)
        out[key] = {"p": pvalues[key], "p_holm": adjusted, "reject": bool(still)}
    return out


# --------------------------------------------------------------------------- verdicts

def compare_verdict(iv: dict, higher_is_better: bool) -> str:
    """Rule 3/4 verdict on a paired difference (arm minus reference)."""
    sign = 1 if higher_is_better else -1
    lo, hi = sorted((sign * iv["low"], sign * iv["high"]))
    seeds = [sign * v for v in iv["per_seed"]]
    if lo > 0:
        return "better" if all(v > 0 for v in seeds) else "inconsistent across seeds"
    if hi < 0:
        return "worse" if all(v < 0 for v in seeds) else "inconsistent across seeds"
    return "not separated"


def accuracy_verdict(iv: dict, margin: float = ACCURACY_MARGIN) -> str:
    """Rule 2 verdict on the log ratio of force RMSE (arm over reference)."""
    cut = np.log(margin)
    if iv["high"] <= cut:
        return "comparable" if all(v <= cut for v in iv["per_seed"]) else "inconsistent across seeds"
    if iv["low"] > cut:
        return "worse" if all(v > cut for v in iv["per_seed"]) else "inconsistent across seeds"
    return "inconclusive"


def above_verdict(iv: dict, threshold: float, yes: str, no: str) -> str:
    """Single-arm check that the whole interval lies above a threshold (rules 3 and 5)."""
    if iv["low"] > threshold:
        return yes if all(v > threshold for v in iv["per_seed"]) else "inconsistent across seeds"
    return no


# --------------------------------------------------------------------------- data

def load_runs(run_dirs: list[Path]) -> tuple[dict, dict]:
    """Set roles (identical in every run) and {arm: {set: [per-seed arrays]}}."""
    roles = None
    data: dict[str, dict[str, list[dict]]] = {}
    for run in run_dirs:
        ev = Path(run) / "evaluation"
        report = json.loads((ev / "metrics.json").read_text())
        run_roles = {s: row["role"] for arm in report["arms"].values() for s, row in arm["sets"].items()}
        if roles is None:
            roles = run_roles
        elif run_roles != roles:
            raise ValueError(f"{run}: set roles differ from the first run")
        for arm in report["arms"]:
            for name in run_roles:
                path = ev / arm / f"{name}.npz"
                if not path.is_file():
                    continue
                with np.load(path, allow_pickle=False) as z:
                    data.setdefault(arm, {}).setdefault(name, []).append(
                        {k: z[k] for k in ("err", "sigma", "frame", "run", "scf_step")})
    n = len(run_dirs)
    for arm, sets in data.items():
        for name, seeds in sets.items():
            if len(seeds) != n:
                raise ValueError(f"{arm}/{name}: found in {len(seeds)} of {n} runs")
    return roles or {}, data


def data_units(d: dict, block_steps: int | None) -> np.ndarray:
    """Per-frame data-unit labels: replicas if the set has several runs, else time blocks."""
    runs = d["run"].astype(str)
    if len(set(runs)) > 1:
        return np.unique(runs, return_inverse=True)[1]
    if block_steps:
        blocks = np.asarray(d["scf_step"]) // block_steps
        return np.unique(blocks, return_inverse=True)[1]
    return np.zeros(len(runs), dtype=int)


def _frames_of(d: dict, frames: np.ndarray) -> dict:
    keep = np.isin(d["frame"], frames)
    renumber = {f: i for i, f in enumerate(frames)}
    return {"err": d["err"][keep], "sigma": d["sigma"][keep],
            "frame": np.array([renumber[f] for f in d["frame"][keep]])}


def cell_metrics(d: dict, alpha: float, q: float, rmse_cal: float) -> dict:
    e, s, f = d["err"], d["sigma"], d["frame"]
    row = {"log_rmse": float(np.log(np.sqrt(np.mean(e ** 2))))}
    if np.isnan(s).all():
        return row
    fe, fs = um.per_frame(e, s, f)
    ae, an = um.per_atom(e, s)
    row.update({"rho_frame": um.spearman(fe, fs), "rho_atom": um.spearman(ae, an),
                "miscal_cal": um.miscalibration_area(e, alpha * s),
                "nll_cal": um.gaussian_nll(e, alpha * s),
                "nll_cal_minus_const": um.gaussian_nll(e, alpha * s) - um.gaussian_nll(e, np.full_like(s, rmse_cal)),
                "coverage_1sigma_cal": um.coverage(e, alpha * s, 1.0),
                "coverage_2sigma_cal": um.coverage(e, alpha * s, 2.0),
                "conformal_coverage": um.coverage(e, s, q)})
    return row


def grid(arm_sets: dict, name: str, calib: dict, block_steps: int | None) -> tuple[dict, np.ndarray]:
    """{metric: (units, seeds) array} for one arm on one set, and the units of seed 0."""
    seeds = arm_sets[name]
    units0 = data_units(seeds[0], block_steps)
    labels = np.unique(units0)
    if len(labels) < MIN_UNITS:
        labels = np.array([0])
        units0 = np.zeros_like(units0)
    cells: dict[str, list[list[float]]] = {}
    for si, d in enumerate(seeds):
        alpha, q, rmse_cal = calib[si]
        units = units0 if len(labels) > 1 else np.zeros(len(d["run"]), dtype=int)
        for ui, u in enumerate(labels):
            row = cell_metrics(_frames_of(d, np.flatnonzero(units == u)), alpha, q, rmse_cal)
            for k, v in row.items():
                cells.setdefault(k, [[np.nan] * len(seeds) for _ in labels])[ui][si] = v
    return {k: np.array(v) for k, v in cells.items()}, labels


def calibration_constants(arm_sets: dict, cal_name: str | None) -> list[tuple[float, float, float]]:
    seeds = arm_sets[next(iter(arm_sets))]
    if cal_name is None or cal_name not in arm_sets:
        return [(1.0, 1.0, np.nan)] * len(seeds)
    out = []
    for d in arm_sets[cal_name]:
        e, s = d["err"], d["sigma"]
        if np.isnan(s).all():
            out.append((1.0, 1.0, float(np.sqrt(np.mean(e ** 2)))))
        else:
            out.append((um.nll_scale(e, s), um.conformal_multiplier(e, s, CONFORMAL_TARGET),
                        float(np.sqrt(np.mean(e ** 2)))))
    return out


# --------------------------------------------------------------------------- report

def infer(roles: dict, data: dict, block_steps: dict[str, int] | None = None) -> dict:
    block_steps = block_steps or {}
    cal = next((n for n, r in roles.items() if r == "calibration"), None)
    id_test = next((n for n, r in roles.items() if r == "test"), None)
    note = "replica test set"
    if id_test is None and "test_timeblock" in roles:
        id_test, note = "test_timeblock", "time-block holdout: pipeline check only (ADR-0009 rule 10)"
    if REFERENCE_ARM not in data:
        raise ValueError(f"reference arm {REFERENCE_ARM!r} is missing")
    eval_sets = [n for n, r in roles.items() if r not in ("calibration", "train", "monitor")]
    grids, labels = {}, {}
    for arm, sets in data.items():
        calib = calibration_constants(sets, cal)
        for name in eval_sets:
            if name in sets:
                grids[(arm, name)], labels[(arm, name)] = grid(sets, name, calib, block_steps.get(name))
    report: dict = {"calibration_set": cal, "id_test": id_test, "id_test_note": note,
                    "rules": "assets/research/decisions/ADR-0009-design-c-decision-rules.md",
                    "calibrated": cal is not None, "sets": {}, "primary": {}, "leakage": {}}
    primary_p = {}
    for name in eval_sets:
        ref = grids.get((REFERENCE_ARM, name))
        if ref is None:
            continue
        set_out = {"units": len(labels[(REFERENCE_ARM, name)]), "arms": {}}
        for arm in data:
            g = grids.get((arm, name))
            if g is None or g["log_rmse"].shape != ref["log_rmse"].shape:
                continue
            row: dict = {"force_rmse": float(np.exp(g["log_rmse"]).mean())}
            if arm != REFERENCE_ARM:
                iv = two_way_interval(g["log_rmse"] - ref["log_rmse"])
                row["rmse_ratio"] = {"mean": float(np.exp(iv["mean"])), "low": float(np.exp(iv["low"])),
                                     "high": float(np.exp(iv["high"])), "verdict": accuracy_verdict(iv)}
            if "rho_frame" in g:
                for key in ("rho_frame", "rho_atom", "miscal_cal", "nll_cal", "conformal_coverage",
                            "coverage_1sigma_cal", "coverage_2sigma_cal"):
                    row[key] = two_way_interval(g[key])
                row["rho_frame"]["verdict"] = above_verdict(row["rho_frame"], 0.0, "ranks errors", "not shown")
                if cal is not None:
                    gap = two_way_interval(g["nll_cal_minus_const"])
                    in_band = CONFORMAL_BAND[0] <= row["conformal_coverage"]["mean"] <= CONFORMAL_BAND[1]
                    row["usable_error_bar"] = bool(gap["high"] < 0 and in_band)
                    row["nll_minus_constant_sigma"] = gap
                if arm != REFERENCE_ARM and "rho_frame" in ref:
                    for key, better_high in (("rho_frame", True), ("rho_atom", True), ("miscal_cal", False)):
                        iv = two_way_interval(g[key] - ref[key])
                        iv["verdict"] = compare_verdict(iv, better_high)
                        row[f"delta_{key}"] = iv
                        if name == id_test and roles.get(name) == "test" and arm in UNCERTAINTY_ARMS \
                                and key in ("rho_frame", "miscal_cal"):
                            primary_p[f"{arm}:{key}"] = iv["p"]
            set_out["arms"][arm] = row
        report["sets"][name] = set_out
    # Rule 5: AUROC of per-frame sigma, ID test replica (cell) against the whole shift set.
    if id_test is not None:
        for arm, sets in data.items():
            if id_test not in sets or np.isnan(sets[id_test][0]["sigma"]).all():
                continue
            for name in eval_sets:
                if name == id_test or name not in sets or name not in report["sets"] \
                        or roles[name] == "test_timeblock":
                    continue
                cells = []
                units0 = data_units(sets[id_test][0], None)
                for u in np.unique(units0):
                    row = []
                    for d_id, d_out in zip(sets[id_test], sets[name]):
                        sub = _frames_of(d_id, np.flatnonzero(units0 == u))
                        row.append(um.shift_auroc(sub["sigma"], sub["frame"], d_out["sigma"], d_out["frame"]))
                    cells.append(row)
                iv = two_way_interval(np.array(cells))
                iv["verdict"] = above_verdict(iv, 0.5, "detects the shift", "not shown")
                iv["conditional_on_shift_trajectory"] = True
                report["sets"][name]["arms"].setdefault(arm, {})["auroc_vs_id"] = iv
    # Rule 10: leakage, the same metric on test_timeblock minus the ID test, per seed set.
    if id_test is not None and id_test != "test_timeblock" and "test_timeblock" in roles:
        for arm in data:
            g_id, g_tb = grids.get((arm, id_test)), grids.get((arm, "test_timeblock"))
            if g_id is None or g_tb is None:
                continue
            out = {}
            for key in ("log_rmse", "rho_frame", "miscal_cal"):
                if key in g_id and key in g_tb:
                    out[key] = two_way_interval((g_tb[key].mean(axis=0) - g_id[key].mean(axis=0))[None, :])
            report["leakage"][arm] = out
    # Rule 9: Holm on the primary family; a primary better/worse verdict also needs Holm.
    if primary_p:
        adjusted = holm(primary_p)
        for key, res in adjusted.items():
            arm, metric = key.split(":")
            iv = report["sets"][id_test]["arms"][arm][f"delta_{metric}"]
            iv["p_holm"] = res["p_holm"]
            if not res["reject"] and iv["verdict"] in ("better", "worse"):
                iv["verdict"] = "not separated after Holm"
            report["primary"][key] = {"verdict": iv["verdict"], "p": res["p"], "p_holm": res["p_holm"]}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="one run directory per seed set")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-steps", action="append", default=[], metavar="SET=STEPS",
                    help="time-block length for a single-trajectory set (at least its force-envelope lag)")
    args = ap.parse_args()
    blocks = {k: int(v) for k, v in (item.split("=", 1) for item in args.block_steps)}
    roles, data = load_runs(args.runs)
    report = infer(roles, data, blocks)
    args.out.write_text(json.dumps(report, indent=2))
    for key, res in report["primary"].items():
        print(f"{key:28s} {res['verdict']:28s} p={res['p']:.3g} Holm p={res['p_holm']:.3g}")


if __name__ == "__main__":
    main()
