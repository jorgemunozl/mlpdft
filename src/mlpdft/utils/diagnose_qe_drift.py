#!/usr/bin/env python3
"""Diagnose energy drift in a QE ``pw.x`` MD run.

Parses, per ionic step: total (potential) energy, ionic kinetic energy,
temperature, the printed constant of motion, number of SCF iterations,
estimated SCF accuracy, Andersen collisions, and atomic positions. From those
it reports whether a drift in the potential energy comes with

* a temperature drift (thermostat problem),
* worsening SCF convergence (numerical problem), or
* a structural change (physical relaxation),

and saves the per-step series to a ``.npz`` for plotting. It also reports the
thermostat settings and the short-lag mean-squared-displacement exponent:
about 2 for physical (ballistic) motion, about 1 when every velocity is redrawn
at every step, as with an Andersen collision frequency of 1/timestep.

Fixed-cell MD only (the ``Ekin + Etot (const)`` format); ``vc-md`` output is
not supported.

Usage
-----
    PYTHONPATH=src python3 src/mlpdft/utils/diagnose_qe_drift.py \
        --out dataset/LIFINTERFACE_KJPAW_V2/LIFINTERFACE_KJPAW_V2.out \
        --label interface_V2 --save outputs/drift/interface_V2.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RY_TO_EV = 13.605693122994
NUM = r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)"


def parse(path: Path) -> dict[str, np.ndarray]:
    """One record per completed MD step (an SCF block followed by an MD update)."""
    etot, ekin, temp, const, niter, acc, coll = [], [], [], [], [], [], []
    positions: list[np.ndarray] = []
    species: list[str] = []
    cur: dict = {}
    reading_pos = False
    pos_buf: list[list[float]] = []
    sp_buf: list[str] = []
    last_acc = np.nan
    with path.open(errors="ignore") as fh:
        for line in fh:
            if reading_pos:
                parts = line.split()
                if len(parts) == 4 and parts[0].isalpha():
                    sp_buf.append(parts[0])
                    pos_buf.append([float(v) for v in parts[1:]])
                    continue
                reading_pos = False
                cur["pos"] = np.asarray(pos_buf)
                if not species:
                    species = sp_buf
            if "estimated scf accuracy" in line:
                m = re.search(NUM + r"\s+Ry", line)
                if m:
                    last_acc = float(m.group(1))
            elif line.startswith("!") and "total energy" in line:
                cur = {"etot": float(re.search(NUM, line).group(1)) * RY_TO_EV,
                       "acc": last_acc * RY_TO_EV}
            elif "convergence has been achieved in" in line:
                cur["niter"] = int(re.search(r"in\s+(\d+)\s+iter", line).group(1))
            elif "Andersen thermostat:" in line and "collisions" in line:
                cur["coll"] = int(re.search(r"(\d+)\s+collisions", line).group(1))
            elif line.startswith("ATOMIC_POSITIONS"):
                reading_pos, pos_buf, sp_buf = True, [], []
            elif "kinetic energy (Ekin)" in line:
                cur["ekin"] = float(re.search(NUM, line).group(1)) * RY_TO_EV
            elif line.strip().startswith("temperature") and "K" in line and "=" in line:
                cur["temp"] = float(re.search(NUM, line).group(1))
            elif "Ekin + Etot (const)" in line:
                cur["const"] = float(re.search(NUM, line).group(1)) * RY_TO_EV
                if {"etot", "ekin", "temp", "pos"} <= cur.keys():
                    etot.append(cur["etot"]); ekin.append(cur["ekin"])
                    temp.append(cur["temp"]); const.append(cur["const"])
                    niter.append(cur.get("niter", -1)); acc.append(cur["acc"])
                    coll.append(cur.get("coll", -1)); positions.append(cur["pos"])
                cur = {}
    return {
        "etot": np.asarray(etot), "ekin": np.asarray(ekin), "temp": np.asarray(temp),
        "const": np.asarray(const), "niter": np.asarray(niter), "acc": np.asarray(acc),
        "coll": np.asarray(coll), "pos": np.asarray(positions),
        "species": np.asarray(species),
    }


def cell_from_out(path: Path) -> np.ndarray:
    """Cell vectors in angstrom from the header (fixed cell, NVT)."""
    alat, axes = None, []
    with path.open(errors="ignore") as fh:
        for line in fh:
            if "lattice parameter (alat)" in line:
                alat = float(re.search(NUM, line).group(1)) * 0.529177210903
            elif re.match(r"\s+a\(\d\) = \(", line):
                axes.append([float(v) for v in re.findall(NUM, line)])
                if len(axes) == 3:
                    break
    return np.asarray(axes) * alat


def unwrap(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Remove periodic jumps (orthorhombic cell)."""
    lengths = np.diag(cell)
    d = np.diff(pos, axis=0)
    d -= np.round(d / lengths) * lengths
    return np.concatenate([pos[:1], pos[:1] + np.cumsum(d, axis=0)])


def thermostat_info(path: Path) -> dict[str, str | None]:
    """Thermostat lines from the first MD header."""
    info: dict[str, str | None] = {"thermostat": None, "collision_frequency": None}
    with path.open(errors="ignore") as fh:
        for line in fh:
            if "temperature is controlled by" in line and info["thermostat"] is None:
                info["thermostat"] = line.split("controlled by", 1)[1].strip()
            elif "Collision frequency" in line and info["collision_frequency"] is None:
                info["collision_frequency"] = line.split("=", 1)[1].strip()
            elif "Entering Dynamics" in line:
                break
    return info


def msd_exponent(data: dict, cell: np.ndarray, burn_in: int,
                 lags: tuple[int, ...] = (1, 2, 5, 10, 20)) -> dict[str, float]:
    """Log-log slope of the MSD over short lags, per species, centre of mass removed."""
    up = unwrap(data["pos"], cell)[burn_in:]
    up = up - up.mean(axis=1, keepdims=True)
    out: dict[str, float] = {}
    for s in np.unique(data["species"]):
        m = data["species"] == s
        msd = [((up[k:, m] - up[:-k, m]) ** 2).sum(-1).mean() for k in lags]
        out[str(s)] = float(np.polyfit(np.log(lags), np.log(msd), 1)[0])
    return out


def slope_per_1000(y: np.ndarray) -> float:
    t = np.arange(len(y))
    return float(np.polyfit(t, y, 1)[0] * 1000)


def structural(data: dict, cell: np.ndarray) -> dict[str, np.ndarray]:
    pos = data["pos"]
    sp = data["species"]
    up = unwrap(pos, cell)
    disp = up - up[0]
    out: dict[str, np.ndarray] = {}
    for s in np.unique(sp):
        m = sp == s
        out[f"msd_{s}"] = (disp[:, m] ** 2).sum(-1).mean(-1)
        out[f"msd_z_{s}"] = (disp[:, m, 2] ** 2).mean(-1)
    # F layer: spread of F z coordinates about the F centre (layer thickness),
    # computed in unwrapped coordinates.
    f = sp == "F"
    if f.any():
        zf = up[:, f, 2]
        out["F_z_std"] = zf.std(-1)
        out["F_z_mean"] = zf.mean(-1)
    # Li-F nearest-neighbour coordination with minimum image (cutoff 2.4 A).
    li = sp == "Li"
    if f.any() and li.any():
        L = np.diag(cell)
        cn = []
        for p in pos:
            d = p[li][:, None, :] - p[f][None, :, :]
            d -= np.round(d / L) * L
            r = np.linalg.norm(d, axis=-1)
            cn.append((r < 2.4).sum(0).mean())
        out["cn_F_Li"] = np.asarray(cn)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--burn-in", type=int, default=0)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    data = parse(args.out)
    if len(data["etot"]) == 0:
        raise SystemExit(f"no fixed-cell MD steps found in {args.out} (vc-md is not supported)")
    cell = cell_from_out(args.out)
    st = structural(data, cell)
    b = args.burn_in
    n = len(data["etot"])
    rep = {
        "label": args.label or args.out.stem, "steps": n, "burn_in": b,
        "atoms": int(data["pos"].shape[1]),
        **thermostat_info(args.out),
        "collisions_median": float(np.median(data["coll"])),
        "msd_short_lag_exponent": msd_exponent(data, cell, b),
        "T_mean": float(data["temp"][b:].mean()), "T_std": float(data["temp"][b:].std()),
        "slope_per_1000_steps": {
            "Etot_eV": slope_per_1000(data["etot"][b:]),
            "const_eV": slope_per_1000(data["const"][b:]),
            "T_K": slope_per_1000(data["temp"][b:]),
            "scf_iter": slope_per_1000(data["niter"][b:].astype(float)),
            **{k: slope_per_1000(v[b:]) for k, v in st.items()},
        },
        "Etot_std_eV": float(data["etot"][b:].std()),
        "scf_iter_mean": float(data["niter"][b:].mean()),
        "scf_acc_max_eV": float(np.nanmax(data["acc"][b:])),
        "Etot_window_means_eV": [float(w.mean()) for w in np.array_split(data["etot"][b:], 5)],
        "corr_Etot_vs": {
            k: float(np.corrcoef(data["etot"][b:], v[b:])[0, 1]) if np.std(v[b:]) > 0 else None
            for k, v in st.items()
        },
    }
    print(json.dumps(rep, indent=2))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, cell=cell, **{k: v for k, v in data.items()}, **st)


if __name__ == "__main__":
    main()
