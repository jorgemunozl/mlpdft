#!/usr/bin/env python3
"""Curate a two-environment interim dataset from the existing Phase 1 audit.

No independence is implied by time-blocked splits or the provisional burn-in.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from build_interim_uq_dataset import FILES, evenly_spaced, verify_scf
from build_phase1_splits import build_split
from phase1_audit import frame_hash, parse_energy, parse_lattice, read_frames

SOURCES = {"bulk_LiF": "LIF64_KJPAW_NPT_V3", "F_doped_Li": "LIWITHF_V3"}
BURN_IN = {"bulk_LiF": 100, "F_doped_Li": 2400}
GUARD = 60
QUOTAS = {"train_pool": 500, "validation": 100, "calibration": 100, "test": 100}


def select(rows):
    chosen = []
    for env, source in SOURCES.items():
        src = [r for r in rows if r["environment"] == env and r["group"] == source
               and int(r["stride"]) == 3 and r["split"] in ("train_pool", "calibration", "test")]
        groups = {s: [r for r in src if r["split"] == s] for s in ("train_pool", "calibration", "test")}
        end_train = max(int(r["time_index"]) for r in groups["train_pool"])
        start_cal = min(int(r["time_index"]) for r in groups["calibration"])
        end_cal = max(int(r["time_index"]) for r in groups["calibration"])
        start_test = min(int(r["time_index"]) for r in groups["test"])
        mid = (start_cal + end_cal) // 2
        pools = {
            "train_pool": [r for r in groups["train_pool"] if BURN_IN[env] <= int(r["time_index"]) <= end_train-GUARD],
            "validation": [r for r in groups["calibration"] if start_cal+GUARD <= int(r["time_index"]) <= mid-GUARD],
            "calibration": [r for r in groups["calibration"] if mid+GUARD+1 <= int(r["time_index"]) <= end_cal-GUARD],
            "test": [r for r in groups["test"] if int(r["time_index"]) >= start_test+GUARD],
        }
        for split, pool in pools.items():
            for row in evenly_spaced(pool, QUOTAS[split]):
                chosen.append({**row, "audit_split": row["split"], "split": split})
    return chosen


def autocorrelation(values):
    """Mean component autocovariance, normalized; first-nonpositive truncation.

    Values are time x component. Nonstationarity and short windows make this
    diagnostic unreliable; samples are from one continuous stride-3 trajectory.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(axis=0, keepdims=True)
    n = len(x)
    var = float(np.mean(x * x))
    if n < 8 or var <= 1e-30:
        return None
    length = 1 << (2*n-1).bit_length()
    fft = np.fft.rfft(x, n=length, axis=0)
    sums = np.fft.irfft(fft.conj() * fft, n=length, axis=0)[:n].mean(axis=1)
    acf = sums / (np.arange(n, 0, -1) * var)
    max_lag = n // 4
    stop = next((i for i in range(1, max_lag + 1) if acf[i] <= 0), max_lag + 1)
    tau = max(1.0, float(1 + 2 * np.sum(acf[1:stop])))
    return {"lag1": float(acf[1]), "tau_frames": tau,
            "tau_md_steps": 3*tau, "ess_window_approx": n/tau,
            "truncation_lag": stop-1, "reached_window_limit": stop == max_lag+1,
            "acf_first_20": acf[:min(20,len(acf))].tolist()}


def temporal_diagnostics(chosen, root):
    result = {}
    for env, source in SOURCES.items():
        subset = [r for r in chosen if r["environment"] == env]
        filename = subset[0]["file"]
        source_path = root / source / "xyz_files" / filename
        windows = {}
        for split in QUOTAS:
            rows = [r for r in subset if r["split"] == split]
            times = sorted(int(r["time_index"]) for r in rows)
            windows[split] = {"lo": times[0], "hi": times[-1], "selected": len(rows),
                              "selected_gap_median_steps": float(np.median(np.diff(times))),
                              "energy": [], "forces": [], "volume": []}
        burn = {"early": [], "later": []}
        bins = {}
        for i, (natoms, comment, symbols, positions, forces) in enumerate(read_frames(source_path)):
            t = i*3
            e = parse_energy(comment) / natoms
            f = np.asarray(forces, dtype=float)
            cell = np.asarray(parse_lattice(comment), dtype=float).reshape(3,3)
            vol = abs(float(np.linalg.det(cell))) / natoms
            bin_start = (t // 300) * 300
            bins.setdefault(bin_start, []).append((e, float(np.mean(np.linalg.norm(f, axis=1))), vol))
            for name, (lo,hi) in {"early": (0,100), "later": (300,400)}.items():
                if lo <= t < hi:
                    burn[name].append((e, float(np.mean(np.linalg.norm(f, axis=1))), vol))
            for w in windows.values():
                if w["lo"] <= t <= w["hi"]:
                    w["energy"].append(e); w["forces"].append(f.flatten()); w["volume"].append(vol)
        out = {}
        for split,w in windows.items():
            energy = np.array(w.pop("energy")); force = np.array(w.pop("forces")); volume = np.array(w.pop("volume"))
            w["source_window_frames"] = len(energy)
            w["energy_acf"] = autocorrelation(energy)
            w["force_component_acf"] = autocorrelation(force)
            taus = [a["tau_md_steps"] for a in (w["energy_acf"],w["force_component_acf"]) if a]
            w["suggested_min_block_steps"] = math.ceil(max(taus)) if taus else None
            w["energy_first_half_mean_ev_per_atom"] = float(np.mean(energy[:len(energy)//2]))
            w["energy_second_half_mean_ev_per_atom"] = float(np.mean(energy[len(energy)//2:]))
            w["volume_first_half_mean_a3_per_atom"] = float(np.mean(volume[:len(volume)//2]))
            w["volume_second_half_mean_a3_per_atom"] = float(np.mean(volume[len(volume)//2:]))
            out[split] = w
        b = {}
        for name, values in burn.items():
            a = np.array(values)
            b[name] = {"n": len(a), "energy_per_atom_mean": float(a[:,0].mean()),
                       "mean_atomic_force_norm": float(a[:,1].mean()), "volume_per_atom_mean": float(a[:,2].mean())}
        bin_summary = []
        for start, values in sorted(bins.items()):
            a = np.array(values)
            bin_summary.append({"start_step":start,"end_step_exclusive":start+300,"n":len(a),
                                "energy_per_atom_mean":float(a[:,0].mean()),
                                "mean_atomic_force_norm":float(a[:,1].mean()),
                                "volume_per_atom_mean":float(a[:,2].mean())})
        result[env] = {"source": source, "source_stride_steps": 3, "burnin_diagnostic": b,
                       "trajectory_300_step_windows": bin_summary, "splits": out}
    return result


def qc(chosen, output, old_test):
    identities = set()
    for r in chosen:
        ident = (r["group"], r["file"], int(r["frame_index"]))
        if ident in identities:
            raise ValueError(f"reused source frame: {ident}")
        identities.add(ident)
    seen = {}; results = {}
    for split,name in FILES.items():
        counts=Counter(); maxforce=0.; force_over=0
        for n,c,s,p,f in read_frames(output/name):
            lat=parse_lattice(c); e=parse_energy(c)
            values=[e,*lat,*(v for x in p for v in x),*(v for x in f for v in x)]
            if e is None or len(f)!=n or len(s)!=n or any(not math.isfinite(v) for v in values):
                raise ValueError(f"invalid labels in {name}")
            if np.linalg.det(np.array(lat).reshape(3,3)) <= 0: raise ValueError("invalid cell volume")
            h=frame_hash(s,p,[lat[i] for i in (0,4,8)])
            if h in seen: raise ValueError(f"duplicate across dataset: {split} / {seen[h]}")
            seen[h]=split
            norm=max(math.sqrt(sum(v*v for v in x)) for x in f)
            maxforce=max(maxforce,norm); force_over+=norm>5
            env=next(x.split('=',1)[1] for x in c.split() if x.startswith('environment='))
            counts[env]+=1
        if counts != Counter({env:QUOTAS[split] for env in SOURCES}): raise ValueError(f"bad quota: {counts}")
        results[split]={"frames":sum(counts.values()),"environments":dict(counts),"maximum_force_ev_per_angstrom":maxforce,"frames_over_5_ev_per_angstrom":force_over}
    prior=Counter()
    if old_test:
        for _,c,s,p,_ in read_frames(old_test):
            lat=parse_lattice(c); h=frame_hash(s,p,[lat[i] for i in (0,4,8)])
            if h in seen: prior[seen[h]]+=1
    return {"splits":results,"duplicate_source_frames":0,"duplicate_rounded_geometry":0,
            "previous_pilot_test_hash_overlap":dict(prior), "hash_precision_angstrom":0.001}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--frames',type=Path,default=Path('assets/research/PHASE-1/audit/frames.csv'))
    ap.add_argument('--previous-test',type=Path)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    rows=list(csv.DictReader(args.frames.open()))
    chosen=select(rows)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'frame_manifest.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(chosen[0])); w.writeheader(); w.writerows(chosen)
    outputs=[]
    for split,name in FILES.items():
        outputs.append(build_split(split,[r for r in chosen if r['split']==split],args.data_root,args.output_dir/name,None))
    diagnostics=temporal_diagnostics(chosen,args.data_root)
    checks=qc(chosen,args.output_dir,args.previous_test)
    checks['scf_alignment']=verify_scf(chosen,args.data_root)
    (args.output_dir/'temporal_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')
    summary={'dataset':'interim two-environment Li-F', 'sources':SOURCES, 'per_environment_split_sizes':QUOTAS,
             'source_stride_md_steps':3,'provisional_burn_in_md_steps':BURN_IN,'guard_steps_each_side':GUARD,
             'independence_claim':False,'outputs':outputs,'qc':checks,'audit_sha256':hashlib.sha256(args.frames.read_bytes()).hexdigest()}
    (args.output_dir/'dataset_manifest.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
