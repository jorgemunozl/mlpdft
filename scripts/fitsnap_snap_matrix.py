"""FitSNAP LAMMPSSNAP: design matrix `a` = SNAP bispectrum rows (see FitSNAP docs / LAMMPS compute snap)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fitsnap_json_scrape import patch_randint


def bispectrum_width(cfg) -> int:
    sec = cfg.sections["BISPECTRUM"]
    if cfg.sections["CALCULATOR"].nonlinear:
        return int(sec.ncoeff)
    w = sec.ncoeff * sec.numtypes
    if not sec.bzeroflag:
        w += sec.numtypes
    return int(w)


def snap_design_matrix(
    infile: Path,
    repo_root: Path,
    *,
    nofit: bool = True,
    verbose: bool = False,
    dump_npy: Path | None = None,
    extra_args: list[str] | None = None,
) -> tuple[np.ndarray, object]:
    patch_randint()
    infile = infile.resolve()
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    rel = os.path.relpath(infile, repo_root)

    fs_args: list[str] = ["--overwrite"]
    if nofit:
        fs_args.append("-nf")
    if verbose:
        fs_args.append("-v")
    if extra_args:
        fs_args.extend(extra_args)

    from fitsnap3lib.fitsnap import FitSnap

    snap = FitSnap(input=rel, comm=None, arglist=fs_args)
    snap.scrape_configs(delete_scraper=True)
    snap.process_configs(delete_data=True)
    if not nofit:
        snap.perform_fit()
        snap.write_output()

    a = np.asarray(snap.pt.shared_arrays["a"].array)
    if dump_npy is not None:
        dump_npy = dump_npy.resolve()
        dump_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(dump_npy, a)
    return a, snap
