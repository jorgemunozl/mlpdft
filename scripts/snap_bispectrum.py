#!/usr/bin/env python3
"""Compute SNAP bispectrum design matrix via FitSNAP (LAMMPSSNAP); optional save to .npy."""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    cand = Path(p)
    if cand.is_file():
        return cand.resolve()
    for base in (Path.cwd(), _REPO):
        alt = (base / p).resolve()
        if alt.is_file():
            return alt
    raise SystemExit(f"not found: {p!r}")


def main() -> int:
    from fitsnap_snap_matrix import bispectrum_width, snap_design_matrix

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infile", default=str(_REPO / "configs/fitsnap/LiBF4-minimal.in"))
    ap.add_argument("--repo-root", default=str(_REPO))
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Also write A to this path via FitSNAP dump_descriptors",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--fit",
        action="store_true",
        help="Run RIDGE fit after descriptors (default: --nofit, descriptors only)",
    )
    args = ap.parse_args()

    infile = _resolve(args.infile)
    root = Path(args.repo_root).resolve()

    out = Path(args.output) if args.output else None
    if out and not out.is_absolute():
        out = root / out

    _a, snap = snap_design_matrix(
        infile,
        root,
        nofit=not args.fit,
        verbose=args.verbose,
        dump_npy=out,
    )
    nw = bispectrum_width(snap.config)
    print("A", _a.shape, _a.dtype, "bispectrum_cols", nw)
    if out:
        print("dump", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
