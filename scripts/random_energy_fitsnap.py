#!/usr/bin/env python3
"""JSON via FitSNAP scraper → toy random linear energy (no SNAP bispectrum, no fit)."""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

from fitsnap_json_scrape import scrape_frames
from toy_energy import features_post_scrape, linear_energy, random_weights


def _resolve_infile(p: str) -> Path:
    cand = Path(p)
    if cand.is_file():
        return cand.resolve()
    for base in (Path.cwd(), _REPO):
        alt = (base / p).resolve()
        if alt.is_file():
            return alt
    raise SystemExit(f"not found: {p!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infile", default=str(_REPO / "configs/fitsnap/LiBF4-minimal.in"))
    ap.add_argument("--repo-root", default="", help="cwd for FitSNAP PATH (default: repo root)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-cutoff", type=float, default=4.0)
    ap.add_argument("--weight-scale", type=float, default=1.0)
    args = ap.parse_args()

    infile = _resolve_infile(args.infile)
    root = Path(args.repo_root).resolve() if args.repo_root else _REPO

    frames = scrape_frames(infile, root)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    w = random_weights(args.seed, args.weight_scale)

    print("# frames", len(frames), "seed", args.seed, "w", w)
    print("idx,E_true,E_rand,n,vol,sum|F|,sum1/r,file")

    for i, d in enumerate(frames):
        n, vol, sf, ir = features_post_scrape(d, args.pair_cutoff)
        e = linear_energy(w, n, vol, sf, ir)
        print(
            f"{i},{float(d['Energy']):.6f},{e:.6f},"
            f"{n:.0f},{vol:.6f},{sf:.6f},{ir:.6f},{d.get('File', '?')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
