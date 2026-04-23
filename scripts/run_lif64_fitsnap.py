#!/usr/bin/env python3
"""
Run FitSNAP on LiF64 JSON using configs/fitsnap/LiF64-NEWJSON.in.

Expects FitSNAP-style frames under:
  examples/LiF64_kjpaw_v2/NEWJSON/DEFAULT/*.json

From repo root:
  python scripts/run_lif64_fitsnap.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_JSON_DIR = _REPO / "examples" / "LiF64_kjpaw_v2" / "NEWJSON" / "DEFAULT"
_DEFAULT_IN = _REPO / "configs" / "fitsnap" / "LiF64-NEWJSON.in"
_PATCHED = _REPO / "scripts" / "run_fitsnap3_patched.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--infile",
        default=str(_DEFAULT_IN),
        help="FitSNAP input deck (default: configs/fitsnap/LiF64-NEWJSON.in)",
    )
    ap.add_argument(
        "--json-dir",
        default=str(_DEFAULT_JSON_DIR),
        help="Directory that must contain at least one *.json (sanity check)",
    )
    args = ap.parse_args()

    infile = Path(args.infile).resolve()
    json_dir = Path(args.json_dir).resolve()

    if not infile.is_file():
        print(f"error: input not found: {infile}", file=sys.stderr)
        return 2
    if not json_dir.is_dir():
        print(f"error: JSON directory not found: {json_dir}", file=sys.stderr)
        return 2
    if not list(json_dir.glob("*.json")):
        print(f"error: no *.json under {json_dir}", file=sys.stderr)
        print(
            "Create DEFAULT and move frames, e.g.\n"
            f"  mkdir -p {json_dir}\n"
            f"  mv examples/LiF64_kjpaw_v2/NEWJSON/output_*.json {json_dir}/",
            file=sys.stderr,
        )
        return 2
    if not _PATCHED.is_file():
        print(f"error: missing {_PATCHED}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(_PATCHED), str(infile)]
    print("Running:", " ".join(cmd))
    print("cwd:", _REPO)
    r = subprocess.run(cmd, cwd=str(_REPO))
    return int(r.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
