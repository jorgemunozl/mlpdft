import os
import sys
import random
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _patch_fitsnap_randint() -> None:
    """
    FitSNAP 3.1.x uses randint(0, 1e5) which is a float in Python 3.14+.
    Monkey-patch the already-imported symbol in fitsnap3lib.parallel_tools.
    """
    import fitsnap3lib.parallel_tools as pt  # type: ignore

    def _randint_int_bounds(a, b):
        return random.randint(int(a), int(b))

    pt.randint = _randint_int_bounds


def main(argv: list[str]) -> int:
    log_path = str(_REPO_ROOT / "logs" / "fitsnap_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("run_fitsnap3_patched.py starting\n")
        log.write(f"argv={argv!r}\n")
        log.write(f"cwd={os.getcwd()}\n")

    if len(argv) != 2:
        print("usage: python scripts/run_fitsnap3_patched.py <input.in>", file=sys.stderr)
        return 2

    raw = argv[1]
    p = Path(raw)
    if not p.is_file():
        alt = _REPO_ROOT / raw
        if alt.is_file():
            p = alt
    if not p.is_file():
        print(f"error: input file not found: {raw}", file=sys.stderr)
        return 2

    os.chdir(_REPO_ROOT)
    infile_rel = os.path.relpath(p.resolve(), _REPO_ROOT)

    try:
        _patch_fitsnap_randint()

        from fitsnap3lib.fitsnap import FitSnap  # type: ignore

        snap = FitSnap(input=infile_rel, comm=None, arglist=["--overwrite", "--verbose"])
        snap.scrape_configs(delete_scraper=True)
        snap.process_configs(delete_data=True)
        snap.perform_fit()
        snap.write_output()

        with open(log_path, "a", encoding="utf-8") as log:
            log.write("FitSNAP finished successfully\n")
        return 0
    except Exception:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("FitSNAP failed with exception:\n")
            log.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

