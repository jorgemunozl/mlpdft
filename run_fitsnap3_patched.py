import os
import sys
import random
import traceback


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
    log_path = os.path.abspath("fitsnap_run.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("run_fitsnap3_patched.py starting\n")
        log.write(f"argv={argv!r}\n")
        log.write(f"cwd={os.getcwd()}\n")

    if len(argv) != 2:
        print("usage: python run_fitsnap3_patched.py <input.in>", file=sys.stderr)
        return 2

    infile = argv[1]
    if not os.path.isfile(infile):
        print(f"error: input file not found: {infile}", file=sys.stderr)
        return 2

    try:
        _patch_fitsnap_randint()

        from fitsnap3lib.fitsnap import FitSnap  # type: ignore

        snap = FitSnap(input=infile, comm=None, arglist=["--overwrite", "--verbose"])
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

