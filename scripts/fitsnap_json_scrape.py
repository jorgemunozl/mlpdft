"""FitSNAP JSON scraper only: same path as training before descriptors (no SNAP bispectrum)."""

from __future__ import annotations

import os
from pathlib import Path


def patch_randint() -> None:
    import fitsnap3lib.parallel_tools as pt
    import random as R

    pt.randint = lambda a, b: R.randint(int(a), int(b))


def scrape_frames(infile: Path, repo_root: Path) -> list:
    """
    infile: absolute path to FitSNAP .in
    repo_root: cwd FitSNAP uses to resolve [PATH] dataPath
    """
    patch_randint()
    infile = infile.resolve()
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    rel = os.path.relpath(infile, repo_root)

    from fitsnap3lib.io.input import Config
    from fitsnap3lib.parallel_tools import ParallelTools
    from fitsnap3lib.scrapers.scraper_factory import scraper as make_scraper

    pt = ParallelTools(comm=None)
    cfg = Config(pt, input=rel, arguments_lst=["--overwrite"])
    scr = make_scraper(cfg.sections["SCRAPER"].scraper, pt, cfg)
    scr.scrape_groups()
    scr.divvy_up_configs()
    return scr.scrape_configs()
