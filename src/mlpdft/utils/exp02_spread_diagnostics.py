"""Compare the committee and snapshot spreads of exp-02, per environment.

Reads the per-frame prediction files that exp-02 wrote
(``independent_committee_predictions.csv``, ``warm_snapshots_predictions.csv``,
``zero_shot_predictions.csv``) and prints, for each environment:

* the median force spread of each arm and the median per-frame ratio;
* the median ratio of frame force RMSE to spread;
* Spearman correlations between the arms' errors, and of each arm's spread with
  the committee's error.

These numbers back section 14.3 of ``assets/research/literature/REVIEW.md``.
The correlations are over time-correlated frames from contiguous splits, so they
are descriptive.

Usage
-----
    python -m mlpdft.utils.exp02_spread_diagnostics <exp-02 evaluation dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SIGMA = "force_uncertainty_ev_per_angstrom"
ERROR = "force_rmse_ev_per_angstrom"


def load(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y)[0])


def main(evaluation_dir: Path) -> None:
    committee = load(evaluation_dir / "independent_committee_predictions.csv")
    snapshot = load(evaluation_dir / "warm_snapshots_predictions.csv")
    if not np.array_equal(committee["frame_index"], snapshot["frame_index"]):
        raise SystemExit("the two prediction files list different frames")
    for group in sorted(set(committee["group"])):
        ix = committee["group"] == group
        s_c, s_s = committee[SIGMA][ix].astype(float), snapshot[SIGMA][ix].astype(float)
        e_c, e_s = committee[ERROR][ix].astype(float), snapshot[ERROR][ix].astype(float)
        print(f"{group} (n = {ix.sum()})")
        print(f"  median sigma (meV/A): committee {1e3 * np.median(s_c):.2f}, "
              f"snapshot {1e3 * np.median(s_s):.2f}, per-frame ratio {np.median(s_s / s_c):.2f}")
        print(f"  median error/sigma: committee {np.median(e_c / s_c):.1f}, "
              f"snapshot {np.median(e_s / s_s):.1f}")
        print(f"  rho(snapshot error, committee error) {rho(e_s, e_c):+.2f}")
        print(f"  rho(committee sigma, committee error) {rho(s_c, e_c):+.2f}; "
              f"rho(snapshot sigma, committee error) {rho(s_s, e_c):+.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
