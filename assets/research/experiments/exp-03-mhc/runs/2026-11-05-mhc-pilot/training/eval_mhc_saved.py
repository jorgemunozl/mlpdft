import json
from pathlib import Path
from types import SimpleNamespace

from mlpdft.uq_compare import _evaluate_method, _read_frames

run = Path("/workspace/runs/lif_uq_mhc_002")
model = run / "mhc_saved.model"
heads = [f"mhc_head_{i}" for i in range(4)]
frames = _read_frames(Path("/workspace/mlpdft/dataset/xyz_files/uq_splits/uq_test.extxyz"))
out = run / "evaluation"
out.mkdir(exist_ok=True)
summary = _evaluate_method(
    "frozen_mhc",
    [(head, model, head) for head in heads],
    frames,
    SimpleNamespace(device="cuda", dtype="float64"),
    out,
)
(out / "frozen_mhc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({
    "method": summary["method"],
    "member_count": summary["member_count"],
    "overall_force_rmse": summary["overall"]["force_rmse_ev_per_angstrom"],
    "overall_energy_rmse": summary["overall"]["energy_rmse_mev_per_atom"],
    "by_group": {g: {"n": v.get("n_frames"), "force_rmse": v.get("force_rmse_ev_per_angstrom"), "spearman": v.get("spearman_uncertainty_vs_force_error")} for g, v in summary["by_group"].items()},
}, indent=2))
