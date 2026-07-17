#!/usr/bin/env python3
"""One-shot migration: convert existing per-group .txt metrics to a single JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from mlpdft.plot_metrics import ATOM_COUNTS, SHORT_LABELS, GroupMetrics, MetricsReader

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    metrics_dir = SCRIPT_DIR / "metrics"

    # MACE mock_2_test
    mace_txt_dir = metrics_dir / "mock_2_test"
    mace_metrics = MetricsReader.from_txt_dir(mace_txt_dir, style="mace")

    json_path = mace_txt_dir / "mock_2_test.json"
    payload = {
        "model_key": "mock_2_test",
        "frame_stride": 5,
        "max_frames": 100,
        "results": [
            {
                "group": m.group,
                "short_label": SHORT_LABELS.get(m.group, m.group),
                "energy_rmse_per_atom": m.energy_rmse_per_atom_mev / 1000,
                "force_rmse_all": m.force_rmse_mev_ang / 1000,
            }
            for m in sorted(mace_metrics, key=lambda x: x.group)
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"✓ Wrote {json_path}")
    print(f"  {len(payload['results'])} groups")


if __name__ == "__main__":
    main()
