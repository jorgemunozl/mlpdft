#!/usr/bin/env python3
"""Convert the three protocol Li/F (NVT) QE outputs into labeled extxyz for the UQ test.

Settings (agreed for the UQ comparison):
  - equilibration cutoff: discard the first 100 parsed frames
  - frame stride: 30 (all groups, to decorrelate temporally adjacent AIMD frames)
  - cap: 200 frames per group

The NPT variants are excluded: the downstream diffusion validation runs in
fixed-cell NVT, and NPT carries a density/barostat confound (non-uniform
settings).

Produces ``dataset/<group>/xyz_files/<group>_<stride>_200.extxyz`` with
``REF_energy`` / ``REF_forces`` labels and ``config_type=<group>``.
"""

from __future__ import annotations

from mlpdft.config import DataSetConfig
from mlpdft.dataset_mlpdft import DataSet

EQUILIBRATION_CUTOFF = 100
MAX_FRAMES = 200

# group -> frame stride. Stride 30 decorrelates the temporally adjacent AIMD
# frames; the tradeoff is fewer frames per group (most severe for the short
# LIFINTERFACE_KJPAW_FINAL trajectory).
UQ_GROUPS = {
    "LIF64_KJPAW_FINAL": 30,
    "LIWITHF_KJPAW_FINAL": 30,
    "LIFINTERFACE_KJPAW_FINAL": 30,
}


def main() -> None:
    for group, stride in UQ_GROUPS.items():
        print(f"\n=== {group} (stride {stride}) ===")
        config = DataSetConfig(
            group=group,
            frame_stride=stride,
            max_frames=MAX_FRAMES,
            equilibration_cutoff=EQUILIBRATION_CUTOFF,
        )
        DataSet(config).convert_qe_out_to_extxyz()
    print("\nDone.")


if __name__ == "__main__":
    main()
