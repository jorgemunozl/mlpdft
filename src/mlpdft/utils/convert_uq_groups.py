#!/usr/bin/env python3
"""Convert the six protocol Li/F QE outputs into labeled extxyz for the UQ test.

Settings (agreed for the UQ comparison):
  - equilibration cutoff: discard the first 100 parsed frames
  - frame stride: 10 (6 for LIFINTERFACE_KJPAW_FINAL, whose trajectory is short)
  - cap: 200 frames per group

Produces ``dataset/<group>/xyz_files/<group>_<stride>_200.extxyz`` with
``REF_energy`` / ``REF_forces`` labels and ``config_type=<group>``.
"""

from __future__ import annotations

from mlpdft.config import DataSetConfig
from mlpdft.dataset_mlpdft import DataSet

EQUILIBRATION_CUTOFF = 100
MAX_FRAMES = 200

# group -> frame stride. LIFINTERFACE_KJPAW_FINAL has only ~1436 raw frames, so
# stride 6 is needed to reach 200 after the 100-frame cutoff.
UQ_GROUPS = {
    "LIF64_KJPAW_FINAL": 10,
    "LIF64_KJPAW_NPT_FINAL": 10,
    "LIWITHF_KJPAW_FINAL": 10,
    "LIWITHF_NPT_FINAL": 10,
    "LIFINTERFACE_KJPAW_FINAL": 6,
    "LIFINTERFACE_KJPAW_NPT_FINAL": 10,
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
