from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, Optional

from constants import (
    LIF64_DIR,
    LIF64_GROUP,
    PATH_MODEL,
    PREDICTION_DIR,
    XYZ_DIR,
)


@dataclass
class MaceConfig:
    """
    Settings for a MACE model evaluation.
    """

    model_path: str = field(
        default=PATH_MODEL,
        metadata={"description": "MACE model path"},
    )

    extyz_path: str = field(
        default=str(LIF64_DIR / XYZ_DIR / "LIF64_10_20.extxyz"),
        metadata={"description": "Path to test extxyz file"},
    )

    output: Path = field(
        default=Path(PREDICTION_DIR / "pred_LIF64_10_20.extxyz"),
        metadata={"description": "Output path"},
    )

    perconfig: Optional[Path] = field(
        default=None, metadata={"description": "Path to per-configuration file"}
    )

    batch_size: int = field(
        default=1,
        metadata={"description": "Batch size for evaluation"},
    )

    compute_stress: bool = field(
        default=False,
        metadata={"description": "Compute stress"},
    )

    compute_bec: bool = field(
        default=False,
        metadata={"description": "Compute BEC"},
    )
    device: Literal["cpu", "cuda"] = "cpu"

    dtype: Literal["float32", "float64"] = "float32"

    info_prefix: str = field(
        default="",
        metadata={"description": ""},
    )


@dataclass
class QEToExtXYZConfig:
    """Configuration for converting a QE output file into extxyz."""

    in_path: Path = field(
        default=Path(LIF64_DIR / "LiF64_kjpaw.out"),
        metadata={"description": "Path to Quantum ESPRESSO .out file"},
    )
    out_path: Path = field(
        default=Path("out.extxyz"),
        metadata={"description": "Output multi-frame extxyz path"},
    )
    frame_stride: int = field(
        default=10,
        metadata={"description": "Keep one frame every N parsed frames (>=1)"},
    )
    group: str = field(
        default=LIF64_GROUP,
        metadata={"description": "Group name for the output frames"},
    )
    max_frames: int | None = field(
        default=20,
        metadata={"description": "Optional cap on written frames after striding"},
    )
    include_stress: bool = field(
        default=False,
        metadata={
            "description": "If true, copy QE stress labels into output frames when available"
        },
    )
    config_type: str = field(
        default="Default",
        metadata={
            "description": "Value stored in Atoms.info['config_type'] for each frame"
        },
    )

    def __post_init__(self):
        self.set_out_path()

    @classmethod
    def describe_fields(cls) -> dict[str, str]:
        """Return a field->description map for documentation or logging."""
        return {f.name: f.metadata.get("description", "") for f in fields(cls)}

    def set_out_path(self):
        path = (
            LIF64_DIR
            / XYZ_DIR
            / f"{self.group.split('_')[0]}_{self.frame_stride}_{self.max_frames}.extxyz"
        )
        self.out_path = path

    def validate(self) -> None:
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be >= 1")
