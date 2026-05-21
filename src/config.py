from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MaceConfig:
    """
    Settings for a MACE model evaluation.
    """

    model: str = "small"
    perconfig: Optional[Path] = None
    test_extxyz: Optional[Path] = None
    mace_model: Optional[Path] = None

    group: str = "DEFAULT"
    training_frac: float = 0.8
    testing_frac: float = 0.2
    device: Device = "cpu"

    dtype: Dtype = "float32"
    out_csv: Path = field(default_factory=lambda: Path("outputs/mace_eval_results.csv"))
    max_frames: Optional[int] = None

    def __post_init__(self) -> None:
        if self.perconfig is not None:
            self.perconfig = Path(self.perconfig)
        if self.test_extxyz is not None:
            self.test_extxyz = Path(self.test_extxyz)
        if self.mace_model is not None:
            self.mace_model = Path(self.mace_model)
        if self.json_root is not None:
            self.json_root = Path(self.json_root)
        self.out_csv = Path(self.out_csv)

    def resolve_json_root(self) -> Path:
        if self.json_root is not None:
            return self.json_root
        return _REPO_ROOT
