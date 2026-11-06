"""Resolve the calibration split without allowing accidental split reuse."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path


def resolve_calibration_file(
    calibration_file: Path | None, train_file: Path, valid_file: Path, test_file: Path
) -> tuple[Path, bool]:
    """Return the effective path and whether it is the legacy validation fallback.

    Explicit calibration is required to be a distinct file with different
    content from each other split. Frame-level disjointness still requires the
    dataset manifest; file hashes cannot detect partial overlap.
    """
    if calibration_file is None:
        warnings.warn(
            "No --calibration-file supplied; using validation for calibration "
            "for compatibility with older runs.",
            UserWarning,
            stacklevel=2,
        )
        return valid_file, True

    path = calibration_file.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).digest()
    for name, other in (("train", train_file), ("validation", valid_file), ("test", test_file)):
        if path == other.resolve() or digest == hashlib.sha256(other.read_bytes()).digest():
            raise ValueError(f"Calibration file is identical to the {name} split: {path}")
    return path, False
