"""Machine-readable timing lines shared by the training and evaluation steps.

Every step prints ``[timing] {json}`` lines to stdout. The run log therefore
holds the measured cost of each epoch, LLPR frame and evaluated model, and
``mlpdft.utils.project_run_time`` turns a smoke run's lines into a projection
of the full run.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

PREFIX = "[timing] "


def _sync() -> None:
    """Wait for queued GPU work so wall-clock times are not under-counted."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:  # timing must never break a run
        pass


def emit(record: dict[str, Any]) -> None:
    print(PREFIX + json.dumps(record, sort_keys=True), flush=True)


@contextmanager
def timed(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Time the block and emit ``record`` with ``seconds`` added.

    The caller may add fields (such as a frame count) to the yielded dict.
    """
    _sync()
    start = time.perf_counter()
    try:
        yield record
    finally:
        _sync()
        record["seconds"] = round(time.perf_counter() - start, 4)
        emit(record)


def parse(lines) -> list[dict[str, Any]]:
    """Timing records from log lines (other lines are ignored)."""
    records = []
    for line in lines:
        index = line.find(PREFIX)
        if index < 0:
            continue
        try:
            records.append(json.loads(line[index + len(PREFIX):]))
        except json.JSONDecodeError:
            continue
    return records
