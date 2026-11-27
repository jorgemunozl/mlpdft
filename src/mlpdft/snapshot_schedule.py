"""Cycle length, run length and member epochs of the warm-restart snapshot arm.

Kept free of torch and MACE so the run script, the run-time projection and the
tests can use it without a GPU environment.

The snapshot run trains ``burn_in + count`` cosine cycles and keeps the model
at the end of each of the last ``count`` cycles (0-based epoch indices, one
epoch before the next restart). Burn-in cycles are trained and discarded: in
exp-02 the first cycle's member was still converging, so part of the snapshot
spread was a trend between cycles rather than diversity
(``assets/research/methods/ensemble-diversity.md``).

Without burn-in and without an explicit cycle length, the schedule is the one
used before 2026-11-27: ``epochs // count`` epochs per cycle (at least 2), the
run lasts ``epochs`` epochs, and member ``i`` is taken at epoch
``min(i * cycle - 1, epochs - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotSchedule:
    cycle_epochs: int
    total_epochs: int
    burn_in_cycles: int
    member_epochs: tuple[int, ...]


def snapshot_schedule(epochs: int, count: int, burn_in_cycles: int = 0,
                      cycle_epochs: int | None = None) -> SnapshotSchedule:
    """The schedule for ``count`` members after ``burn_in_cycles`` discarded cycles.

    ``cycle_epochs`` sets the cycle length directly; by default it is
    ``epochs // count`` (at least 2), the committee's budget split into
    ``count`` cycles.
    """
    if count < 1:
        raise ValueError(f"snapshot count must be at least 1, got {count}")
    if burn_in_cycles < 0:
        raise ValueError(f"burn-in cycles must be non-negative, got {burn_in_cycles}")
    if cycle_epochs is not None and cycle_epochs < 2:
        raise ValueError(f"a snapshot cycle needs at least 2 epochs, got {cycle_epochs}")
    cycle = cycle_epochs if cycle_epochs is not None else max(2, epochs // count)
    if burn_in_cycles == 0 and cycle_epochs is None:
        total = epochs
        members = tuple(min(i * cycle - 1, total - 1) for i in range(1, count + 1))
    else:
        total = cycle * (burn_in_cycles + count)
        members = tuple((burn_in_cycles + i) * cycle - 1 for i in range(1, count + 1))
    if len(set(members)) != count:
        raise ValueError(f"{count} snapshots do not fit in {total} epochs of {cycle}-epoch cycles")
    return SnapshotSchedule(cycle_epochs=cycle, total_epochs=total,
                            burn_in_cycles=burn_in_cycles, member_epochs=members)
