"""Pedagogy-only: hand-made features + random linear energy (not a trained potential)."""

from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np


def _det3(m: list[list[float]]) -> float:
    return (
        float(m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]))
        - float(m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]))
        + float(m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    )


def _norm3(v: Iterable[float]) -> float:
    x, y, z = v
    return math.sqrt(float(x * x + y * y + z * z))


def _sum_inv_r(positions: np.ndarray, cutoff: float) -> float:
    n = positions.shape[0]
    c2 = cutoff * cutoff
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            r2 = float(np.dot(d, d))
            if 1e-24 < r2 <= c2:
                s += 1.0 / math.sqrt(r2)
    return s


def features_post_scrape(data: dict, pair_cutoff: float) -> tuple[float, float, float, float]:
    pos = np.asarray(data["Positions"])
    frc = np.asarray(data["Forces"])
    lat = np.asarray(data["Lattice"])
    n = float(pos.shape[0])
    vol = abs(_det3(lat.tolist()))
    sum_f = float(sum(_norm3(frc[i]) for i in range(pos.shape[0])))
    invr = _sum_inv_r(pos, pair_cutoff)
    return n, vol, sum_f, invr


def random_weights(seed: int, scale: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0.0, scale) for _ in range(5)]


def linear_energy(w: list[float], n: float, vol: float, sum_f: float, invr: float) -> float:
    x = [1.0, n, vol, sum_f, invr]
    return sum(wi * xi for wi, xi in zip(w, x))
