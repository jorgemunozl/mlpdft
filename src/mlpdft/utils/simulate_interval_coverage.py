"""Coverage of candidate 95 % intervals (ADR-0009, Inference) for a paired difference on an a x b grid
(a = 3 data units such as replicas, b = 3 seed sets): d = R_r + S_s + E_rs, true mean 0.
Vectorised over simulations."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(1)
A, B, N = 3, 3, 5000

def draw(scale, shape, heavy):
    return (rng.standard_t(3, shape) / np.sqrt(3) if heavy else rng.standard_normal(shape)) * scale

def two_way(d, conservative):
    m = d.mean((1, 2)); rm = d.mean(2); cm = d.mean(1)
    msr = B * rm.var(1, ddof=1); mss = A * cm.var(1, ddof=1)
    resid = d - rm[:, :, None] - cm[:, None, :] + m[:, None, None]
    mse = (resid ** 2).sum((1, 2)) / ((A - 1) * (B - 1))
    if conservative:
        comb = msr + mss; df = comb ** 2 / (msr ** 2 / (A - 1) + mss ** 2 / (B - 1))
    else:
        comb = np.maximum(msr + mss - mse, mse)
        df = np.maximum(comb ** 2 / (msr ** 2 / (A - 1) + mss ** 2 / (B - 1) + mse ** 2 / ((A - 1) * (B - 1))), 1.0)
    return m, stats.t.ppf(0.975, df) * np.sqrt(comb / (A * B))

def t_of(x):  # x: (N, k)
    k = x.shape[1]; return x.mean(1), stats.t.ppf(0.975, k - 1) * x.std(1, ddof=1) / np.sqrt(k)

def boot_rep(d, nb=200):
    r = d.mean(2); idx = rng.integers(0, A, (nb, A)); vals = r[:, idx].mean(2)   # (N, nb)
    lo, hi = np.percentile(vals, [2.5, 97.5], axis=1); return (lo + hi) / 2, (hi - lo) / 2

methods = {"2-way Satterthwaite": lambda d: two_way(d, False), "2-way conservative": lambda d: two_way(d, True),
           "t, 9 cells": lambda d: t_of(d.reshape(N, -1)), "t, replica means": lambda d: t_of(d.mean(2)),
           "t, seed means": lambda d: t_of(d.mean(1)), "boot, 3 replicas": boot_rep}
scen = {"replica-dominated": (1, .2, .3, False), "seed-dominated": (.2, 1, .3, False), "both": (1, 1, .3, False),
        "cell noise": (.3, .3, 1, False), "both, t3 tails": (1, 1, .5, True)}
print("coverage of a nominal 95% interval (median half-width)")
print(f"{'scenario':18s}" + "".join(f"{k:>21s}" for k in methods))
for name, (sr, ss, se, heavy) in scen.items():
    d = draw(sr, (N, A, 1), heavy) + draw(ss, (N, 1, B), heavy) + draw(se, (N, A, B), heavy)
    row = []
    for k, f in methods.items():
        c, h = f(d); row.append(f"{np.mean(np.abs(c) <= h):>13.3f} ({np.median(h):4.2f})")
    print(f"{name:18s}" + "".join(row))
