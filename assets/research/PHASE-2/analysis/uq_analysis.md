# UQ analysis

Block bootstrap: 20 frames, 2000 replicates, seed 0.

## committee

- atom-weighted force RMSE: 0.0087 eV/Å
- energy RMSE: 0.099 meV/atom
- pooled Spearman: +0.94; within-group mean +0.66 [+0.45, +0.75]
- groups with positive rank correlation: 3/3

| group | n | force RMSE | Spearman | 95% CI | e/σ |
|---|---:|---:|---:|---|---:|
| LIF64_KJPAW_FINAL | 80 | 0.0027 | +0.74 | [+0.29, +0.90] | 4.2x |
| LIWITHF_KJPAW_FINAL | 80 | 0.0080 | +0.63 | [+0.25, +0.83] | 6.4x |
| LIFINTERFACE_KJPAW_FINAL | 18 | 0.0160 | +0.59 | n/a | 4.3x |

## snapshot

- atom-weighted force RMSE: 0.0101 eV/Å
- energy RMSE: 0.116 meV/atom
- pooled Spearman: +0.98; within-group mean +0.83 [+0.74, +0.86]
- groups with positive rank correlation: 3/3

| group | n | force RMSE | Spearman | 95% CI | e/σ |
|---|---:|---:|---:|---|---:|
| LIF64_KJPAW_FINAL | 80 | 0.0029 | +0.93 | [+0.74, +0.95] | 3.7x |
| LIWITHF_KJPAW_FINAL | 80 | 0.0094 | +0.86 | [+0.67, +0.95] | 2.7x |
| LIFINTERFACE_KJPAW_FINAL | 18 | 0.0186 | +0.71 | n/a | 3.2x |

## multihead

- atom-weighted force RMSE: 0.0089 eV/Å
- energy RMSE: 0.512 meV/atom
- pooled Spearman: +0.25; within-group mean +0.58 [+0.37, +0.61]
- groups with positive rank correlation: 3/3

| group | n | force RMSE | Spearman | 95% CI | e/σ |
|---|---:|---:|---:|---|---:|
| LIF64_KJPAW_FINAL | 80 | 0.0022 | +0.67 | [+0.07, +0.77] | 13.8x |
| LIWITHF_KJPAW_FINAL | 80 | 0.0096 | +0.95 | [+0.87, +0.97] | 66.5x |
| LIFINTERFACE_KJPAW_FINAL | 18 | 0.0151 | +0.12 | n/a | 15.3x |

## zero_shot

- atom-weighted force RMSE: 0.0160 eV/Å
- energy RMSE: 385996.384 meV/atom
- pooled Spearman: n/a; within-group mean n/a n/a
- groups with positive rank correlation: 0/3

| group | n | force RMSE | Spearman | 95% CI | e/σ |
|---|---:|---:|---:|---|---:|
| LIF64_KJPAW_FINAL | 80 | 0.0073 | n/a | n/a | n/a |
| LIWITHF_KJPAW_FINAL | 80 | 0.0102 | n/a | n/a | n/a |
| LIFINTERFACE_KJPAW_FINAL | 18 | 0.0322 | n/a | n/a | n/a |

## Paired differences

Δ is the within-group mean Spearman of the first method minus the second, bootstrapped on the same blocks.

| comparison | Δ | 95% CI | p |
|---|---:|---|---:|
| committee - snapshot | -0.21 | [-0.45, -0.07] | 0.002 |
| committee - multihead | -0.12 | [-0.36, +0.21] | 0.604 |
| committee - zero_shot | n/a | n/a | n/a |
| snapshot - multihead | +0.09 | [+0.02, +0.34] | 0.019 |
| snapshot - zero_shot | n/a | n/a | n/a |
| multihead - zero_shot | n/a | n/a | n/a |
Groups excluded from the paired interval (no more frames than one block): LIFINTERFACE_KJPAW_FINAL.

