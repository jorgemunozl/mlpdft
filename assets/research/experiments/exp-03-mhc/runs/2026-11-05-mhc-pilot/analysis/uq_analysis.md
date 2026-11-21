# UQ analysis

Block bootstrap: 20 frames, 2000 replicates, seed 0.

## mhc

- atom-weighted force RMSE: 0.0102 eV/Å
- energy RMSE: 2205.095 meV/atom
- pooled Spearman: +0.86; within-group mean +0.46 [+0.23, +0.57]
- groups with positive rank correlation: 3/3

| group | n | force RMSE | Spearman | 95% CI | e/σ |
|---|---:|---:|---:|---|---:|
| LIF64_KJPAW_FINAL | 80 | 0.0044 | +0.57 | [+0.04, +0.83] | 11.6x |
| LIWITHF_KJPAW_FINAL | 80 | 0.0095 | +0.71 | [+0.19, +0.89] | 23.9x |
| LIFINTERFACE_KJPAW_FINAL | 18 | 0.0183 | +0.08 | n/a | 16.1x |

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

## Paired differences

Δ is the within-group mean Spearman of the first method minus the second, bootstrapped on the same blocks.

| comparison | Δ | 95% CI | p |
|---|---:|---|---:|
| mhc - committee | -0.04 | [-0.34, +0.21] | 0.856 |
| mhc - snapshot | -0.25 | [-0.57, -0.05] | 0.017 |
| mhc - multihead | -0.17 | [-0.45, +0.19] | 0.448 |
| committee - snapshot | -0.21 | [-0.45, -0.07] | 0.002 |
| committee - multihead | -0.12 | [-0.36, +0.21] | 0.604 |
| snapshot - multihead | +0.09 | [+0.02, +0.34] | 0.019 |
Groups excluded from the paired interval (no more frames than one block): LIFINTERFACE_KJPAW_FINAL.

