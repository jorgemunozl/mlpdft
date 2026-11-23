# Run the QE campaign

7 SLURM jobs with the same settings as the earlier LiF runs (QE 7.4, 40 Ry).
They take about 1 day if they run in parallel. A second batch of 5 jobs will
follow later.

## Prerequisites

| Requirement | Notes |
|---|---|
| **SLURM account** | The one from the earlier `QE_JOB` scripts |
| **QuantumESPRESSO 7.4** | Loaded by the scripts |
| **Pseudopotentials** | Nothing to do; they come with the repo |

### 1. Get the files

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft/assets/quantum_expresso/campaign-2026-11
```

### 2. Configure

```bash
cp config.env.example config.env
nano config.env
```

Set `SLURM_ACCOUNT` and `SCRATCH_BASE` (e.g. `/scratch/user/<you>/QE/campaign-2026-11`).

### 3. Submit

```bash
./submit.sh all
```

### 4. Check after ~15 minutes

```bash
cat stage0_preflight/*/preflight.log
```

- `PREFLIGHT PASS`: done; the rest runs by itself.
- `PREFLIGHT FAIL`, or any job that stops early or runs out of time: send me
  its `pw.out` and `slurm.e*` files.

## 5. Give me the results

When `squeue -u $USER` shows no jobs left:

```bash
tar czf qe_campaign_results.tar.gz stage0_preflight stage1_pilot stage2_cutoff
```

Send me `qe_campaign_results.tar.gz`.
