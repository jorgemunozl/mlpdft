# Run the QE campaign (stages 1 and 2)

Six SLURM jobs with the same settings as the earlier LiF runs (QE 7.4, 40 Ry,
48 cores per node):

| Stage | Jobs | What | Expected time |
|---|---|---|---|
| `stage1_pilot` | 2 | bulk LiF MD at 300 K, 3,000 steps each (SVR and Andersen thermostats) | ~18 h each (30 h requested) |
| `stage2_cutoff` | 4 | 77 single-point calculations on frames of the earlier runs, at 30–80 Ry | 0.5–5 h each (3–9 h requested) |

All six can run at the same time. A second batch (5 MD jobs like stage 1) will
follow once the stage 1 results are back.

## Prerequisites

| Requirement | Notes |
|---|---|
| **SLURM account** | The one used in the earlier `QE_JOB` scripts (`#SBATCH --account=...`) |
| **QuantumESPRESSO 7.4** | The scripts run `module load GCC/13.3.0 OpenMPI/5.0.3 QuantumESPRESSO/7.4` |
| **The original pseudopotentials** | `Li.pbe-sl-kjpaw_psl.1.0.0.UPF` and `F.pbe-n-kjpaw_psl.1.0.0.UPF`, **the same files** used for the earlier 40 Ry runs (their `pseudos` folder) |
| **Scratch space** | ~5 GB per MD job (kept for restarts), ~15 GB peak for a cutoff job (cleaned after each calculation) |

### 1. Unpack

```bash
tar xzf campaign-2026-11.tar.gz
cd campaign-2026-11
```

### 2. Configure (once)

```bash
cp config.env.example config.env
nano config.env
```

Fill in the three values:

| Variable | Example |
|---|---|
| `SLURM_ACCOUNT` | the account number from the old `QE_JOB` scripts |
| `ESPRESSO_PSEUDO` | `/scratch/user/<you>/QE/pseudos` |
| `SCRATCH_BASE` | `/scratch/user/<you>/QE/campaign-2026-11` |

Check that the pseudopotentials are the original files:

```bash
source config.env
md5sum $ESPRESSO_PSEUDO/Li.pbe-sl-kjpaw_psl.1.0.0.UPF $ESPRESSO_PSEUDO/F.pbe-n-kjpaw_psl.1.0.0.UPF
```

Expected: `d2f1e9ef3385d70aefdbf456043e54ab` (Li) and
`48ea186a387ffd7661a6cc7d97657bc3` (F). If they differ, find the original
files. Every job checks this again at start-up and stops if they don't match.

### 3. Submit

```bash
./submit.sh stage1_pilot      # 2 jobs
./submit.sh stage2_cutoff     # 4 jobs
squeue -u $USER
```

> Do not edit the `pw.in` files or the inputs (the only exception is a restart,
> step 5).

### 4. Check after ~10 minutes

```bash
grep -A1 "controlled by" stage1_pilot/*/pw.out      # thermostat of each MD job
grep -c "^!" stage1_pilot/*/pw.out                  # MD steps done (target 3,000)
ls stage2_cutoff/*/outputs/                          # single points appearing
```

- The Andersen job should show `Collision frequency = 0.0100/timestep`.
- If a job stops right away, read its `slurm.e*` file:
  - `pseudopotential checksum mismatch` (exit code 3): `ESPRESSO_PSEUDO` points
    to different UPF files. Fix `config.env` and resubmit only that job, e.g.
    `./submit.sh stage1_pilot bulk_svr_s101`.
  - `pw.x stopped before reading the pseudopotentials`: QE failed to start;
    send the job's `pw.out` and `slurm.e*`.

### 5. If a job hits its time limit

This should not happen with these requests.
- **MD jobs** stop cleanly one hour before the limit. To continue, change
  `restart_mode = 'from_scratch'` to `restart_mode = 'restart'` in that job's
  `pw.in`, then resubmit it, e.g. `./submit.sh stage1_pilot bulk_svr_s101`.
  The new output goes to `pw.restart1.out`; `pw.out` is kept.
- **Cutoff jobs:** just resubmit; finished calculations are skipped.

Keep `$SCRATCH_BASE` until the MD jobs have finished, since restarts need it.

## 6. Give me the results

When all six jobs have finished:

```bash
cd campaign-2026-11
tar czf qe_campaign_results.tar.gz \
  stage1_pilot/*/pw.in stage1_pilot/*/pw*.out stage1_pilot/*/slurm.* \
  stage2_cutoff/*/outputs stage2_cutoff/*/slurm.*
```

Send me `qe_campaign_results.tar.gz` (about 50–100 MB). After that,
`$SCRATCH_BASE` can be deleted.
