# Run the QE campaign (stages 0–2)

Seven SLURM jobs with the same settings as the earlier LiF runs (QE 7.4, 40 Ry,
48 cores per node):

| Stage | Jobs | What | Expected time |
|---|---|---|---|
| `stage0_preflight` | 1 | one bulk LiF single point that must reproduce the earlier 40 Ry run | ~5 min |
| `stage1_pilot` | 2 | bulk LiF MD at 300 K, 3,000 steps each (SVR and Andersen thermostats); they start only if the preflight passes | ~18 h each (30 h requested) |
| `stage2_cutoff` | 4 | 77 single points on frames of the earlier runs, at 30–80 Ry | 0.5–5 h each (3–9 h requested) |

A second batch (5 MD jobs like stage 1) will follow once the stage 1 results
are back.

## Prerequisites

| Requirement | Notes |
|---|---|
| **SLURM account** | The one used in the earlier `QE_JOB` scripts (`#SBATCH --account=...`) |
| **QuantumESPRESSO 7.4** | The scripts run `module load GCC/13.3.0 OpenMPI/5.0.3 QuantumESPRESSO/7.4` |
| **Pseudopotentials** | Nothing to download: the jobs use the repository's `assets/quantum_expresso/pseudo/` |
| **Scratch space** | ~5 GB per MD job (kept for restarts), ~15 GB peak per cutoff job (cleaned after each calculation) |

### 1. Get the files

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft/assets/quantum_expresso/campaign-2026-11
```

If you were sent `campaign-2026-11.tar.gz` instead:
`tar xzf campaign-2026-11.tar.gz && cd quantum_expresso/campaign-2026-11`.

### 2. Configure (once)

```bash
cp config.env.example config.env
nano config.env
```

Fill in the two empty values:

| Variable | Example |
|---|---|
| `SLURM_ACCOUNT` | the account number from the old `QE_JOB` scripts |
| `SCRATCH_BASE` | `/scratch/user/<you>/QE/campaign-2026-11` |

Leave `ESPRESSO_PSEUDO` as it is; it already points to `../pseudo`.

### 3. Submit

```bash
./submit.sh all
squeue -u $USER
```

This submits the preflight, the two MD jobs (they wait for the preflight) and
the four cutoff jobs.

> Do not edit the `pw.in` files or the inputs (the only exception is a restart,
> step 5).

### 4. Check after ~15 minutes

```bash
cat stage0_preflight/*/preflight.log
```

- `PREFLIGHT PASS`: the MD jobs start by themselves. Nothing else to do.
- `PREFLIGHT FAIL`: the MD jobs are cancelled automatically, while the cutoff
  jobs keep running. Send me `stage0_preflight/*/preflight.log` and
  `stage0_preflight/*/pw.out`.

Once the MD jobs are running:

```bash
grep -A1 "controlled by" stage1_pilot/*/pw.out   # thermostat of each MD job
grep -c "^!" stage1_pilot/*/pw.out               # MD steps done (target 3,000)
ls stage2_cutoff/*/outputs/                       # single points appearing
```

If any job stops right away, read its `slurm.e*` file:
- `pseudopotential checksum mismatch` (exit code 3): `ESPRESSO_PSEUDO` was
  changed. Restore it as in `config.env.example`, then resubmit that job.
- `pw.x stopped before reading the pseudopotentials`: QE failed to start. Send
  me that job's `pw.out` and `slurm.e*`.

To resubmit one job: `./submit.sh stage1_pilot bulk_svr_s101`.

### 5. If a job hits its time limit

This should not happen with these requests.
- **MD jobs** stop cleanly one hour before the limit. To continue, change
  `restart_mode = 'from_scratch'` to `restart_mode = 'restart'` in that job's
  `pw.in` and resubmit it. The new output goes to `pw.restart1.out`; `pw.out` is
  kept.
- **Cutoff jobs:** just resubmit; finished calculations are skipped.

Keep `$SCRATCH_BASE` until the MD jobs have finished, since restarts need it.

## 6. Give me the results

When all jobs have finished:

```bash
tar czf qe_campaign_results.tar.gz \
  stage0_preflight/*/preflight.log stage0_preflight/*/pw.out \
  stage1_pilot/*/pw.in stage1_pilot/*/pw*.out stage1_pilot/*/slurm.* \
  stage2_cutoff/*/outputs stage2_cutoff/*/slurm.*
```

Send me `qe_campaign_results.tar.gz` (about 50–100 MB). After that,
`$SCRATCH_BASE` can be deleted.
