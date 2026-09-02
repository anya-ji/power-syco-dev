# exp2-solo runbook

How to take `exp2solo_245x101` from where it stands now to a finished, judged,
analysed and published arm of exp2. Written for whoever (or whatever) picks this
up next.

Design notes live in [README.md](README.md#uncrossing-the-two-sides--run-exp2solo_245x101);
this file is only the operational sequence.

## State as of handoff

| | |
|---|---|
| Run directory | `exp2/results/exp2solo_245x101` |
| Control (245 prompts × 3 models = **735 rows**) | ✅ imported from `exp2_245x101`, already judged |
| Eight dressed conditions (**73,500 rows**) | ⏳ not generated — the launcher is parked waiting for a GPU |
| Analysis / figures / report code | ✅ written and smoke-tested on synthetic data |
| Dashboard run tabs | ✅ live; the solo tab appears once the run is judged |

A screen named `syco2s-all` should be running. It waits for a card with ≥24 GiB
free, takes it, sizes `--gpu-memory-utilization` from what is actually free, and
retries when it loses the race (nlp4 is shared and heavily contended — three
launches died this way before the retry loop existed).

```bash
screen -ls | grep syco2s          # is it still parked?
tail -f exp2/results/exp2solo_245x101/logs/gen-all.log
```

## 1. Generate

If the screen is gone, or you have freed up GPUs and want it to move now:

```bash
GPUS="0,1 2,3 4,5" ./cluster/run_exp2solo.sh      # one screen per model
```

or let it pick for itself:

```bash
./cluster/run_exp2solo.sh
```

Generation **resumes** from each model's shard, so relaunching after a failure,
a preemption, or with more GPUs is always safe and never regenerates a finished
cell. Do not pass `--no-resume`.

Expect `Grid: 24500 generations per model x 3 models = 73500 total` in the log.
**If it says 24,745 / 74,235 the control is being regenerated** — `INHERIT_CONTROL`
was unset or empty. Kill it and relaunch with the default.

Done when `logs/status.txt` shows `EXIT all attempt N = 0` and the log ends with
`Done ->`.

```bash
wc -l exp2/results/exp2solo_245x101/raw/generations__qwen3-8b-*.jsonl   # 24500 each
```

## 2. Import the control (only if the run directory was rebuilt)

Already done. It is idempotent, so re-running is harmless — it reports
`0 already present` if there is nothing to add.

```bash
uv run python scripts/data/import_control.py \
  --from exp2/results/exp2_245x101 --to exp2/results/exp2solo_245x101
```

It refuses if the two runs disagree on anything that changes what the control
measures. If it refuses, **do not force it** — read the reason and fix the
mismatch, or generate the control with `INHERIT_CONTROL= ./cluster/run_exp2solo.sh`.

## 3. Judge

```bash
uv run python scripts/judge/score.py --experiment exp2 \
  --run exp2/results/exp2solo_245x101 --max-workers 24
```

Needs `GEMINI_API_KEY` in `.env`. Resumes, and skips the 735 inherited control
rows because they are already in `judged.jsonl` — it should report
`Resuming: 735 already judged`. Expect **74,235 judged rows** at the end
(73,500 generated + 735 inherited) and zero judge errors, as in the other runs.

## 4. Analyse

```bash
uv run python exp2/analysis/analyze_solo.py
uv run python exp2/analysis/build_report_solo.py
```

`analyze_solo.py` defaults to `exp2solo_245x101` and **refuses a crossed run**;
`analyze.py` refuses a solo run. That guard exists because both live under
`exp2/results`, so "the newest run" is no longer unambiguous.

Writes to `results/exp2solo_245x101/`:

- `tables/solo_effects.csv` — the headline: level and presence contrasts
- `tables/side_asymmetry.csv` — model side minus user side, differenced inside each bootstrap draw
- `tables/arm_means.csv`, `condition_means.csv`, `vs_control.csv`,
  `dimension_effects.csv`, `category_effects.csv`, `effects_by_prompt_type.csv`,
  `allpass_ci.csv`, `solo_effects_mixed.csv`
- `figures/exp2solo_fig1..fig8` (PNG) and `report/*.pdf` (vector)
- `report/report.pdf`

`--no-mixed` skips the mixed-effects cross-check, `--no-figures` the plots,
`--n-boot` changes B (default 4000).

**Sanity checks before believing anything:**

- the bootstrap and mixed-effects estimates should agree to ~1e-13; the script
  prints `max |estimate| gap`
- control should be ~52–90 % pass depending on model, matching `exp2_245x101`'s
  control exactly (they are the same rows)
- `p` floors at `2/(B+1) = 0.0005`; a printed `0.0005` means "at the floor", not zero

## 5. Publish

```bash
uv run python scripts/analysis/serve_dashboard.py --all
```

Rebuilds every out-of-date page and serves an entry page on localhost. The exp2
page gains a third run tab, **Solo roles**, beside the crossed run and its
no-suffix replication. To refresh an already-published tunnel instead:

```bash
./scripts/analysis/tunnel.sh refresh
```

## 6. Write up

Fill in the placeholder under
*"Uncrossing the two sides — run `exp2solo_245x101`"* in [README.md](README.md):
a results table in the same shape as the crossed run's, then the reading. The
two questions worth answering there:

1. **Does the level effect survive uncrossing?** An effect present in both
   designs is a status effect proper; one that appears only in the crossed grid
   was contingent on the partner also claiming a standing.
2. **Is presence non-zero?** If it is, behaviour changes when a side is
   described *at all*, before any question of high or low — a different
   mechanism from status sensitivity, and one the crossed design cannot
   separate from its cell means.

Update the status line in the root [README](../README.md) table from 🔄 to ✅.
