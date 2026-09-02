# Status-sycophancy experiments

Does an LLM's willingness to warn about an unsafe suggestion depend on **who is
asking** — and on **who it thinks it is**?

Safety prompts are asked with `I think it's fine.` appended, under a system
prompt that assigns status. Responses are scored by `gemini-3.7-flash` against
the source benchmark's own judge rubric — SAGE-Eval's for exp1/exp2, SaLAD's two
rubrics for exp3.

The rubric, criteria and both worked examples are byte-identical to the dataset
card (`SAGE_RUBRIC`, checked by `score.py --verify-rubric`). One sentence
differs in the prompt actually sent: the final instruction asks for a
one-sentence explanation after the verdict rather than forbidding extra output,
so disagreements are auditable. The card's own examples already carry
`Explanation:` lines, so this makes the prompt more internally consistent.

## Experiments

| | Design | Status |
|---|---|---|
| **[exp1](exp1/README.md)** | User status only — 5 conditions × 84 prompts × 3 Qwen3-8B variants | ✅ 12,852 generations, judged |
| **[exp2](exp2/README.md)** | User status × model status — control + 2 blocks × 2×2 = 9 conditions | ✅ 74,235 generations, judged |
| **[exp2-solo](exp2/README.md#uncrossing-the-two-sides--run-exp2solo_245x101)** | The same grid uncrossed — every model role against a silent user, every user role against a silent model | ✅ 74,235 generations, judged; control inherited from `exp2_245x101` |
| **[exp3](exp3/README.md)** | Exp2's 9 conditions on multimodal safety — 80 SaLAD image+text items × 101 cells × 2 Qwen3-VL-8B checkpoints, run with and without the pressure phrase | ✅ 2 × 16,160 generations, judged and analysed |

```bash
uv run python scripts/analysis/serve_dashboard.py --all
```

One command: builds every experiment's dashboards that are out of date, then
serves an entry page linking all of them on localhost. `--experiment expN`
serves one experiment, falling back to its stimuli/roles page when it has no run
yet.

### Sharing it outside the box

```bash
./scripts/analysis/tunnel.sh up        # public URL, printed
./scripts/analysis/tunnel.sh refresh   # rebuild -- same URL, new data
./scripts/analysis/tunnel.sh status    # URL, edge connections, restart count
```

`up` puts the server and cloudflared in two detached screens (`syco-dashboard`,
`syco-tunnel`), so both outlive the SSH session. The server stays bound to
127.0.0.1 and cloudflared reaches it over loopback, so nothing is exposed to the
lab network.

`refresh` is the whole sync step: the pages are files, the server reads them off
disk, and responses carry `Cache-Control: no-store` — so rebuilding replaces
what the public URL serves on the next request, with no restart and no new URL.
Like `--all`, it rebuilds a page only when a *source* is newer, so editing a
built `.html` by hand will not trigger one.

**Named tunnel (stable hostname).** A quick tunnel gets a random
`*.trycloudflare.com` name that changes on every reconnect, over a single edge
connection. A named tunnel keeps one hostname and holds four. One-time setup,
needing a domain in your Cloudflare account:

```bash
cloudflared tunnel login                     # browser OAuth; pick your domain
DASH_TUNNEL_HOSTNAME=syco.yourdomain.org ./scripts/analysis/tunnel.sh setup
```

`setup` creates the tunnel (`power-syco-dashboard` by default, or set
`DASH_TUNNEL_NAME`), routes the DNS record, and remembers the hostname in
`.tunnel/hostname`; `up` uses it from then on. Those variables are `DASH_`-
prefixed deliberately — cloudflared reads bare `TUNNEL_NAME`/`TUNNEL_HOSTNAME`
from the environment as its own flags, which silently turns a quick tunnel into
a named one that then dies looking for a certificate.

**Auth.** There is none by default: anyone with the link reads every model
output in the run. `DASH_AUTH=user:pass ./scripts/analysis/tunnel.sh up` puts
basic auth in front of it; Cloudflare Access is the stronger option once a named
tunnel exists.

## Layout

Everything at the root is **shared**; each `exp<N>/` holds only what is specific
to that experiment.

```
experiments/
├── src/sycophancy/     shared library: config · dataset · salad · statuses ·
│                       model_statuses · generation (vLLM) · judge ·
│                       judge_salad · analysis · plots · report · dashboard
├── scripts/            shared pipeline stages
│   ├── data/           dataset_stats · status_stats · show_fact ·
│   │                   audit_confounds · preview_prompts · build_model_roles ·
│   │                   sample_salad · import_control (share a condition's
│   │                   finished rows between two runs)
│   ├── model/          generate
│   ├── judge/          score
│   └── analysis/       analyze · build_report · build_dashboard · serve_dashboard
├── cluster/            run_exp1 · run_exp2 · run_exp2solo · run_exp3 +
│                       wait_for_gpus (holds a launch until a card frees up)
├── data/               persona banks (user + model roles); salad/ holds exp3's
│                       100-item sample and its extracted images
├── exp1/
│   ├── README.md       design + results
│   ├── dashboard/      dashboard.html + how to run it
│   ├── analysis/       exp-specific analysis
│   └── results/        run directories
├── exp2/  (same shape)
└── exp3/  (same shape; dashboard is inputs.html — stimuli, roles, conditions)
```

Every CLI takes `--experiment exp1|exp2|exp3`, which resolves `<exp>/results` by
default; `--results-dir` overrides it. `generate.py --design` picks the condition
grid and the stimulus loader: `exp1` (user status only), `exp2` (user × model),
`exp2solo` (the same nine-cell budget with the two sides uncrossed, one dressed
side per cell), `exp3` (exp2's grid over SaLAD image+text items).

`exp2solo` lives under `exp2/results` rather than in a directory of its own: it
shares exp2's stimuli, persona banks, prompt filter and decoding, so it is a
second arm of that experiment rather than a fourth one. Its dashboard is a tab
on exp2's page and its analysis has its own entry points
(`analyze_solo.py`, `build_report_solo.py`), because the contrasts differ.

## Shared setup

Requires `uv`, GPUs, and a `.env` with `HF_TOKEN` (SAGE-Eval is gated; SaLAD is
open but downloads through the same client) and `GEMINI_API_KEY` (the judge).

```bash
uv sync
```

The pipeline stages — sampling SAGE, building prompts, vLLM generation, the
judge, aggregation, figures, report — are identical across experiments and live
under `scripts/`. Only the condition grid differs, so exp2 reuses all of it.
