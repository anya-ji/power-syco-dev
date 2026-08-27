# power-syco-dev

Status-based sycophancy experiments: does an LLM's willingness to warn about an
unsafe suggestion depend on who the user appears to be?

SAGE-Eval safety prompts are asked with `I think it's fine.` appended, under a
system prompt assigning the user a persona, across a 2×2 of status level ×
domain relevance plus a no-role control. Responses are scored with the SAGE-Eval
judge rubric.

**See [`experiments/README.md`](experiments/README.md)** for the design, the data
audit, and how to run the pipeline.

```
experiments/
├── src/sycophancy/   config · dataset · statuses · generation (vLLM)
│                     judge (SAGE rubric) · analysis · plots · report
├── scripts/          data/ · model/ · judge/ · analysis/
├── cluster/          screen launchers + SCF sbatch
└── data/             persona banks
```

Requires a `.env` (untracked) with `HF_TOKEN` — SAGE-Eval is gated — and
`GEMINI_API_KEY` for the judge.
