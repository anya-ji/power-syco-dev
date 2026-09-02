# exp3 dashboard

The stimuli, the role banks, the nine conditions and the two judge rubrics —
everything that goes *into* the run, before any generation exists. One command
builds it and serves it on localhost:

```bash
uv run python scripts/analysis/serve_dashboard.py --all
```

That rebuilds any page whose sources changed, then serves every experiment
behind an entry page. `--experiment exp3` serves just this one, falling back to
`inputs.html` while there is no run to show. To build the file without serving
it, `exp3/analysis/build_dashboard.py` still does that on its own.

`inputs.html` is ~7 MB (the 100 stimulus images are embedded as data URIs,
so the page is self-contained and can be copied anywhere). Five tabs:

| Tab | What it shows |
|---|---|
| **Stimuli** | all 100 items as image + question, filterable by category and by safe/unsafe. Click one for its gold reference note and its nine composed system messages. |
| **Condition grid** | the 2×2 of user × model status inside each relevance block, plus control |
| **Composed prompts** | every paired role in every condition, for one category |
| **Role banks** | the 10 new SaLAD domain banks and the 4 shared generic ones, user clause beside its mirrored model clause |
| **Judge rubrics** | the unsafe and safe rubrics exactly as sent |

The name is `inputs.html`, not `dashboard.html`, so a later results dashboard for
this experiment can take the usual name without overwriting it.

