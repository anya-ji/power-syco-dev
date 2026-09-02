# exp2 dashboard

Roles and composed system prompts (no results yet):

```bash
uv run python exp2/analysis/build_roles_dashboard.py
```

Then open `exp2/dashboard/roles.html`, or serve everything behind an entry page:

```bash
uv run python scripts/analysis/serve_dashboard.py --all
```

`dashboard.html` carries **one tab per judged run** in `results/`, oldest first
— the crossed design, its no-suffix replication, and the solo-role arm — so a
new design sits beside the one it came from rather than replacing it. Switching
tabs swaps every view and clears the filters, since the runs do not share a
condition vocabulary.

The embedded row cap is the page's budget, not each run's: it is split between
the tabs, so adding a run costs browsing depth in the output explorer rather
than page weight. Every aggregate on the page is computed over all rows either
way.
