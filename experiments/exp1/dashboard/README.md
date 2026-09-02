# exp1 dashboard

```bash
uv run python scripts/analysis/serve_dashboard.py --experiment exp1
```

Open the `http://localhost:8000/` it prints.

Rebuild the file without serving:

```bash
uv run python scripts/analysis/build_dashboard.py --experiment exp1 --out exp1/dashboard/dashboard.html
```
