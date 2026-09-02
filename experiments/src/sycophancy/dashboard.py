"""Build a single-file local HTML dashboard for a run.

No external assets: the data is embedded as JSON and the charts are inline SVG,
so the file opens straight from disk with `file://`. Long generations are
truncated for display only -- the untruncated text stays in the JSONL.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import RunPaths
from .config import COND_LABEL_FLAT, CONDITIONS, MODEL_VARIANTS, SALAD_IMAGE_DIR
from .model_statuses import (
    BLOCKS, EXP2_CONDITIONS, EXP2_LABEL_FLAT, EXP2_SOLO_CONDITIONS, LEVELS,
)


def cond_label(c: str) -> str:
    """Human label for a condition from either design."""
    return COND_LABEL_FLAT.get(c) or EXP2_LABEL_FLAT.get(c) or c


#: exp2-solo: hue = the one dressed side, shade = its level. Matches
#: plots.SOLO_COLOR so the page and the figures agree.
SOLO_COLOR = {"uhigh": "#3d5a80", "ulow": "#9fb3c8",
              "mhigh": "#9e3d52", "mlow": "#e0a0ac"}


def cond_color(c: str) -> str:
    """Colour for a condition from any of the designs."""
    if c in COND_COLOR:
        return COND_COLOR[c]
    if c == "control":
        return "#8d8579"
    parts = c.split("_")
    if len(parts) == 2 and parts[1] in SOLO_COLOR:
        return SOLO_COLOR[parts[1]]
    # exp2: teal = domain block, rose = irrelevant; dark = model high.
    for b, base in (("domain", ("#1b6b5e", "#7fc0b0")), ("irrel", ("#9e3d52", "#e0a0ac"))):
        if c.startswith(b + "_"):
            return base[0] if c.endswith("_mhigh") else base[1]
    return "#8d8579"

MAX_CHARS = 500           # per text field, preview only
COND_COLOR = {
    "control": "#8b8b8b", "domain_high": "#1f78b4", "domain_low": "#8ec7e6",
    "irrel_high": "#e31a1c", "irrel_low": "#f79a9b",
}


def _truncate(text, limit: int = MAX_CHARS) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + f"\n… [+{len(text)-limit} chars]"


def load_rows(paths: RunPaths) -> list[dict]:
    """Judged rows if present, otherwise raw generations."""
    if paths.judged.exists():
        rows = [json.loads(l) for l in paths.judged.read_text().splitlines() if l.strip()]
        for r in rows:
            r["_judged"] = True
        return rows
    rows = []
    for f in paths.all_generations():
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for r in rows:
        r["_judged"] = False
    return rows


def summarize(rows: list[dict]) -> dict:
    """Aggregates computed over ALL rows, before any display truncation."""
    judged = [r for r in rows if r.get("verdict")]
    by_mc = defaultdict(lambda: {"n": 0, "pass": 0})
    by_md = defaultdict(lambda: {"n": 0, "pass": 0})
    per_model = defaultdict(
        lambda: {"n": 0, "nj": 0, "pass": 0, "trunc": 0, "empty": 0, "tok": 0})

    for r in rows:
        m = r["model"]
        p = per_model[m]
        p["n"] += 1
        p["trunc"] += bool(r.get("truncated"))
        p["empty"] += not str(r.get("response") or "").strip()
        p["tok"] += int(r.get("n_output_tokens") or 0)
        if r.get("verdict"):
            ok = int(r["verdict"] == "pass")
            p["nj"] += 1
            p["pass"] += ok
            c = by_mc[(m, r["condition"])]
            c["n"] += 1
            c["pass"] += ok
            d = by_md[(m, r["condition"], r.get("status_dimension", "none"))]
            d["n"] += 1
            d["pass"] += ok

    # All-persona pass rate: a fact counts only if every one of its rows passed
    sage = defaultdict(lambda: {"facts": 0, "passed": 0})
    fact_ok = defaultdict(lambda: True)
    fact_seen = set()
    for r in judged:
        k = (r["model"], r["condition"], r["safety_fact"])
        fact_seen.add(k)
        if r["verdict"] != "pass":
            fact_ok[k] = False
    per_cell = Counter((r["model"], r["condition"], r["safety_fact"])
                                   for r in judged)
    for k in fact_seen:
        s = sage[(k[0], k[1])]
        s["facts"] += 1
        s["passed"] += int(fact_ok[k])
        s["per"] = per_cell[k]

    # Roles actually used, per side. exp1 names the user's role
    # ``status_label``; exp2/exp3 give both sides one and name them
    # ``user_label`` / ``model_label`` -- which is why a dashboard reading only
    # ``status_label`` showed "(no role)" on every exp2/exp3 row.
    roles: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "pass": 0})
    for r in judged:
        cond = r.get("condition") or ""
        implied = ("high" if cond.endswith("_high")
                   else "low" if cond.endswith("_low") else "")
        sides = [("user", r.get("user_label"), r.get("user_level") or implied),
                 ("model", r.get("model_label"), r.get("model_level") or "")]
        if r.get("status_label"):
            sides = [("user", r.get("status_label"), implied)]
        for side, label, level in sides:
            if not label:
                continue
            v = roles[(side, r.get("status_dimension", "none"), level or "—", label)]
            v["n"] += 1
            v["pass"] += int(r.get("verdict") == "pass")

    # Categories, split by gold label where the dataset has one. SaLAD is half
    # genuine hazards and half safe requests, and the two are scored by opposite
    # rubrics, so a bare row count hides which half a number came from.
    cats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "unsafe": 0, "safe": 0, "pass": 0,
                 "unsafe_n": 0, "unsafe_pass": 0, "safe_n": 0, "safe_pass": 0}
    )
    items: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in judged:
        c = cats[r["category"]]
        c["n"] += 1
        c["pass"] += int(r.get("verdict") == "pass")
        label = r.get("safety_type")
        if label in ("safe", "unsafe"):
            c[label] += 1
            c[f"{label}_n"] += 1
            c[f"{label}_pass"] += int(r.get("verdict") == "pass")
            items[r["category"]][label].add(r.get("prompt_idx"))
    for name, c in cats.items():
        c["unsafe_items"] = len(items[name]["unsafe"])
        c["safe_items"] = len(items[name]["safe"])

    return {
        "per_model": {k: dict(v) for k, v in per_model.items()},
        "by_condition": {f"{m}||{c}": v for (m, c), v in by_mc.items()},
        "by_dimension": {f"{m}||{c}||{d}": v for (m, c, d), v in by_md.items()},
        "sage": {f"{m}||{c}": v for (m, c), v in sage.items()},
        "n_judged": len(judged),
        "categories": {k: dict(v) for k, v in
                       sorted(cats.items(), key=lambda kv: -kv[1]["n"])},
        "hasLabels": any(r.get("safety_type") for r in rows),
        "dimensions": sorted({r.get("status_dimension", "none") for r in rows}),
        "roles": [{"side": side, "dimension": dim, "level": lvl, "label": lab, **v}
                  for (side, dim, lvl, lab), v in sorted(roles.items())],
    }


def compact(rows: list[dict], max_chars: int = MAX_CHARS) -> list[dict]:
    """Display-sized records for the browser table.

    These are previews. The dashboard embeds all 12k+ rows, so every extra
    kilobyte per row costs ~12 MB of page weight; the untruncated text lives in
    raw/*.jsonl and the prompt dumps. ``raw_output`` is dropped whenever it is
    identical to ``response`` (true for every non-thinking generation), which
    roughly halves the payload.
    """
    # Repeated strings become indices into shared tables: each of the 84 prompts
    # and safety facts otherwise appears ~150 times, and each system prompt ~100.
    tables: dict[str, list[str]] = {"prompt": [], "safety_fact": [],
                                    "system_prompt": [], "status_label": [],
                                    "user_label": [], "model_label": [],
                                    "user_msg": []}
    index: dict[str, dict[str, int]] = {k: {} for k in tables}

    def ref(field: str, value) -> int:
        value = "" if value is None else str(value)
        idx = index[field]
        if value not in idx:
            idx[value] = len(tables[field])
            tables[field].append(value)
        return idx[value]

    keep = ("model", "prompt_idx", "category", "condition", "status_dimension",
            "persona_idx", "verdict", "finish_reason", "n_output_tokens",
            "truncated", "judge_error",
            # exp3 only: the gold label picks the judge rubric, and the image is
            # half the stimulus, so both belong on the row.
            "safety_type", "image")
    out = []
    for r in rows:
        rec = {k: r.get(k) for k in keep}
        for field in tables:
            rec[field] = ref(field, r.get(field))
        response = r.get("response") or ""
        raw = r.get("raw_output") or ""
        rec["response"] = _truncate(response, max_chars)
        rec["reasoning"] = _truncate(r.get("reasoning"), max_chars)
        rec["raw_output"] = "" if raw == response else _truncate(raw, max_chars)
        rec["judge_raw"] = _truncate(r.get("judge_raw"), 200)
        rec["judge_explanation"] = _truncate(r.get("judge_explanation"), 400)
        out.append(rec)
    return out, tables


#: detail-panel thumbnails: big enough to see the hazard, small enough that 80
#: of them add a few MB rather than a few hundred
THUMB_MAX_SIDE = 560
THUMB_QUALITY = 75


def collect_images(rows: list[dict], image_dir: Path,
                   max_side: int = THUMB_MAX_SIDE) -> dict[str, str]:
    """Data-URI thumbnails for the stimuli these rows reference.

    exp3 rows name an image; exp1/exp2 rows have none and this returns {}. The
    16,160 rows of a run are backed by only 80 distinct images, so they are
    interned by filename rather than carried per row.
    """
    names = sorted({r["image"] for r in rows if r.get("image")})
    if not names:
        return {}
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"NOTE: {len(names)} rows name an image but {image_dir} is missing; "
              f"building without thumbnails")
        return {}

    import base64
    import io

    from PIL import Image

    out = {}
    for name in names:
        path = image_dir / name
        if not path.exists():
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=THUMB_QUALITY)
        out[name] = ("data:image/jpeg;base64,"
                     + base64.b64encode(buf.getvalue()).decode())
    if len(out) < len(names):
        print(f"NOTE: {len(names) - len(out)} image(s) not found under {image_dir}")
    return out


def read_prompt_files(paths: RunPaths) -> dict:
    files = {}
    for d, tag in ((paths.templates, "template"), (paths.samples, "sample")):
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == ".txt":
                files[f"{tag}: {f.name}"] = _truncate(f.read_text(), 20000)
    return files


def sample_rows(rows: list[dict], cap: int | None, seed: int = 0) -> list[dict]:
    """Cap the rows embedded for the explorer, stratified by model x condition.

    Aggregates are always computed on the full set; this only limits what the
    browser has to parse. A run of 74k rows with thinking traces embeds to
    ~170 MB otherwise, which no longer opens comfortably.
    """
    if cap is None or len(rows) <= cap:
        return rows
    import random

    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("model"), r.get("condition"))].append(r)
    per = max(1, cap // max(1, len(groups)))
    rng = random.Random(seed)
    out = []
    for g in groups.values():
        out.extend(rng.sample(g, min(per, len(g))))
    return out


#: short name for a run tab, so "exp2solo_245x101" reads as what it is rather
#: than as a directory name. Matched by prefix, longest first.
RUN_TAB_LABEL = {
    "exp2solo": "Solo roles",
    "exp2": "Crossed roles",
    "exp1": "User status",
    "exp3": "Multimodal",
}


def run_tab_label(name: str) -> str:
    for prefix in sorted(RUN_TAB_LABEL, key=len, reverse=True):
        if name.startswith(prefix):
            suffix = "no suffix" if "nosuffix" in name else ""
            base = RUN_TAB_LABEL[prefix]
            return f"{base} ({suffix})" if suffix else base
    return name


def run_payload(paths: RunPaths, max_chars: int = MAX_CHARS,
                max_rows: int | None = None,
                image_dir: Path | None = None) -> dict:
    """Everything one run contributes to the page.

    Aggregates are computed over every row; only the explorer's row list is
    sampled, so a smaller ``max_rows`` costs browsing depth and never changes a
    number on the page.
    """
    rows = load_rows(paths)
    if not rows:
        raise SystemExit(f"no generations found under {paths.raw}")

    cfg = json.loads(paths.run_config.read_text()) if paths.run_config.exists() else {}
    # Conditions come from the data, not a fixed list: exp1 has five, exp2 nine
    # crossed or nine solo.
    present = {r.get("condition") for r in rows}
    known = list(dict.fromkeys(
        list(CONDITIONS) + list(EXP2_CONDITIONS) + list(EXP2_SOLO_CONDITIONS)))
    conditions = [c for c in known if c in present]
    conditions += sorted(c for c in present if c and c not in conditions)

    shown = sample_rows(rows, max_rows)
    rows_c, tables_c = compact(shown, max_chars)
    return {
        "run": paths.root.name,
        "tab": run_tab_label(paths.root.name),
        "config": cfg,
        "summary": summarize(rows),          # always over ALL rows
        "nTotalRows": len(rows),
        "nShownRows": len(shown),
        "rows": rows_c,
        "tables": tables_c,
        "prompts": read_prompt_files(paths),
        # The stimulus is half of an exp3 item; a response that names a hazard
        # cannot be checked against a filename.
        "images": collect_images(rows, image_dir or SALAD_IMAGE_DIR),
        "conditions": conditions,
        "condColor": {c: cond_color(c) for c in conditions},
        "condLabel": {c: cond_label(c) for c in conditions},
    }


def build(paths: RunPaths | list[RunPaths], out: Path | None = None,
          max_chars: int = MAX_CHARS, max_rows: int | None = None,
          image_dir: Path | None = None) -> Path:
    """Render one page. Several runs become several tabs on it.

    The row cap is the page's budget, not each run's: embedding N runs at the
    full cap each would multiply page weight by N, and these pages already run
    to tens of megabytes. It is split between them instead, so adding a run
    costs browsing depth rather than making the page too heavy to open.
    """
    runs = [paths] if isinstance(paths, RunPaths) else list(paths)
    if not runs:
        raise SystemExit("build() needs at least one run")
    per_run = max_rows if max_rows is None else max(1, max_rows // len(runs))
    payload = {
        "runs": [run_payload(r, max_chars, per_run, image_dir) for r in runs],
        "models": {k: v.display for k, v in MODEL_VARIANTS.items()},
    }
    out = out or runs[0].dashboard
    out.write_text(_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    mb = out.stat().st_size / 1e6
    total = sum(p["nTotalRows"] for p in payload["runs"])
    shown = sum(p["nShownRows"] for p in payload["runs"])
    note = "" if shown == total else f", {shown:,} embedded for browsing"
    tabs = "" if len(runs) == 1 else f", {len(runs)} run tabs"
    print(f"Dashboard -> {out}  ({mb:.1f} MB, {total:,} rows{note}{tabs})")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sycophancy run dashboard</title>
<style>
:root{color-scheme:light;--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e2e1de;--card:#fff;--accent:#3d5a80;
--pass:#2f7d52;--fail:#b3392f;--code:#f4f4f2}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
background:var(--bg);color:var(--fg)}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px}
nav{display:flex;gap:2px;padding:0 24px;border-bottom:1px solid var(--line);background:var(--card);
position:sticky;top:0;z-index:10;flex-wrap:wrap}
nav button{border:0;background:none;color:var(--mut);padding:11px 14px;cursor:pointer;
font:inherit;border-bottom:2px solid transparent}
nav button.on{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}
nav.runs{background:var(--code);padding-top:4px}
nav.runs button{padding:9px 16px;display:inline-flex;flex-direction:column;gap:1px;
line-height:1.25;text-align:left}
nav.runs button .rn{font:10.5px ui-monospace,Menlo,monospace;color:var(--mut)}
nav.runs button.on{background:var(--card);border-bottom-color:var(--accent)}
main{padding:22px 24px;max-width:1500px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 16px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:26px;font-weight:650;margin-top:3px;letter-spacing:-.02em}
.card .d{color:var(--mut);font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
tbody tr:hover{background:var(--code)}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:9px}
pre{background:var(--code);padding:11px 13px;border-radius:7px;overflow-x:auto;white-space:pre-wrap;
word-break:break-word;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:6px 0}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600}
.pass{background:color-mix(in srgb,var(--pass) 18%,transparent);color:var(--pass)}
.fail{background:color-mix(in srgb,var(--fail) 18%,transparent);color:var(--fail)}
.u{background:color-mix(in srgb,#1b6b5e 16%,transparent);color:#1b6b5e}
.m{background:color-mix(in srgb,#9e3d52 16%,transparent);color:#9e3d52}
.t-unsafe{background:color-mix(in srgb,#9e3d52 15%,transparent);color:#9e3d52}
.t-safe{background:color-mix(in srgb,#3f7a4f 15%,transparent);color:#3f7a4f}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
select,input{font:inherit;padding:6px 9px;border:1px solid var(--line);border-radius:7px;
background:var(--card);color:var(--fg)}
input[type=search]{min-width:240px}
h2{font-size:15px;margin:26px 0 10px;letter-spacing:-.01em}
h2:first-child{margin-top:0}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:7px 0;padding:9px 13px}
summary{cursor:pointer;font-weight:600;font-size:13px}
.mut{color:var(--mut)}
.bar{height:9px;border-radius:5px;background:var(--code);overflow:hidden;min-width:70px}
.bar>i{display:block;height:100%}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin:8px 0}
.legend span{display:flex;align-items:center;gap:5px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.stim{display:block;max-width:440px;width:100%;border-radius:8px;margin:6px 0;
border:1px solid var(--line)}
.note{color:var(--mut);font-size:12px;margin:8px 0 16px}
.mono{font:12px ui-monospace,Menlo,monospace}
</style></head><body>
<header><h1>Status-Sycophancy Run</h1><div class="sub" id="sub"></div></header>
<nav id="runs" class="runs"></nav>
<nav id="nav"></nav><main id="main"></main>
<script>
const ALL = __DATA__;
// One page, one run selected. Every view below reads D, so switching runs is a
// reassignment plus a redraw rather than a second copy of the rendering code.
let runIdx = 0;
let D = ALL.runs[runIdx];
const E=(s)=>String(s==null?"":s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=(a,b)=>b?(100*a/b).toFixed(1)+"%":"—";
const mname=(m)=>ALL.models[m]||m;
// A function, not a snapshot: the palette belongs to the selected run, and a
// captured object would keep colouring the page with the first run's.
const CC=(c)=>(D.condColor&&D.condColor[c])||"#8d8579";
// Repeated strings are interned in D.tables; rows carry indices into them.
const S=(r,f)=>D.tables[f][r[f]]||"";

function subline(){
  document.getElementById("sub").textContent =
    `${D.run} · ${(D.nTotalRows||D.rows.length).toLocaleString()} generations · ${D.summary.n_judged.toLocaleString()} judged`
    + ((D.nShownRows && D.nShownRows < D.nTotalRows)
       ? ` · ${D.nShownRows.toLocaleString()} sampled into the explorer` : "");
}

const TABS=[["overview","Overview"],["conditions","By condition"],["dims","By dimension"],
            ["explore","Explore outputs"],["prompts","Prompts"],["config","Config"]];
let tab="overview";
const nav=document.getElementById("nav");
TABS.forEach(([id,label])=>{const b=document.createElement("button");b.textContent=label;
  b.onclick=()=>{tab=id;draw()};b.dataset.id=id;nav.appendChild(b)});

// The run bar is only drawn when there is a choice to make; a single-run page
// looks exactly as it did before.
const runNav=document.getElementById("runs");
if(ALL.runs.length>1){
  ALL.runs.forEach((r,i)=>{const b=document.createElement("button");
    b.innerHTML=`${E(r.tab)}<span class="rn">${E(r.run)}</span>`;
    // Filters are reset on switch: the runs do not share a condition
    // vocabulary, so a filter carried over from one would silently match
    // nothing in the other.
    b.onclick=()=>{runIdx=i;D=ALL.runs[i];resetFilters();draw()};
    b.dataset.idx=i;runNav.appendChild(b)});
}else{runNav.remove()}

// exp1 names the user's role status_label; exp2/exp3 give both sides one.
function roleOf(r){
  const u=S(r,"user_label"), m=S(r,"model_label");
  if(m||u) return `model: ${m||"—"} · user: ${u||"—"}`;
  return S(r,"status_label")||"(no role)";
}

function overview(){
  const s=D.summary, pm=s.per_model;
  let h='<div class="grid">';
  for(const m of Object.keys(pm)){const v=pm[m];
    h+=`<div class="card"><div class="k">${E(mname(m))}</div>
    <div class="v">${s.n_judged?pct(v.pass,v.nj):v.n.toLocaleString()}</div>
    <div class="d">${s.n_judged?`pass rate · ${v.n.toLocaleString()} gens`:"generations"}</div>
    <div class="d">truncated ${pct(v.trunc,v.n)} · empty ${pct(v.empty,v.n)}</div>
    <div class="d">${Math.round(v.tok/Math.max(1,v.n))} out-tokens avg</div></div>`}
  h+='</div>';
  if(!s.n_judged) h+='<div class="note">Not yet judged — pass rates appear after the judge runs.</div>';
  const L=s.hasLabels;
  h+='<h2>Safety categories</h2>';
  if(L) h+='<div class="note">Every category contributes both halves of the benchmark: <b>unsafe</b> items where the response has to catch the hazard in the image, and <b>safe</b> items where it has to help without refusing or over-warning. The two are scored by opposite rubrics, so pass rates are shown per half — a single number would average them into something meaningless.</div>';
  h+='<div class="wrap"><table><thead><tr><th>Category</th>'
   + (L?'<th>Unsafe items</th><th>Safe items</th>':'')
   + '<th>Rows</th>'
   + (L&&s.n_judged?'<th>Unsafe pass</th><th>Safe pass</th>':'')
   + '</tr></thead><tbody>';
  let tot={n:0,ui:0,si:0,un:0,up:0,sn:0,sp:0};
  for(const [c,v] of Object.entries(s.categories)){
    tot.n+=v.n; tot.ui+=v.unsafe_items||0; tot.si+=v.safe_items||0;
    tot.un+=v.unsafe_n||0; tot.up+=v.unsafe_pass||0; tot.sn+=v.safe_n||0; tot.sp+=v.safe_pass||0;
    h+=`<tr><td>${E(c)}</td>`
     + (L?`<td><span class="pill t-unsafe">${v.unsafe_items||0}</span></td><td><span class="pill t-safe">${v.safe_items||0}</span></td>`:"")
     + `<td>${v.n.toLocaleString()}</td>`
     + (L&&s.n_judged?`<td>${v.unsafe_n?pct(v.unsafe_pass,v.unsafe_n):"—"} <span class="mut">(${(v.unsafe_n||0).toLocaleString()})</span></td>
        <td>${v.safe_n?pct(v.safe_pass,v.safe_n):"—"} <span class="mut">(${(v.safe_n||0).toLocaleString()})</span></td>`:"")
     + `</tr>`;
  }
  h+=`<tr><td><b>all</b></td>`
   + (L?`<td><b>${tot.ui}</b></td><td><b>${tot.si}</b></td>`:"")
   + `<td><b>${tot.n.toLocaleString()}</b></td>`
   + (L&&s.n_judged?`<td><b>${tot.un?pct(tot.up,tot.un):"—"}</b></td><td><b>${tot.sn?pct(tot.sp,tot.sn):"—"}</b></td>`:"")
   + `</tr></tbody></table></div>`;

  const rl=s.roles||[];
  h+=`<h2>Roles used (${rl.length})</h2>`;
  if(!rl.length) h+='<div class="note">This run assigns no roles — control only.</div>';
  else{
    const twoSided=rl.some(r=>r.side==="model");
    h+=`<div class="note">${twoSided?"Both sides carry a role: the assistant's own claimed standing and the user's. ":""}The rate here pools every condition a role appears in, so it describes the bank — read the condition table for the contrasts.</div>`;
    h+='<div class="wrap"><table><thead><tr><th>Role</th><th>Side</th><th>Bank</th><th>Level</th><th>Rows</th>'
     + (s.n_judged?'<th>Pass rate</th>':'') + '</tr></thead><tbody>';
    for(const r of rl)
      h+=`<tr><td>${E(r.label)}</td><td><span class="pill ${r.side==="model"?"m":"u"}">${E(r.side)}</span></td>
        <td class="mono">${E(r.dimension)}</td><td>${E(r.level)}</td><td>${r.n.toLocaleString()}</td>`
       + (s.n_judged?`<td>${pct(r.pass,r.n)}</td>`:"") + `</tr>`;
    h+='</tbody></table></div>';
  }
  return h;
}

function condTable(){
  const s=D.summary; if(!s.n_judged) return '<div class="note">No judged rows yet.</div>';
  let h='<div class="note"><b>All-persona</b> = share of safety facts where <em>every</em> persona passed. Cell sizes differ (irrel_* spans 4 status channels = 20 personas; domain_* has 5), so read it <em>within</em> a condition, not across. This is not the published SAGE-Eval score, which varies prompts rather than personas.</div>';
  h+='<div class="legend">';
  for(const c of D.conditions) h+=`<span><i class="sw" style="background:${CC(c)}"></i>${E((D.condLabel&&D.condLabel[c])||c)}</span>`;
  h+='</div>';
  for(const m of Object.keys(s.per_model)){
    h+=`<h2>${E(mname(m))}</h2><div class="wrap"><table><thead><tr><th>Condition</th>
    <th>Pass rate</th><th></th><th>All-persona</th><th>n</th></tr></thead><tbody>`;
    for(const c of D.conditions){
      const v=s.by_condition[m+"||"+c]; if(!v) continue;
      const sg=s.sage[m+"||"+c]; const r=100*v.pass/v.n;
      h+=`<tr><td><i class="sw" style="background:${CC(c)}"></i> ${E((D.condLabel&&D.condLabel[c])||c)}</td>
      <td><b>${r.toFixed(1)}%</b></td>
      <td style="width:190px"><div class="bar"><i style="width:${r}%;background:${CC(c)}"></i></div></td>
      <td>${sg?pct(sg.passed,sg.facts)+` <span class="mut">(${sg.passed}/${sg.facts} facts, ${sg.per} personas)</span>`:"—"}</td>
      <td>${v.n.toLocaleString()}</td></tr>`}
    h+='</tbody></table></div>'}
  return h;
}

function dimTable(){
  const s=D.summary; if(!s.n_judged) return '<div class="note">No judged rows yet.</div>';
  let h='<div class="note">Irrelevant conditions span four generic status dimensions. A wide spread here means the status <em>channel</em> matters, not just the level.</div>';
  for(const m of Object.keys(s.per_model)){
    h+=`<h2>${E(mname(m))}</h2><div class="wrap"><table><thead><tr><th>Condition</th>
    <th>Dimension</th><th>Pass rate</th><th></th><th>n</th></tr></thead><tbody>`;
    const keys=Object.keys(s.by_dimension).filter(k=>k.startsWith(m+"||")).sort();
    for(const k of keys){const [,c,d]=k.split("||");const v=s.by_dimension[k];
      const r=100*v.pass/v.n;
      h+=`<tr><td><i class="sw" style="background:${CC(c)}"></i> ${E((D.condLabel&&D.condLabel[c])||c)}</td><td class="mono">${E(d)}</td>
      <td><b>${r.toFixed(1)}%</b></td>
      <td style="width:170px"><div class="bar"><i style="width:${r}%;background:${CC(c)}"></i></div></td>
      <td>${v.n}</td></tr>`}
    h+='</tbody></table></div>'}
  return h;
}

let filt={model:"",cond:"",dim:"",cat:"",verdict:"",label:"",q:""},page=0;
function resetFilters(){filt={model:"",cond:"",dim:"",cat:"",verdict:"",label:"",q:""};page=0}
const PAGE=40;
function explore(){
  const opts=(a,cur)=>['<option value="">all</option>'].concat(
    a.map(v=>`<option ${v===cur?"selected":""}>${E(v)}</option>`)).join("");
  const models=Object.keys(D.summary.per_model);
  const cats=Object.keys(D.summary.categories);
  let h=`<div class="controls">
   <select id="f-model">${opts(models,filt.model)}</select>
   <select id="f-cond">${opts(D.conditions,filt.cond)}</select>
   <select id="f-dim">${opts(D.summary.dimensions,filt.dim)}</select>
   <select id="f-cat">${opts(cats,filt.cat)}</select>
   <select id="f-verdict">${opts(["pass","fail"],filt.verdict)}</select>
   ${D.summary.hasLabels?`<select id="f-label">${opts(["unsafe","safe"],filt.label)}</select>`:""}
   <input type="search" id="f-q" placeholder="search prompt / response / role" value="${E(filt.q)}">
  </div>`;
  const q=filt.q.toLowerCase();
  const rows=D.rows.filter(r=>
    (!filt.model||r.model===filt.model)&&(!filt.cond||r.condition===filt.cond)&&
    (!filt.dim||r.status_dimension===filt.dim)&&(!filt.cat||r.category===filt.cat)&&
    (!filt.verdict||r.verdict===filt.verdict)&&
    (!filt.label||r.safety_type===filt.label)&&
    (!q||(S(r,"prompt")+" "+r.response+" "+roleOf(r)+" "+(r.judge_explanation||"")).toLowerCase().includes(q)));
  const pages=Math.max(1,Math.ceil(rows.length/PAGE));
  page=Math.min(page,pages-1);
  h+=`<div class="note">${rows.length.toLocaleString()} matching${(D.nShownRows&&D.nShownRows<D.nTotalRows)?` of a ${D.nShownRows.toLocaleString()}-row sample (stats above use all ${D.nTotalRows.toLocaleString()})`:""} · page ${page+1}/${pages} · text is previewed; full output in <span class="mono">raw/*.jsonl</span>
   <button id="prev">prev</button> <button id="next">next</button></div>`;
  for(const r of rows.slice(page*PAGE,(page+1)*PAGE)){
    const v=r.verdict?`<span class="pill ${r.verdict==="pass"?"pass":"fail"}">${r.verdict}</span>`:"";
    h+=`<details><summary>${v} <span class="mono">${E(r.model)}</span> ·
      <i class="sw" style="background:${CC(r.condition)}"></i> ${E(r.condition)} ·
      ${r.safety_type?`<span class="pill t-${E(r.safety_type)}">${E(r.safety_type)}</span> `:""}${E(roleOf(r))}
      <span class="mut">— ${E(S(r,"prompt").slice(0,90))}…</span></summary>
      <div class="mut mono" style="margin:6px 0">${E(r.category)} · dim=${E(r.status_dimension)} ·
      persona_idx=${r.persona_idx}${r.image?" · image="+E(r.image):""} · ${r.n_output_tokens} tok ·
      finish=${E(r.finish_reason)}${r.truncated?" · TRUNCATED":""}</div>
      ${r.image&&D.images&&D.images[r.image]?`<b>Image shown to the model</b>
      <img class="stim" src="${D.images[r.image]}" alt="${E(r.image)}">`:""}
      <b>Safety fact</b><pre>${E(S(r,"safety_fact"))}</pre>
      <b>System prompt</b><pre>${E(S(r,"system_prompt")||"(none)")}</pre>
      <b>User</b><pre>${E(S(r,"user_msg")||S(r,"prompt"))}</pre>
      ${r.reasoning?`<b>Reasoning (parsed)</b><pre>${E(r.reasoning)}</pre>`:""}
      <b>Response (parsed)</b><pre>${E(r.response)}</pre>
      ${r.raw_output?`<b>Raw output (differs from parsed)</b><pre>${E(r.raw_output)}</pre>`:""}
      ${r.judge_explanation?`<b>Judge explanation</b><pre>${E(r.judge_explanation)}</pre>`:""}
      ${r.judge_raw?`<b>Judge raw</b><pre>${E(r.judge_raw)}</pre>`:""}
      ${r.judge_error?`<b>Judge error</b><pre>${E(r.judge_error)}</pre>`:""}
    </details>`}
  return h;
}
function wireExplore(){
  const set=(id,k)=>{const el=document.getElementById(id);if(el)el.onchange=e=>{filt[k]=e.target.value;page=0;draw()}};
  set("f-model","model");set("f-cond","cond");set("f-dim","dim");set("f-cat","cat");
  set("f-verdict","verdict");set("f-label","label");
  const q=document.getElementById("f-q");
  if(q)q.oninput=e=>{filt.q=e.target.value;page=0;clearTimeout(window._t);
    window._t=setTimeout(()=>{draw();const n=document.getElementById("f-q");n.focus();
    n.setSelectionRange(n.value.length,n.value.length)},250)};
  const p=document.getElementById("prev"),n=document.getElementById("next");
  if(p)p.onclick=()=>{if(page>0){page--;draw()}};
  if(n)n.onclick=()=>{page++;draw()};
}

function prompts(){
  let h='<div class="note">Templates verbatim, plus one fully-rendered sample per condition. Every rendered prompt is in <span class="mono">prompts/model/</span> and <span class="mono">prompts/judge/</span>.</div>';
  for(const [name,text] of Object.entries(D.prompts))
    h+=`<details><summary>${E(name)}</summary><pre>${E(text)}</pre></details>`;
  return h;
}
function config(){return `<pre>${E(JSON.stringify(D.config,null,2))}</pre>`}

function draw(){
  subline();
  document.querySelectorAll("#nav button").forEach(b=>b.classList.toggle("on",b.dataset.id===tab));
  document.querySelectorAll("#runs button").forEach(b=>
    b.classList.toggle("on",Number(b.dataset.idx)===runIdx));
  const m=document.getElementById("main");
  m.innerHTML={overview,conditions:condTable,dims:dimTable,explore,prompts,config}[tab]();
  if(tab==="explore")wireExplore();
  window.scrollTo(0,0);
}
draw();
</script></body></html>
"""


# ── Entry page ────────────────────────────────────────────────────────────────

INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sycophancy experiments</title>
<style>
:root{color-scheme:light;--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e2e1de;--card:#fff;--accent:#1b6b5e}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif;
background:var(--bg);color:var(--fg);display:flex;justify-content:center;padding:48px 24px}
.page{max-width:760px;width:100%}
h1{font-size:22px;margin:0 0 6px;letter-spacing:-.015em}
.sub{color:var(--mut);margin-bottom:30px;font-size:14px}
a.exp{display:block;text-decoration:none;color:inherit;background:var(--card);
border:1px solid var(--line);border-radius:11px;padding:18px 20px;margin-bottom:14px;
transition:border-color .12s}
a.exp:hover{border-color:var(--accent)}
.name{font-size:16px;font-weight:650;letter-spacing:-.01em}
.desc{color:var(--mut);font-size:13.5px;margin-top:4px}
.links{margin-top:11px;display:flex;gap:8px;flex-wrap:wrap}
.links span{font-size:12px;border:1px solid var(--line);border-radius:99px;
padding:3px 11px;color:var(--mut)}
.empty{color:var(--mut);font-size:13px;font-style:italic}
</style></head><body><div class="page">
<h1>Status-sycophancy experiments</h1>
<div class="sub">Does an LLM's willingness to warn about an unsafe suggestion depend on who is asking &mdash; and on who it thinks it is?</div>
__CARDS__
</div></body></html>
"""


def build_index(root: Path, out: Path | None = None) -> Path:
    """Entry page listing each experiment and what it has available."""
    root = Path(root)
    meta = {
        "exp1": ("User status only",
                 "5 user-status conditions x 84 SAGE prompts x 3 Qwen3-8B variants. "
                 "12,852 generations, judged with the SAGE-Eval rubric."),
        "exp2": ("User status x model status",
                 "Crosses exp1's user factor with the assistant's own claimed "
                 "expertise: 5 user x 4 model = 20 cells per prompt."),
        "exp3": ("Exp2's design on multimodal safety",
                 "Same 9-condition cross, swapped stimuli and model: 100 SaLAD "
                 "image+text items (50 unsafe, 50 safe) x 101 cells x 2 "
                 "Qwen3-VL-8B checkpoints."),
    }
    cards = []
    for exp in sorted(d.name for d in root.iterdir()
                      if d.is_dir() and d.name.startswith("exp")):
        title, desc = meta.get(exp, (exp, ""))
        links, href = [], None
        for rel, label in [(f"{exp}/dashboard/dashboard.html", "results dashboard"),
                           (f"{exp}/dashboard/inputs.html", "stimuli, roles &amp; conditions"),
                           (f"{exp}/dashboard/roles.html", "roles &amp; prompts"),
                           (f"{exp}/README.md", "design notes")]:
            if (root / rel).exists():
                links.append(f"<span>{label}</span>")
                href = href or rel
        for run in sorted((root / exp / "results").glob("*/dashboard.html")):
            rel = str(run.relative_to(root))
            links.append(f"<span>run: {run.parent.name}</span>")
            href = href or rel
        body = (f'<div class="links">{"".join(links)}</div>' if links
                else '<div class="empty">nothing built yet</div>')
        cards.append(
            f'<a class="exp" href="{href or "#"}"><div class="name">'
            f"{exp} &mdash; {title}</div>"
            f'<div class="desc">{desc}</div>{body}</a>'
        )
    out = out or root / "index.html"
    out.write_text(INDEX_TEMPLATE.replace("__CARDS__", "\n".join(cards)))
    print(f"Index -> {out}")
    return out
