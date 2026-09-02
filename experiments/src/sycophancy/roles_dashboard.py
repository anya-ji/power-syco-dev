"""Exp2 roles dashboard: browse the user and model role banks and the 20-cell
cross of composed system prompts.

Standalone single-file HTML, same conventions as the results dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ALL_GENERIC_DIMENSIONS, CATEGORY_TO_DOMAIN
from .model_statuses import (
    BLOCKS, EXP2_CONDITIONS, EXP2_LABEL_FLAT, LEVELS, ModelStatusBank, build_cells,
)
from .statuses import StatusBank


def collect(user_bank: StatusBank, model_bank: ModelStatusBank,
            generic_dimensions: list[str]) -> dict:
    categories = sorted(CATEGORY_TO_DOMAIN)

    # Every (user, model) pairing per condition, per category.
    cells = {}
    for cat in categories:
        per_cond = {}
        for cond in EXP2_CONDITIONS:
            per_cond[cond] = [
                {"dimension": c.dimension, "user_label": c.user_label,
                 "model_label": c.model_label, "user_clause": c.user_clause,
                 "model_clause": c.model_clause, "prompt": c.system_prompt,
                 "pair_idx": c.pair_idx}
                for c in build_cells(cond, cat, user_bank, model_bank,
                                     generic_dimensions, None)
            ]
        cells[cat] = per_cond

    # The banks themselves, side by side.
    banks = {}
    for cat in categories:
        rows = []
        domain = CATEGORY_TO_DOMAIN[cat]
        for level in LEVELS:
            u = user_bank.domain_by_name[domain][level]
            m = model_bank.entries("domain", level, domain)
            for i, (ue, me) in enumerate(zip(u, m)):
                rows.append({"block": "domain", "dimension": domain, "level": level,
                             "idx": i, "user": ue["system_prompt_snippet"],
                             "model": me["system_prompt_snippet"],
                             "label": ue.get("label"), "mirrored": True})
        for dim in generic_dimensions:
            for level in LEVELS:
                u = user_bank.generic_by_dim[dim][level]
                m = model_bank.entries("irrel", level, dim)
                for i, (ue, me) in enumerate(zip(u, m)):
                    rows.append({"block": "irrel", "dimension": dim, "level": level,
                                 "idx": i, "user": ue["system_prompt_snippet"],
                                 "model": me["system_prompt_snippet"],
                                 "label": ue.get("label"),
                                 "mirrored": model_bank.generic[dim].get("source")
                                             != "override"})
        banks[cat] = rows

    n_pairs = {c: {k: len(v) for k, v in cells[c].items()} for c in categories}
    return {
        "categories": categories, "conditions": EXP2_CONDITIONS,
        "condLabel": EXP2_LABEL_FLAT, "blocks": BLOCKS, "levels": LEVELS,
        "dimensions": list(generic_dimensions),
        "cells": cells, "banks": banks, "nPairs": n_pairs,
    }


def build(out: Path, generic_dimensions: list[str] | None = None) -> Path:
    dims = generic_dimensions or ALL_GENERIC_DIMENSIONS
    payload = collect(StatusBank(generic_dimensions=dims), ModelStatusBank(), dims)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_TEMPLATE.replace("__DATA__", json.dumps(payload,
                                                           separators=(",", ":"))))
    print(f"Roles dashboard -> {out}  ({out.stat().st_size/1000:.0f} KB)")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp2 roles</title>
<style>
:root{color-scheme:light;--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e2e1de;--card:#fff;
--user:#1b6b5e;--model:#9e3d52;--code:#f4f4f2}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif;
background:var(--bg);color:var(--fg)}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px}
nav{display:flex;gap:2px;padding:0 24px;border-bottom:1px solid var(--line);
background:var(--card);position:sticky;top:0;z-index:10}
nav button{border:0;background:none;color:var(--mut);padding:11px 14px;cursor:pointer;
font:inherit;border-bottom:2px solid transparent}
nav button.on{color:var(--fg);border-bottom-color:var(--fg);font-weight:600}
main{padding:22px 24px;max-width:1450px}
.controls{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
select{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:7px;
background:var(--card);color:var(--fg)}
h2{font-size:14px;margin:22px 0 9px;letter-spacing:.02em;text-transform:uppercase;
color:var(--mut)}
h2:first-child{margin-top:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:13px 15px;margin-bottom:10px}
.role{display:grid;grid-template-columns:230px 1fr;gap:14px;padding:7px 0;
border-bottom:1px solid var(--line);align-items:start}
.role:last-child{border-bottom:0}
.tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;
display:inline-block;white-space:nowrap}
.u{background:color-mix(in srgb,var(--user) 16%,transparent);color:var(--user)}
.m{background:color-mix(in srgb,var(--model) 16%,transparent);color:var(--model)}
.lab{font-weight:600;font-size:13px}
.dim{color:var(--mut);font-size:11px;font-family:ui-monospace,Menlo,monospace}
.clause{color:var(--fg);font-size:13px}
.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:12.5px;table-layout:fixed}
th,td{text-align:left;padding:9px 10px;border:1px solid var(--line);vertical-align:top}
th{background:var(--code);font-size:11px;text-transform:uppercase;letter-spacing:.05em;
color:var(--mut);font-weight:600}
th.rh{width:150px}
td .who{font-size:10.5px;color:var(--mut);margin-bottom:4px;font-family:ui-monospace,Menlo,monospace}
.stat{display:inline-block;margin-right:20px}
.stat b{font-size:20px;display:block;letter-spacing:-.02em}
.stat span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.note{color:var(--mut);font-size:12.5px;margin:10px 0 18px;line-height:1.6}
.wrap{overflow-x:auto}
</style></head><body>
<header><h1>Exp2 — user status × model status</h1>
<div class="sub" id="sub"></div></header>
<nav id="nav"></nav><main id="main"></main>
<script>
const D = __DATA__;
const E=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let cat = D.categories.includes("Child") ? "Child" : D.categories[0];
let tab = "grid";
let dim = D.dimensions[0];

const nPairs = Object.values(D.nPairs[cat]).reduce((a,b)=>a+b,0);
document.getElementById("sub").textContent =
 `${D.conditions.length} conditions · control + 2 relevance blocks × user{high,low} × model{high,low} · ${D.categories.length} safety categories`;

const TABS=[["grid","Condition grid"],["prompts","Composed prompts"],["banks","Role banks"]];
const nav=document.getElementById("nav");
TABS.forEach(([id,label])=>{const b=document.createElement("button");
 b.textContent=label;b.dataset.id=id;b.onclick=()=>{tab=id;draw()};nav.appendChild(b)});

function picker(extra){
  let h=`<div class="controls"><label class="mut">Category</label>
   <select id="cat">${D.categories.map(c=>`<option ${c===cat?"selected":""}>${E(c)}</option>`).join("")}</select>`;
  if(extra) h+=`<label class="mut">Generic dimension</label>
   <select id="dim">${D.dimensions.map(d=>`<option ${d===dim?"selected":""}>${E(d)}</option>`).join("")}</select>`;
  return h+`</div>`;
}

// ── 2x2 grid per block, plus control ─────────────────────────────────────────
function gridTab(){
  let h=picker(true);
  h+=`<div class="note">Relevance is a property of the <b>block</b>: inside <em>domain</em> both sides draw from this category's expertise bank, inside <em>irrelevant</em> both draw from the same generic status dimension. Rows are the user's level, columns the assistant's.</div>`;
  for(const block of D.blocks){
    const title = block==="domain" ? `Domain-relevant &mdash; ${E(cat)}` : `Domain-irrelevant &mdash; ${E(dim)}`;
    h+=`<h2>${title}</h2><div class="wrap"><table><thead><tr><th class="rh"></th>`;
    for(const m of D.levels) h+=`<th><span class="tag m">model ${m}</span></th>`;
    h+=`</tr></thead><tbody>`;
    for(const u of D.levels){
      h+=`<tr><th class="rh"><span class="tag u">user ${u}</span></th>`;
      for(const m of D.levels){
        const cond=`${block}_u${u}_m${m}`;
        const all=(D.cells[cat][cond]||[]).filter(c=>block==="domain"||c.dimension===dim);
        const c=all[0];
        h+=`<td>${c?`<div class="who">${E(c.model_label||"—")} &nbsp;/&nbsp; ${E(c.user_label||"—")}</div>${c.prompt?E(c.prompt):'<i class="mut">no system message</i>'}<div class="who" style="margin-top:6px">${all.length} paired role${all.length===1?"":"s"} in this cell</div>`:"—"}</td>`;
      }
      h+=`</tr>`;
    }
    h+=`</tbody></table></div>`;
  }
  const ctl=(D.cells[cat]["control"]||[])[0];
  h+=`<h2>Control</h2><div class="card"><div class="role"><div class="lab mut">no role either side</div><div class="clause"><i class="mut">no system message is sent at all &mdash; the user turn goes to the model on its own</i></div></div></div>`;
  return h;
}

// ── every composed prompt ────────────────────────────────────────────────────
function promptsTab(){
  let h=picker(true);
  h+=`<div class="note">Every paired role in each condition. Pairing is a <b>seeded random permutation</b>, and on same-level cells a derangement, so the mirrored banks never pair a persona with itself. A cell holds 5 pairs, not the 25 of a full cross.</div>`;
  for(const cond of D.conditions){
    const all=(D.cells[cat][cond]||[]).filter(c=>c.dimension==="none"||c.dimension===dim||!D.dimensions.includes(c.dimension));
    if(!all.length) continue;
    h+=`<h2>${E(D.condLabel[cond])} <span class="mut">${all.length} pair${all.length===1?"":"s"}</span></h2><div class="card">`;
    for(const c of all)
      h+=`<div class="role"><div><div class="lab">${E(c.model_label||"—")}</div><div class="dim">model</div><div class="lab" style="margin-top:6px">${E(c.user_label||"—")}</div><div class="dim">user</div></div><div class="clause">${c.prompt?E(c.prompt):'<i class="mut">no system message</i>'}</div></div>`;
    h+=`</div>`;
  }
  return h;
}

// ── the banks, user vs model side by side ────────────────────────────────────
function banksTab(){
  let h=picker(false);
  h+=`<div class="note">Model roles are the user roles re-voiced to second person, so both sides draw on one vocabulary and a high/high cell pairs comparable standings. <b>Exception:</b> <span class="mono">subscription_tier</span> has no second-person form — the assistant has no subscription — so its model roles are hand-authored as the assistant's own <em>product tier</em>, the closest analogue: platform-conferred standing that says nothing about competence.</div>`;
  const rows=D.banks[cat];
  let last="";
  for(const r of rows){
    const key=`${r.block}/${r.dimension}/${r.level}`;
    if(key!==last){
      if(last) h+=`</div>`;
      h+=`<h2><span class="tag ${r.block==="domain"?"u":"m"}">${E(r.block)}</span> ${E(r.dimension)} &mdash; ${E(r.level)}${r.mirrored===false?' <span class="mut" style="text-transform:none;font-weight:400">— model roles hand-authored, not mirrored</span>':''}</h2><div class="card">`;
      last=key;
    }
    h+=`<div class="role"><div><div class="lab">${E(r.label||"—")}</div><div class="dim">#${r.idx}</div></div>
      <div><div class="clause"><b class="mut">user</b> ${E(r.user)}</div>
      <div class="clause" style="margin-top:3px"><b class="mut">model</b> ${E(r.model)}</div></div></div>`;
  }
  return h+(last?`</div>`:"");
}

function draw(){
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.id===tab));
  document.getElementById("main").innerHTML =
    tab==="grid"?gridTab():tab==="prompts"?promptsTab():banksTab();
  const sc=document.getElementById("cat"); if(sc) sc.onchange=e=>{cat=e.target.value;draw()};
  const sd=document.getElementById("dim"); if(sd) sd.onchange=e=>{dim=e.target.value;draw()};
}
draw();
</script></body></html>
"""
