"""Export a local blind listening form alongside a measured speech comparison."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any


def write_listening_review(root: Path, report: dict[str, Any]) -> Path:
    names = [item["label"] for item in report["candidates"]]
    random.Random(7291).shuffle(names)
    labels = {name: chr(65 + index) for index, name in enumerate(names)}
    rows = [{"prompt": row["prompt_id"], "seed": row["seed"], "text": row["text"],
             "candidate": labels[row["checkpoint"]],
             "audio": Path(row["audio"]).resolve().relative_to(root.resolve()).as_posix()} for row in report["cells"]]
    data = json.dumps({"rows": rows, "identities": {code: name for name, code in labels.items()},
                       "dataset_identity": report["dataset_identity"]}, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speech listening comparison</title><style>
body{font:17px system-ui;margin:35px auto;max-width:1000px;padding:0 20px;background:#111827;color:#eef2ff} select,button{font:inherit;padding:8px;margin:5px;background:#25334e;color:white;border:1px solid #64748b;border-radius:6px} article{padding:16px;background:#1e293b;margin:14px 0;border-radius:12px} audio{display:block;width:100%;margin:10px 0} label{display:inline-block;margin-right:12px} #text{line-height:1.6} small{color:#cbd5e1}
</style><h1>Speech listening comparison</h1><p>Compare the same prompt and seed across candidates. Listen for correct words, complete endings, voice identity, pace and naturalness. Candidate identities stay hidden until you reveal them.</p>
<label>Prompt <select id="prompt"></select></label><label>Seed <select id="seed"></select></label>
<p id="text"></p><main id="clips"></main><button id="reveal">Reveal candidate identities</button><button id="save">Download listening ratings</button><p id="identities"></p>
<small>Ratings are your listening judgments. Automated measurements do not fill them in. Download the JSON to keep your ratings; this page does not send them anywhere.</small>
<script>const data=__DATA__;const ratings={};const prompt=document.querySelector('#prompt'),seed=document.querySelector('#seed');
function option(select,value){const item=document.createElement('option');item.value=value;item.textContent=value;select.append(item)}
[...new Set(data.rows.map(r=>r.prompt))].forEach(p=>option(prompt,p));[...new Set(data.rows.map(r=>r.seed))].forEach(s=>option(seed,s));
function show(){const rows=data.rows.filter(r=>r.prompt===prompt.value&&String(r.seed)===seed.value).sort((a,b)=>a.candidate.localeCompare(b.candidate));document.querySelector('#text').textContent=rows[0]?.text||'';const container=document.querySelector('#clips');container.replaceChildren();for(const row of rows){const article=document.createElement('article');const title=document.createElement('strong');title.textContent='Candidate '+row.candidate;article.append(title);const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=row.audio;article.append(audio);const key=[row.prompt,row.seed,row.candidate].join('|');for(const metric of ['Words and ending','Voice match','Naturalness']){const label=document.createElement('label');label.textContent=metric+' ';const score=document.createElement('select');option(score,'Unrated');for(let i=1;i<=5;i++)option(score,String(i));score.value=ratings[key]?.[metric]||'Unrated';score.onchange=()=>{ratings[key]??={prompt:row.prompt,seed:row.seed,candidate:row.candidate};ratings[key][metric]=score.value};label.append(score);article.append(label)}container.append(article)}}
prompt.onchange=seed.onchange=show;document.querySelector('#reveal').onclick=()=>{document.querySelector('#identities').textContent=Object.entries(data.identities).map(([code,name])=>code+': '+name).join(' · ')};
document.querySelector('#save').onclick=()=>{const blob=new Blob([JSON.stringify({dataset_identity:data.dataset_identity,identities:data.identities,ratings:Object.values(ratings),rated_at:new Date().toISOString()},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='listening_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};show();</script></html>""".replace("__DATA__", data)
    path = root / "listening_review.html"
    path.write_text(html, encoding="utf-8")
    return path
