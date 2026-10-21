"""Secondary pooled-score view of the same frozen candidates, no new fits."""
from pathlib import Path
import argparse
import json
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'behavior_context'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def records():
    p=read(HERE/'protocol.json');s=read(HERE/'selection_lock.json')['selections'];out={}
    for family,b in p['baselines'].items():
        rows=[]
        for rid in [b['source_run']]+[r['run_id'] for r in s[family]['candidates']]:
            folder=(PRIOR if rid==b['source_run'] else HERE)/'runs'/rid
            r=read(folder/'result.json');scores=[f['development_metrics']['utility'] for f in r['folds']]
            rows.append(dict(run_id=rid,dropped=[c for c in b['features'] if c not in r['features']],
                mean_score_per_10000=float(np.mean([f['development_metrics']['utility_per_10000'] for f in r['folds']])),
                pooled_score=sum(scores),features=len(r['features']),scores=scores))
        out[family]=rows
    return p,s,out


def lock():
    p,s,allrows=records();leaders={}
    for family,rows in allrows.items():
        best=min(rows,key=lambda r:(-r['pooled_score'],-r['mean_score_per_10000'],r['features'],r['run_id']))
        leaders[family]=dict(pooled_leader=best,original_equal_window_leader=s[family]['score_leader'])
    path=HERE/'pooled_view_lock.json'
    if path.exists():assert read(path)==leaders
    else:path.write_text(json.dumps(leaders,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(leaders,ensure_ascii=False,indent=2),flush=True)


def append():
    p,s,rows=records();locked=read(HERE/'pooled_view_lock.json')
    text='\n## Secondary total-score view\n\nSame frozen candidates, no new models or changed primary selection. Pooled leaders were recorded before generating June predictions. Different fold lengths make pooled total and equal-window mean genuinely different weights.\n\n'
    text+='| Family | View | Removed | Mean score/10000 | Total score | Total gain | Positive windows | June score |\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for family,b in p['baselines'].items():
        baseline=next(r for r in rows[family] if r['run_id']==b['source_run'])
        for label,rid in [('baseline',b['source_run']),('equal-window score leader',s[family]['score_leader']),('robust score leader',s[family]['robust_leader']),('pooled total score leader',locked[family]['pooled_leader']['run_id'])]:
            r=next(r for r in rows[family] if r['run_id']==rid)
            folder=(PRIOR if rid==b['source_run'] else HERE)/'runs'/rid
            june=read(folder/'June_result.json')['selected']['utility']
            wins=sum(a>z for a,z in zip(r['scores'],baseline['scores']))
            text+=f'| {family} | {label} | '+(', '.join('`'+x+'`' for x in r['dropped']) or 'none')+f' | {r["mean_score_per_10000"]:,.2f} | {r["pooled_score"]:,} | {r["pooled_score"]-baseline["pooled_score"]:+,} | {wins}/3 | {june:,} |\n'
    target=HERE/'RESULTS.md';body=target.read_text(encoding='utf-8').split('\n## Secondary total-score view')[0]
    target.write_text(body+text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['lock','append'],required=True)
    {'lock':lock,'append':append}[parser.parse_args().stage]()
