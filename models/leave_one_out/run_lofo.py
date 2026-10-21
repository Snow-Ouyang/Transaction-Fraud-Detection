"""Fixed-parameter leave-one-feature-out and bounded joint-drop robustness check."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import gc
import json
import platform
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'behavior_context'
CAP=HERE.parent/'capacity_comparison'
sys.path.insert(0,str(CAP))
import run_comparison as engine
from run_core import sha256,write_json,evaluate
from model_io import load_model,predict,suffix
DATA=None


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    for name in ['runs','qa']:(HERE/name).mkdir(parents=True,exist_ok=True)
    if (HERE/'manifest.json').exists():check();return
    previous=read(PRIOR/'protocol.json');results=read(PRIOR/'results.json')
    p={k:previous[k] for k in ['seed','categorical_features','catboost_fixed','xgboost_fixed','folds','utility']}
    p.update(experiment='v6_leave_one_input_out',workers=4,
        primary='Equal-window mean absolute score/10000, higher is better. Also report pooled total score.',
        stability='Flag robust improvement only if mean paired score gain>0 and >=2/3 windows improve. Report mean-positive but mixed cases too.',
        joint='After all singles: one union of all mean-positive drops, and one union of robust-positive drops. Only >=2 drops; identical sets deduplicated; singleton sets reuse their leave-one-out fit. No further subsets or iterative elimination.',
        selection='Primary score leader among baseline, every single drop and predeclared joint drops. Separately report best robust-improvement candidate. No auxiliary metric veto.',
        thresholds='Per run/fold exact score maximum on preceding calibration window only; equal optima fewer flags; never optimize on development or June.',
        interpretation='Delete model input columns only. Other derived features remain unchanged, even if they encode related information. This is input-representation robustness, not deletion of the raw information source.',
        june='Previously observed diagnostic only; new June predictions only after all score choices locked. No reselection.',
        official_test='No loading, predictions, submission or upload.',
        output='Compact Markdown results and necessary model/parameter/prediction logs. No HTML or visualization.')
    p['baselines']={};p['jobs']=[]
    for family,tag in [('catboost','CB'),('xgboost','XGB')]:
        rid=results['selections'][family]['final_run'];r=read(PRIOR/'runs'/rid/'result.json')
        p['baselines'][family]=dict(source_run=rid,tag=tag,features=r['features'],config=r['config'])
        for name in r['features']:
            p['jobs'].append(dict(run_id=tag+'__drop__'+name,family=family,dropped=[name],
                config=r['config'],features=[c for c in r['features'] if c!=name],kind='single'))
    write_json(HERE/'protocol.json',p)
    paths=[HERE/'protocol.json',HERE/'run_lofo.py',CAP/'run_comparison.py',CAP/'model_io.py',
        HERE.parent/'core_baseline/run_core.py',PRIOR/'feature_cache.pkl',PRIOR/'features.py',
        PRIOR/'qa/feature_validation.json',PRIOR/'qa/independent_validation.json',ROOT/'fraudTrain.csv']
    for b in p['baselines'].values():paths += [x for x in (PRIOR/'runs'/b['source_run']).iterdir() if x.is_file()]
    write_json(HERE/'manifest.json',dict(started_utc=engine.stamp(),
        hashes={str(x.relative_to(ROOT)):sha256(x) for x in paths},
        runtime=dict(python=platform.python_version(),executable=sys.executable,catboost=engine.catboost.__version__,xgboost=engine.xgboost.__version__)))
    check();refresh('prepared')
    print(f'Frozen {len(p["jobs"])} leave-one-out branches, {3*len(p["jobs"])} initial fits.',flush=True)


def check():
    m=read(HERE/'manifest.json')
    assert all(sha256(ROOT/k)==v for k,v in m['hashes'].items()),'Frozen code/input/baseline changed'
    assert read(PRIOR/'qa/independent_validation.json')['all_passed']
    assert all(read(PRIOR/'qa/feature_validation.json')['checks'].values())
    engine.HERE=HERE;engine.P=read(HERE/'protocol.json')
    return engine.P


def initialize():
    global DATA
    engine.HERE=HERE;engine.P=read(HERE/'protocol.json')
    DATA=pd.read_pickle(PRIOR/'feature_cache.pkl')


def fit_job(job):
    r=engine.fit_run(job['family'],job['run_id'],job['config'],job['features'],DATA)
    gc.collect()
    return r['run_id']


def parallel(jobs,workers):
    if not jobs:return
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        pending=[pool.submit(fit_job,j) for j in jobs]
        for future in as_completed(pending):
            rid=future.result();print('COMPLETE '+rid,flush=True);refresh('running')


def singles():
    p=check();families=list(p['baselines']);parts={f:[j for j in p['jobs'] if j['family']==f] for f in families};jobs=[]
    for i in range(max(map(len,parts.values()))):
        for family in families:
            if i<len(parts[family]):jobs.append(parts[family][i])
    parallel(jobs,p['workers']);refresh('singles complete')


def comparison(run,parent):
    changes={a['fold']:(a['development_metrics']['utility']-b['development_metrics']['utility'])/a['development_metrics']['n']*10000 for a,b in zip(run['folds'],parent['folds'])}
    avg=float(np.mean(list(changes.values())));wins=sum(x>1e-9 for x in changes.values())
    return dict(changes_per_10000=changes,mean_change_per_10000=avg,improved_windows=wins,
        mean_positive=avg>1e-9,robust_positive=avg>1e-9 and wins>=2,
        pooled_score_change=sum(f['development_metrics']['utility'] for f in run['folds'])-sum(f['development_metrics']['utility'] for f in parent['folds']))


def rank(run):
    values=[f['development_metrics']['utility_per_10000'] for f in run['folds']]
    return (-float(np.mean(values)),-float(min(values)),len(run['features']),run['run_id'])


def select():
    p=check();single_rows=[];jobs=[];joint_sets={}
    for family,b in p['baselines'].items():
        parent=read(PRIOR/'runs'/b['source_run']/'result.json');rows=[]
        for job in [j for j in p['jobs'] if j['family']==family]:
            r=read(HERE/'runs'/job['run_id']/'result.json');rows.append(dict(**job,**comparison(r,parent)))
        single_rows+=rows;joint_sets[family]={};seen={}
        for criterion in ['mean_positive','robust_positive']:
            dropped=[row['dropped'][0] for row in rows if row[criterion]]
            if len(dropped)<2:
                joint_sets[family][criterion]=dict(dropped=dropped,run_id=(b['tag']+'__drop__'+dropped[0]) if dropped else b['source_run'],new_fit=False)
                continue
            key=tuple(dropped)
            if key not in seen:
                rid=b['tag']+'__joint__'+criterion;seen[key]=rid
                jobs.append(dict(run_id=rid,family=family,kind='joint',dropped=dropped,
                    config=b['config'],features=[c for c in b['features'] if c not in dropped]))
            joint_sets[family][criterion]=dict(dropped=dropped,run_id=seen[key],new_fit=True)
    path=HERE/'singles_lock.json'
    record=dict(single_rows=single_rows,joint_sets=joint_sets,joint_jobs=jobs,manifest_sha256=sha256(HERE/'manifest.json'))
    if path.exists():assert all(read(path)[k]==v for k,v in record.items())
    else:write_json(path,dict(locked_utc=engine.stamp(),**record))
    parallel(jobs,min(p['workers'],max(1,len(jobs))))
    selections={}
    for family,b in p['baselines'].items():
        parent=read(PRIOR/'runs'/b['source_run']/'result.json');trials=[parent];robust=[parent];rows=[]
        for job in [j for j in p['jobs']+jobs if j['family']==family]:
            r=read(HERE/'runs'/job['run_id']/'result.json');change=comparison(r,parent)
            trials.append(r)
            if change['robust_positive']:robust.append(r)
            rows.append(dict(run_id=job['run_id'],kind=job['kind'],dropped=job['dropped'],**change))
        winner=min(trials,key=rank);stable=min(robust,key=rank)
        selections[family]=dict(score_leader=winner['run_id'],robust_leader=stable['run_id'],
            score_leader_features=winner['features'],robust_leader_features=stable['features'],candidates=rows)
    output=HERE/'selection_lock.json'
    if output.exists():assert read(output)['selections']==selections
    else:write_json(output,dict(locked_utc=engine.stamp(),selections=selections,singles_lock_sha256=sha256(path),manifest_sha256=sha256(HERE/'manifest.json'),june_not_used_for_selection=True))
    refresh('selection locked')
    print(json.dumps({f:{k:v for k,v in s.items() if k!='candidates'} for f,s in selections.items()},ensure_ascii=False,indent=2),flush=True)


def finalize():
    p=check();lock=read(HERE/'selection_lock.json');h=sha256(HERE/'selection_lock.json')
    jobs=p['jobs']+read(HERE/'singles_lock.json')['joint_jobs']
    data=pd.read_pickle(PRIOR/'feature_cache.pkl');june=data.loc[data.timestamp.ge('2025-06-01')&data.timestamp.lt('2025-07-01')]
    for job in jobs:
        rid=job['run_id'];folder=HERE/'runs'/rid;path=folder/'June_result.json'
        if path.exists():assert read(path)['selection_lock_sha256']==h;continue
        r=read(folder/'result.json');family=job['family'];model=load_model(folder/('D3'+suffix(family)),family)
        scores=predict(model,june,read(folder/'D3_schema.json'),family);t=r['folds'][2]['threshold_decision']['threshold']
        pred=june[['ID','unix_time','is_fraud']].copy();pred['score']=scores;pred['decision']=(scores>=t).astype('int8')
        pred.to_csv(folder/'June_predictions.csv.gz',index=False,compression='gzip')
        write_json(path,dict(run_id=rid,selection_lock_sha256=h,role='reused_diagnostic_not_blind_test',selected=evaluate(june.is_fraud,scores,t)))
    write_json(HERE/'results.json',dict(completed_utc=engine.stamp(),selections=lock['selections'],run_ids=[j['run_id'] for j in jobs],new_fits=3*len(jobs),selection_lock_sha256=h,manifest_sha256=sha256(HERE/'manifest.json')))
    refresh('complete; verification pending')


def refresh(stage):
    p=read(HERE/'protocol.json');jobs=list(p['jobs'])
    if (HERE/'singles_lock.json').exists():jobs+=read(HERE/'singles_lock.json')['joint_jobs']
    rows=[];baseline_rows=[]
    for family,b in p['baselines'].items():
        parent=read(PRIOR/'runs'/b['source_run']/'result.json');values=[f['development_metrics']['utility_per_10000'] for f in parent['folds']]
        baseline_rows.append(f'| {family} | {len(b["features"])} | {np.mean(values):,.2f} | {sum(f["development_metrics"]["utility"] for f in parent["folds"]):,} |')
        for job in [j for j in jobs if j['family']==family]:
            path=HERE/'runs'/job['run_id']/'result.json'
            if not path.exists():continue
            r=read(path);change=comparison(r,parent)
            rows.append(dict(family=family,run_id=job['run_id'],kind=job['kind'],dropped=job['dropped'],features=len(r['features']),**change))
    text=f'# Leave-one-out robustness check\n\nStatus: {stage}; complete candidate runs: {len(rows)}/{len(jobs)}.\n\n'
    text+='Fixed CatBoost16 / XGBoost18 baselines and model hyperparameters. Same D1/D2/D3 chronological folds and prior-window threshold selection. Primary = equal-window mean official score/10000. No figures or HTML.\n\n'
    text+='Input columns are removed; other derived features and historical inputs remain unchanged. A dropped raw amount/history column may still be represented through ratios. This is not proof that its underlying information is unnecessary.\n\n'
    text+='| Family | Baseline inputs | Mean score/10000 | Pooled development score |\n|---|---:|---:|---:|\n'+'\n'.join(baseline_rows)+'\n\n'
    text+='## All completed candidates\n\nPositive gain is reported even if concentrated in one window. Robust+ means mean gain>0 and >=2/3 windows improve. Delta units below are score/10000; pooled delta is total dollars.\n\n'
    text+='| Family | Removed inputs | Kind | D1 delta | D2 delta | D3 delta | Mean delta | Pooled delta | Wins | Robust+ |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|\n'
    for r in sorted(rows,key=lambda x:(x['family'],-x['mean_change_per_10000'])):
        vals=r['changes_per_10000'];text+='| '+r['family']+' | '+', '.join('`'+v+'`' for v in r['dropped'])+' | '+r['kind']+' | '+' | '.join(f'{vals[f]:+,.2f}' for f in ['D1','D2','D3'])+f' | {r["mean_change_per_10000"]:+,.2f} | {r["pooled_score_change"]:+,} | {r["improved_windows"]}/3 | '+('yes' if r['robust_positive'] else 'no')+' |\n'
    if (HERE/'selection_lock.json').exists():
        selections=read(HERE/'selection_lock.json')['selections'];text+='\n## Locked choices\n\n'
        for family,s in selections.items():
            text+=f'- {family}: score leader `{s["score_leader"]}`; robust leader `{s["robust_leader"]}`.\n'
        text+='\n## June reused diagnosis (not selection)\n\n| Family | Run | Score | FP | FN |\n|---|---|---:|---:|---:|\n'
        for family,b in p['baselines'].items():
            ids=list(dict.fromkeys([b['source_run'],selections[family]['score_leader'],selections[family]['robust_leader']]))
            for rid in ids:
                folder=(PRIOR if rid==b['source_run'] else HERE)/'runs'/rid;path=folder/'June_result.json'
                if path.exists():
                    m=read(path)['selected'];text+=f'| {family} | `{rid}` | {m["utility"]:,} | {m["FP"]:,} | {m["FN"]:,} |\n'
    if (HERE/'qa/verification.json').exists():
        q=read(HERE/'qa/verification.json');text+=f'\nVerification: {q["passed"]}/{q["total_checks"]} checks passed; {q["prediction_rows_checked"]:,} saved prediction rows checked.\n'
    text+='\nJoint drops test only the prespecified positive/robust-positive unions, not every combination or iterative backward elimination. Reused development windows imply selection optimism; June was previously observed. Official unlabeled test is not loaded or scored.\n'
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    write_json(HERE/'progress.json',dict(stage=stage,completed_runs=len(rows),planned_runs=len(jobs),updated_utc=engine.stamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','singles','select','finalize','refresh'])
    stage=parser.parse_args().stage
    {'prepare':prepare,'singles':singles,'select':select,'finalize':finalize,'refresh':lambda:refresh('complete')}[stage]()
