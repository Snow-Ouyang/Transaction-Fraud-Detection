"""Fixed-parameter behavior groups and previously rejected extensions."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import gc
import json
import os
import platform
import sys
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'category_ablation'
CAP=HERE.parent/'capacity_comparison'
sys.path.insert(0,str(CAP))
import run_comparison as engine
from run_core import sha256,write_json,evaluate
from model_io import load_model,predict,suffix
from features import FIELDS,NEW,behavior_features,audit
DATA=None


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def prepare():
    HERE.mkdir(exist_ok=True)
    for n in ['runs','tables','figures','qa']:(HERE/n).mkdir(exist_ok=True)
    if (HERE/'manifest.json').exists():check();return
    old=read(PRIOR/'protocol.json')
    p={k:old[k] for k in ['seed','categorical_features','catboost_fixed','xgboost_fixed','folds','utility']}
    p.update(experiment='v5_behavior_context_no_raw_category',workers=3,
        threshold='Each run/fold: exact utility optimum only on preceding threshold window; tied scores grouped, equal utility fewer flags.',
        group_rule='All groups independently added to the accepted no-category family baseline. Accept mean paired utility/10000 >0 AND wins >=2/3.',
        combined_rule='At least two individually accepted groups: fit their union once and apply same rule against baseline. No further search.',
        final_rule='Best equal-window mean delta utility/10000 among baseline, accepted single groups and accepted union; worst window and fewer inputs break ties.',
        june='Previously seen diagnostic only. Both family choices locked before scoring June; no reselection.',
        histories='Strictly earlier transactions only; same-second events mutually excluded; online unlabeled updates; no historical fraud labels.',
        category='Raw category remains excluded. A separate category x personal-history group tests two numeric past-only contexts; no target encoding.',
        official_test='No official test loading, prediction, upload, or submission.',
        limits='Same development windows repeatedly reused; promotion is research selection, not new independent validation.')
    p['behavior_groups']=[
        dict(id='velocity',name='一小时交易密度与商户分散',add=['count_1h','distinct_merchants_1h']),
        dict(id='recent_amount',name='七日消费水平与长期偏离',add=['amount_to_mean_7d','mean_7d_to_past_mean']),
        dict(id='hour_habit',name='个人交易时段习惯',add=['hour_unusualness']),
        dict(id='category_context',name='类别与个人历史交叉',add=['category_past_share','amount_to_shrunk_category_mean'])]
    p['legacy_groups']=[dict(id='age_weekday',name='重新检验：年龄与星期',add=['age','weekday']),
        dict(id='merchant',name='重新检验：商户名称',add=['merchant']),
        dict(id='identity',name='重新检验：职业/城市/邮编',add=['job','city','zip'])]
    p['feature_definitions']={
        'count_1h':'Customer transaction count in [t-3600,t).',
        'distinct_merchants_1h':'Customer distinct merchant count in [t-3600,t).',
        'amount_to_mean_7d':'Current amount / mean customer amount in [t-7days,t); NaN if no such history.',
        'mean_7d_to_past_mean':'Mean customer amount in [t-7days,t) / all-past customer mean; NaN if absent.',
        'hour_unusualness':'-log((past customer count at current hour+1)/(all past customer count+24)); hour from unix_time, no labels.',
        'category_past_share':'Past customer transactions at current category / all past customer transactions; NaN on cold start.',
        'amount_to_shrunk_category_mean':'Current amount / ((past customer-category amount sum+5*all-past customer mean)/(past customer-category count+5)); NaN on cold start. Fixed shrinkage5, not tuned.'}
    p['baselines']={}
    for family,rid,tag in [('catboost','CB5_no_category','CB'),('xgboost','X5_no_category','XGB')]:
        r=read(PRIOR/'runs'/rid/'result.json')
        groups=[g for g in p['behavior_groups']+p['legacy_groups'] if not set(g['add']).issubset(r['features'])]
        assert all(not set(g['add'])&set(r['features']) for g in groups)
        p['baselines'][family]=dict(source_run=rid,tag=tag,config=r['config'],features=r['features'],groups=groups)
    write_json(HERE/'protocol.json',p)
    print('Building seven strictly-past context features, including category-history interactions.',flush=True)
    d=pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl')
    events=d[FIELDS].copy();out=behavior_features(events)
    audit_result=audit(events,out,HERE/'qa')
    write_json(HERE/'qa/feature_validation.json',audit_result)
    assert all(audit_result['checks'].values()),audit_result
    assert np.array_equal(d.ID,out.ID)
    for name in NEW:d[name]=out[name].to_numpy()
    for name in d.select_dtypes(include=['object','string']).columns:d[name]=d[name].astype('category')
    d.to_pickle(HERE/'feature_cache.pkl')
    paths=[HERE/'protocol.json',HERE/'run_context.py',HERE/'features.py',HERE/'feature_cache.pkl',HERE/'qa/feature_validation.json',
        CAP/'run_comparison.py',CAP/'model_io.py',HERE.parent/'core_baseline/run_core.py',
        HERE.parent/'catboost_progression/feature_cache.pkl',HERE.parent/'catboost_progression/features.py',
        HERE.parent/'catboost_progression/qa/feature_validation.json',ROOT/'fraudTrain.csv']
    for b in p['baselines'].values():paths += [x for x in (PRIOR/'runs'/b['source_run']).iterdir() if x.is_file()]
    write_json(HERE/'manifest.json',dict(started_utc=engine.stamp(),
        hashes={str(x.relative_to(ROOT)):sha256(x) for x in paths},
        runtime=dict(python=platform.python_version(),executable=sys.executable,catboost=engine.catboost.__version__,xgboost=engine.xgboost.__version__,pandas=pd.__version__,numpy=np.__version__),
        data_rows=len(d),feature_cache_memory_mib=d.memory_usage(deep=True).sum()/2**20))
    check()
    print(f'Feature audit passed: {audit_result["independent_spots"]} direct spots, {audit_result["prefix_rows"]:,} prefix rows; protocol frozen.',flush=True)


def check():
    m=read(HERE/'manifest.json')
    assert all(sha256(ROOT/k)==v for k,v in m['hashes'].items()),'Frozen source/code/cache changed'
    assert all(read(HERE/'qa/feature_validation.json')['checks'].values())
    engine.HERE=HERE;engine.P=read(HERE/'protocol.json')
    return engine.P


def initialize():
    global DATA
    engine.HERE=HERE;engine.P=read(HERE/'protocol.json')
    DATA=pd.read_pickle(HERE/'feature_cache.pkl')
    assert DATA.unix_time.is_monotonic_increasing


def fit_job(job):
    family,rid,config,features=job
    assert 'category' not in features
    result=engine.fit_run(family,rid,config,features,DATA)
    gc.collect()
    return result['run_id']


def jobs_parallel(jobs,workers=3):
    if not jobs:return
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        pending=[pool.submit(fit_job,j) for j in jobs]
        for future in as_completed(pending):print('COMPLETE '+future.result(),flush=True)


def groups():
    p=check();families=list(p['baselines']);jobs=[]
    # Interleave families; each fit retains the original four-thread setting.
    for i in range(max(len(b['groups']) for b in p['baselines'].values())):
        for family in families:
            b=p['baselines'][family]
            if i<len(b['groups']):
                g=b['groups'][i];rid=b['tag']+'__'+g['id']
                jobs.append((family,rid,b['config'],b['features']+g['add']))
    print(f'{len(jobs)} independent candidate groups, {3*len(jobs)} new fits; no hyperparameter changes.',flush=True)
    jobs_parallel(jobs,p['workers'])


def select():
    p=check();decisions={};union_jobs=[]
    for family,b in p['baselines'].items():
        baseline=read(PRIOR/'runs'/b['source_run']/'result.json');rows=[];accepted=[]
        for g in b['groups']:
            rid=b['tag']+'__'+g['id'];r=read(HERE/'runs'/rid/'result.json')
            change=engine.paired(r,baseline)
            rows.append(dict(run_id=rid,group=g,**change))
            if change['accepted']:accepted.append(g)
        decisions[family]=dict(baseline=b['source_run'],groups=rows,accepted_groups=[g['id'] for g in accepted])
        if len(accepted)>=2:
            features=b['features']+[c for g in accepted for c in g['add']]
            union_jobs.append((family,b['tag']+'__combined',b['config'],features))
    lockpath=HERE/'group_lock.json'
    if lockpath.exists():assert read(lockpath)['decisions']==decisions
    else:write_json(lockpath,dict(locked_utc=engine.stamp(),decisions=decisions,manifest_sha256=sha256(HERE/'manifest.json')))
    jobs_parallel(union_jobs,min(2,len(union_jobs)))
    selections={}
    for family,b in p['baselines'].items():
        baseline=read(PRIOR/'runs'/b['source_run']/'result.json');eligible=[baseline];decision=decisions[family]
        for row in decision['groups']:
            if row['accepted']:eligible.append(read(HERE/'runs'/row['run_id']/'result.json'))
        combined=None
        if len(decision['accepted_groups'])>=2:
            r=read(HERE/'runs'/(b['tag']+'__combined')/'result.json');combined=engine.paired(r,baseline)
            if combined['accepted']:eligible.append(r)
        winner=sorted(eligible,key=engine.rank)[0]
        selections[family]=dict(**decision,combined=combined,final_run=winner['run_id'],final_features=winner['features'],
            eligible_runs=[r['run_id'] for r in eligible],mean_delta_per_10000=winner['mean_delta_per_10000'])
    path=HERE/'selection_lock.json'
    if path.exists():assert read(path)['selections']==selections
    else:write_json(path,dict(locked_utc=engine.stamp(),selections=selections,
        group_lock_sha256=sha256(lockpath),manifest_sha256=sha256(HERE/'manifest.json'),june_not_used_for_selection=True))
    print(json.dumps(selections,ensure_ascii=False,indent=2),flush=True)


def finalize():
    p=check();lock=read(HERE/'selection_lock.json');lockhash=sha256(HERE/'selection_lock.json')
    data=pd.read_pickle(HERE/'feature_cache.pkl');june=data.loc[data.timestamp.ge('2025-06-01')&data.timestamp.lt('2025-07-01')]
    runs=[]
    for family,b in p['baselines'].items():
        ids=[b['tag']+'__'+g['id'] for g in b['groups']]
        if len(lock['selections'][family]['accepted_groups'])>=2:ids.append(b['tag']+'__combined')
        for rid in ids:
            folder=HERE/'runs'/rid;run=read(folder/'result.json');path=folder/'June_result.json'
            if path.exists():assert read(path)['selection_lock_sha256']==lockhash
            else:
                model=load_model(folder/('D3'+suffix(family)),family)
                score=predict(model,june,read(folder/'D3_schema.json'),family)
                t=run['folds'][2]['threshold_decision']['threshold']
                out=june[['ID','unix_time','is_fraud']].copy();out['score']=score;out['decision']=(score>=t).astype('int8')
                out.to_csv(folder/'June_predictions.csv.gz',index=False,compression='gzip')
                write_json(path,dict(run_id=rid,role='reused_diagnostic_not_blind_test',selection_lock_sha256=lockhash,
                    selected=evaluate(june.is_fraud,score,t)))
            runs.append(rid)
    write_json(HERE/'results.json',dict(selections=lock['selections'],run_ids=runs,new_fits=3*len(runs),
        completed_utc=engine.stamp(),manifest_sha256=sha256(HERE/'manifest.json'),selection_lock_sha256=lockhash))
    print(f'Completed {3*len(runs)} fits; both family decisions locked before reused June diagnostics.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','groups','select','finalize'])
    stage=parser.parse_args().stage
    {'prepare':prepare,'groups':groups,'select':select,'finalize':finalize}[stage]()
