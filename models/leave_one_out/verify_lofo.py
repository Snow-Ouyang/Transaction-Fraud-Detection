"""Lean independent score/threshold/selection checks; no fitting or figures."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'behavior_context'
sys.path.insert(0,str(HERE.parent/'core_baseline/.deps'))
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=read(HERE/'protocol.json');manifest=read(HERE/'manifest.json');res=read(HERE/'results.json')
    lock=read(HERE/'selection_lock.json');single_lock=read(HERE/'singles_lock.json')
    checks={};row_count=0;metrics={};actual_runs={}
    def put(k,v):checks[k]=bool(v)
    put('frozen_inputs_and_training_code',all(digest(ROOT/k)==v for k,v in manifest['hashes'].items()))
    put('manifest_lock_binding',res['manifest_sha256']==lock['manifest_sha256']==digest(HERE/'manifest.json'))
    put('selection_lock_binding',res['selection_lock_sha256']==digest(HERE/'selection_lock.json'))
    put('single_lock_binding',lock['singles_lock_sha256']==digest(HERE/'singles_lock.json'))
    data=pd.read_pickle(PRIOR/'feature_cache.pkl').set_index('ID',drop=False)
    raw=pd.read_csv(ROOT/'fraudTrain.csv',usecols=['ID','unix_time','is_fraud']).set_index('ID')
    put('cached_raw_labels_times',all(np.array_equal(data[c],raw.loc[data.ID,c]) for c in ['unix_time','is_fraud']))
    jobs=p['jobs']+single_lock['joint_jobs']
    entries=[dict(run_id=b['source_run'],family=f,features=b['features'],config=b['config'],kind='baseline',dropped=[]) for f,b in p['baselines'].items()]+jobs
    forbidden={'ID','unix_time','timestamp','customer_key','is_fraud','category','first','last','dob','street','cc_num'}
    selected_ids={s[k] for s in res['selections'].values() for k in ['score_leader','robust_leader']}
    for job in entries:
        rid=job['run_id'];family=job['family'];b=p['baselines'][family]
        folder=(PRIOR if job['kind']=='baseline' else HERE)/'runs'/rid
        r=read(folder/'result.json');actual_runs[rid]=r
        put(rid+'_exact_features',r['features']==job['features']==[c for c in b['features'] if c not in job['dropped']])
        put(rid+'_fixed_model_parameters',r['config']==b['config'])
        put(rid+'_no_identifier_label_or_raw_category_inputs',not bool(set(r['features'])&forbidden))
        for fold,fp in zip(r['folds'],p['folds']):
            f=fold['fold'];prefix=rid+'_'+f
            schema=read(folder/f'{f}_schema.json');t=fold['threshold_decision']['threshold']
            fit=data.loc[data.timestamp.lt(fp['fit_end'])]
            put(prefix+'_fit_rows',len(fit)==fold['fit_n'] and int(fit.is_fraud.sum())==fold['fit_fraud'])
            put(prefix+'_schema_hash',digest(folder/f'{f}_schema.json')==fold['schema_sha256'])
            put(prefix+'_schema_features',schema['features']==job['features'])
            put(prefix+'_fit_only_category_levels',schema['categorical']=={c:sorted(fit[c].astype(str).unique().tolist()) for c in job['features'] if c in p['categorical_features']})
            model=CatBoostClassifier() if family=='catboost' else XGBClassifier()
            modelpath=folder/(f+('.cbm' if family=='catboost' else '.ubj'));model.load_model(str(modelpath))
            put(prefix+'_model_hash',digest(modelpath)==fold['model_sha256'])
            put(prefix+'_model_feature_names',list(model.feature_names_ if family=='catboost' else model.get_booster().feature_names)==job['features'])
            expected={**p[family+'_fixed'],**{k:v for k,v in b['config'].items() if k!='id'},'random_seed' if family=='catboost' else 'random_state':p['seed']}
            put(prefix+'_fit_parameter_record',read(folder/f'{f}_model.json')['parameters']==expected)
            if family=='catboost':
                params=model.get_all_params()
                put(prefix+'_saved_tree_budget_depth_L2_seed',model.tree_count_==b['config']['iterations'] and params['depth']==b['config']['depth'] and np.isclose(params['l2_leaf_reg'],b['config']['l2_leaf_reg']) and params['random_seed']==p['seed'])
            else:put(prefix+'_saved_tree_budget',model.get_booster().num_boosted_rounds()==b['config']['n_estimators'])
            roles=[('threshold',f,f'{f}_threshold.csv.gz',fp['fit_end'],fp['threshold_end'],fold['threshold_metrics']),
                   ('development',f,f'{f}_development.csv.gz',fp['dev_start'],fp['dev_end'],fold['development_metrics'])]
            if f=='D3':roles.append(('diagnostic','June','June_predictions.csv.gz','2025-06-01','2025-07-01',read(folder/'June_result.json')['selected']))
            for role,window,filename,start,end,record in roles:
                pred=pd.read_csv(folder/filename,float_precision='round_trip');row_count+=len(pred);key=prefix+'_'+role
                put(key+'_complete_window',np.array_equal(pred.ID,data.loc[data.timestamp.ge(start)&data.timestamp.lt(end),'ID']))
                put(key+'_raw_label_time_alignment',np.array_equal(pred.is_fraud,raw.loc[pred.ID,'is_fraud']) and np.array_equal(pred.unix_time,raw.loc[pred.ID,'unix_time']))
                put(key+'_finite_valid_scores',np.isfinite(pred.score).all() and pred.score.between(0,1).all())
                put(key+'_threshold_decisions',np.array_equal(pred.decision,pred.score.ge(t).astype(int)))
                y=pred.is_fraud.to_numpy();d=pred.decision.to_numpy(bool)
                tp=int(((y==1)&d).sum());fpn=int(((y==0)&d).sum());fn=int(((y==1)&~d).sum());tn=int(((y==0)&~d).sum())
                score=50*tp-100*fn-5*fpn
                put(key+'_counts_and_official_score',tuple(record[c] for c in ['TP','FP','FN','TN','utility'])==(tp,fpn,fn,tn,score))
                put(key+'_normalized_score',np.isclose(record['utility_per_10000'],score/len(pred)*10000))
                metrics[rid,window,role]=dict(score=score,n=len(pred),per10000=score/len(pred)*10000)
                if role=='threshold':
                    q=pred.assign(gain=np.where(y==1,150,-5)).groupby('score').agg(gain=('gain','sum'),n=('gain','size')).sort_index(ascending=False)
                    gains=q.gain.cumsum();best=max(0,int(gains.max()));flags=0 if best==0 else int(q.n.cumsum().loc[gains.eq(best)].iloc[0])
                    put(key+'_exact_optimum_and_tie_rule',score==-100*int(y.sum())+best and int(d.sum())==flags)
                if role=='diagnostic' and job['kind']!='baseline':
                    put(key+'_selection_lock',read(folder/'June_result.json')['selection_lock_sha256']==digest(HERE/'selection_lock.json'))
                    put(key+'_generated_after_selection',(folder/filename).stat().st_mtime_ns>=(HERE/'selection_lock.json').stat().st_mtime_ns)
                ix=np.arange(len(pred)) if role=='diagnostic' and rid in selected_ids else np.linspace(0,len(pred)-1,min(100,len(pred)),dtype=int)
                x=data.loc[pred.ID.iloc[ix],schema['features']].copy()
                for col,levels in schema['categorical'].items():x[col]=pd.Categorical(x[col].astype(str),categories=levels) if family=='xgboost' else x[col].astype(str)
                put(key+'_saved_model_rescore',np.allclose(model.predict_proba(x)[:,1],pred.score.iloc[ix],rtol=0,atol=1e-12))
    def difference(rid,baseline):
        v=[(metrics[rid,w,'development']['score']-metrics[baseline,w,'development']['score'])/metrics[rid,w,'development']['n']*10000 for w in ['D1','D2','D3']]
        avg=float(np.mean(v));wins=sum(x>1e-9 for x in v)
        return avg,wins
    def rank(rid):
        v=[metrics[rid,w,'development']['per10000'] for w in ['D1','D2','D3']]
        return (-float(np.mean(v)),-float(min(v)),len(actual_runs[rid]['features']),rid)
    for family,b in p['baselines'].items():
        rows=[j for j in p['jobs'] if j['family']==family]
        expected_sets={}
        for criterion in ['mean_positive','robust_positive']:
            dropped=[]
            for j in rows:
                avg,wins=difference(j['run_id'],b['source_run'])
                if avg>1e-9 and (criterion=='mean_positive' or wins>=2):dropped+=j['dropped']
            expected_sets[criterion]=dropped
            put(family+'_'+criterion+'_joint_set',single_lock['joint_sets'][family][criterion]['dropped']==dropped)
        put(family+'_joint_sets_deduplicated',len([j for j in single_lock['joint_jobs'] if j['family']==family])==len({tuple(v) for v in expected_sets.values() if len(v)>=2}))
        selected=res['selections'][family];ids=[b['source_run']];robust=[b['source_run']]
        for j in [j for j in jobs if j['family']==family]:
            avg,wins=difference(j['run_id'],b['source_run']);ids.append(j['run_id'])
            if avg>1e-9 and wins>=2:robust.append(j['run_id'])
            rec=next(z for z in selected['candidates'] if z['run_id']==j['run_id'])
            put(j['run_id']+'_score_gain_stability_flags',np.isclose(avg,rec['mean_change_per_10000']) and wins==rec['improved_windows'] and (avg>1e-9)==rec['mean_positive'] and (avg>1e-9 and wins>=2)==rec['robust_positive'])
        put(family+'_primary_score_leader',min(ids,key=rank)==selected['score_leader'])
        put(family+'_robust_score_leader',min(robust,key=rank)==selected['robust_leader'])
    if (HERE/'pooled_view_lock.json').exists():
        pooled=read(HERE/'pooled_view_lock.json')
        for family,b in p['baselines'].items():
            ids=[b['source_run']]+[j['run_id'] for j in jobs if j['family']==family]
            leader=min(ids,key=lambda rid:(-sum(metrics[rid,w,'development']['score'] for w in ['D1','D2','D3']),-np.mean([metrics[rid,w,'development']['per10000'] for w in ['D1','D2','D3']]),len(actual_runs[rid]['features']),rid))
            put(family+'_pooled_total_score_leader',leader==pooled[family]['pooled_leader']['run_id'])
        put('pooled_view_recorded_before_new_june',all((HERE/'pooled_view_lock.json').stat().st_mtime_ns <= (HERE/'runs'/j['run_id']/'June_predictions.csv.gz').stat().st_mtime_ns for j in jobs))
    result=dict(all_passed=all(checks.values()),total_checks=len(checks),passed=sum(checks.values()),failed=[k for k,v in checks.items() if not v],prediction_rows_checked=row_count,no_fits=True,checks=checks)
    (HERE/'qa/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False),flush=True)
    assert result['all_passed'],result['failed']


if __name__=='__main__':main()
