"""Independent accounting and saved-model verification; no fits or choices."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'core_baseline/.deps'))
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from PIL import Image


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def score(model,rows,schema,family):
    x=rows[schema['features']].copy()
    for name,levels in schema['categorical'].items():
        if family=='xgboost':x[name]=pd.Categorical(x[name],categories=levels)
        else:x[name]=x[name].astype(str)
    return model.predict_proba(x)[:,1].astype(float)


def confusion(y,d):
    y,d=np.asarray(y),np.asarray(d,bool)
    tp=int(((y==1)&d).sum());fp=int(((y==0)&d).sum());fn=int(((y==1)&~d).sum());tn=int(((y==0)&~d).sum())
    return tp,fp,fn,tn,50*tp-100*fn-5*fp


def ranking(run):
    c=run['config'];folds=run['folds']
    delta=[f['development_metrics']['delta_vs_always_normal']/f['development_metrics']['n']*10000 for f in folds]
    return (-float(np.mean(delta)),-min(delta),len(run['features']),c.get('depth',c.get('max_depth')),
            c.get('iterations',c.get('n_estimators')),-c.get('l2_leaf_reg',c.get('reg_lambda')),run['run_id'])


def change(a,b):
    aa={f['fold']:f['development_metrics'] for f in a['folds']}
    bb={f['fold']:f['development_metrics'] for f in b['folds']}
    v={fold:(aa[fold]['utility']-bb[fold]['utility'])/aa[fold]['n']*10000 for fold in aa}
    avg=float(np.mean(list(v.values())));wins=sum(x>1e-9 for x in v.values())
    return v,avg,wins,avg>1e-9 and wins>=2


def main():
    result=read(HERE/'results.json');manifest=read(HERE/'manifest.json');protocol=read(HERE/'protocol.json')
    lock=read(HERE/'selection_lock.json');checks={};count_rows=0
    checks['source_hashes_unchanged']=all(digest(ROOT/p)==h for p,h in manifest['source_hashes'].items())
    checks['frozen_code_hashes_unchanged']=all(digest(HERE/p)==h for p,h in manifest['code_hashes'].items())
    checks['manifest_binding']=digest(HERE/'manifest.json')==result['manifest_sha256']==lock['manifest_sha256']
    checks['total_selection_lock_hash']=digest(HERE/'selection_lock.json')==result['selection_lock_sha256']
    checks['v2_history_audit_unchanged_and_passed']=all(read(HERE.parent/'catboost_progression/qa/feature_validation.json')['checks'].values())
    checks['native_category_encoding_checks_passed']=all(read(HERE/'qa/encoding_checks.json').values())
    data=pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl').set_index('ID',drop=False)
    source=pd.read_csv(ROOT/'fraudTrain.csv',usecols=['ID','unix_time','amt','category','is_fraud','dob']).set_index('ID')
    aligned=source.loc[data.ID]
    checks['cache_all_raw_fields_and_labels_align']=all(np.array_equal(data[k],aligned[k]) for k in ['unix_time','amt','category','is_fraud'])
    checks['hour_independently_derived']=np.array_equal(data.hour,pd.to_datetime(data.unix_time,unit='s').dt.hour.astype(str))
    expected_age=(pd.to_datetime(data.unix_time,unit='s')-pd.to_datetime(aligned.dob)).dt.total_seconds()/(365.2425*86400)
    checks['age_independently_derived']=np.allclose(data.age,expected_age,rtol=0,atol=1e-10)
    checks['weekday_independently_derived']=np.array_equal(data.weekday,pd.to_datetime(data.unix_time,unit='s').dt.dayofweek.astype(str))
    checks['all_rows_unique_and_sorted']=data.ID.is_unique and data.unix_time.is_monotonic_increasing
    numeric=data.select_dtypes('number').drop(columns=['customer_key']).to_numpy(dtype=float)
    checks['no_infinite_numeric_values']=not np.isinf(numeric).any();del numeric
    runs={rid:read(HERE/'runs'/rid/'result.json') for rid in result['run_ids']}
    forbidden={'is_fraud','ID','unix_time','timestamp','customer_key','first','last','street','dob','cc_num'}
    def verify_prediction(run,fold,role,pred,recorded,threshold,model,schema,full_rescore=False):
        nonlocal count_rows
        rid=run['run_id'];prefix=f'{rid}_{fold}_{role}';count_rows+=len(pred)
        raw=source.loc[pred.ID]
        checks[prefix+'_raw_alignment']=np.array_equal(pred.is_fraud,raw.is_fraud) and np.array_equal(pred.unix_time,raw.unix_time)
        got=confusion(pred.is_fraud,pred.decision)
        checks[prefix+'_cost_and_counts']=got==tuple(int(recorded[k]) for k in ['TP','FP','FN','TN','utility'])
        baseline=-100*int(pred.is_fraud.sum());delta=got[-1]-baseline
        checks[prefix+'_baseline_and_normalization']=(recorded['n']==len(pred) and recorded['always_normal_utility']==baseline
            and recorded['delta_vs_always_normal']==delta and np.isclose(recorded['utility_per_10000'],got[-1]/len(pred)*10000)
            and np.isclose(recorded['delta_per_10000'],delta/len(pred)*10000))
        checks[prefix+'_threshold_decision']=np.array_equal(pred.score.ge(threshold),pred.decision.astype(bool))
        indices=np.arange(len(pred)) if full_rescore else np.linspace(0,len(pred)-1,min(150,len(pred)),dtype=int)
        p=score(model,data.loc[pred.ID.iloc[indices]],schema,run['family'])
        checks[prefix+'_model_rescore']=np.allclose(p,pred.score.iloc[indices],rtol=0,atol=1e-12)
        return prefix
    for rid,run in runs.items():
        family=run['family'];folder=HERE/'runs'/rid
        checks[rid+'_input_allowlist']=not bool(set(run['features'])&forbidden)
        for f,fp in zip(run['folds'],protocol['folds']):
            name=f['fold'];assert name==fp['name']
            mpath=folder/(name+('.cbm' if family=='catboost' else '.ubj'));schema=read(folder/f'{name}_schema.json')
            checks[rid+'_'+name+'_model_hash']=digest(mpath)==f['model_sha256']
            checks[rid+'_'+name+'_schema_hash']=digest(folder/f'{name}_schema.json')==f['schema_sha256']
            fit=data.loc[data.timestamp.lt(fp['fit_end'])]
            checks[rid+'_'+name+'_fit_size']=len(fit)==f['fit_n'] and int(fit.is_fraud.sum())==f['fit_fraud']
            checks[rid+'_'+name+'_vocabulary_fit_only']=all(levels==sorted(fit[c].astype(str).unique().tolist()) for c,levels in schema['categorical'].items())
            checks[rid+'_'+name+'_schema_names']=schema['features']==run['features']
            model=CatBoostClassifier() if family=='catboost' else XGBClassifier();model.load_model(str(mpath))
            names=model.feature_names_ if family=='catboost' else model.get_booster().feature_names
            checks[rid+'_'+name+'_model_names']=list(names)==run['features']
            expected_trees=run['config'].get('iterations',run['config'].get('n_estimators'))
            actual_trees=model.tree_count_ if family=='catboost' else model.get_booster().num_boosted_rounds()
            checks[rid+'_'+name+'_actual_tree_budget']=actual_trees==expected_trees==f['tree_count']
            if family=='catboost':
                params=model.get_all_params()
                checks[rid+'_'+name+'_actual_depth_and_L2']=(params['depth']==run['config']['depth'] and np.isclose(params['l2_leaf_reg'],run['config']['l2_leaf_reg']))
            if family=='xgboost':
                checks[rid+'_'+name+'_native_types']=model.get_booster().feature_types==['c' if c in schema['categorical'] else ('int' if pd.api.types.is_integer_dtype(data[c]) else 'float') for c in run['features']]
            decision=read(folder/f'{name}_threshold.json');t=decision['threshold']
            for role,key in [('threshold','threshold_metrics'),('development','development_metrics')]:
                pred=pd.read_csv(folder/f'{name}_{role}.csv.gz',float_precision='round_trip')
                prefix=verify_prediction(run,name,role,pred,f[key],t,model,schema)
                start=fp['fit_end'] if role=='threshold' else fp['dev_start']
                end=fp['threshold_end'] if role=='threshold' else fp['dev_end']
                expected=data.loc[data.timestamp.ge(start)&data.timestamp.lt(end),'ID']
                checks[prefix+'_complete_window_ids']=np.array_equal(pred.ID,expected)
                expected_unknown={c:int((~data.loc[pred.ID,c].astype(str).isin(levels)).sum()) for c,levels in schema['categorical'].items()}
                checks[prefix+'_unknown_counts']=expected_unknown==f['unknown_levels_threshold' if role=='threshold' else 'unknown_levels_development']
                if role=='threshold':
                    increments=pred.assign(gain=np.where(pred.is_fraud.eq(1),150,-5)).groupby('score').agg(
                        gain=('gain','sum'),size=('gain','size')).sort_index(ascending=False)
                    cum=increments.gain.cumsum();best=max(0,int(cum.max()))
                    flags=0 if best==0 else int(increments['size'].cumsum().loc[cum.eq(best)].iloc[0])
                    u=-100*int(pred.is_fraud.sum())+best
                    checks[prefix+'_independent_optimum']=u==decision['threshold_window_utility'] and flags==decision['flags']
                checks[prefix+'_decision_precedes_development']= (folder/f'{name}_threshold.json').stat().st_mtime_ns <= (folder/f'{name}_development.csv.gz').stat().st_mtime_ns
            del fit,model

    for family,selection in result['selections'].items():
        checks[family+'_family_lock_binding']=digest(HERE/f'{family}_selection.json')==lock['family_selection_hashes'][family]
        ranked=sorted([runs[c['id']] for c in protocol[family+'_grid']],key=ranking)
        checks[family+'_search_all_fixed_ten_inputs']=all(r['features']==protocol['base_features'] for r in ranked)
        checks[family+'_search_ranking']=selection['parameter_decision']['search_winner']==ranked[0]['run_id']
        if family=='catboost':
            complex_ranked=sorted([r for r in ranked if r['run_id']!='CB0'],key=ranking)
            checks['catboost_best_complex_recomputed']=selection['parameter_decision']['best_complex_candidate']==complex_ranked[0]['run_id']
        eligible_final=[ranked[0]]
        for branch in selection['branches']:
            base=runs[branch['base']];accepted=[];eligible=[base]
            groups=[g for g in selection['group_results'] if g['branch']==base['run_id']]
            for g in groups:
                if g['group']=='combined':continue
                desc=next(x for x in protocol['groups'] if x['id']==g['group']);trial=runs[g['run_id']]
                v,avg,wins,ok=change(trial,base)
                checks[g['run_id']+'_features_match_independent_addition']=trial['features']==protocol['base_features']+desc['add']
                checks[g['run_id']+'_same_parameters_as_parent']=trial['config']==base['config']
                checks[g['run_id']+'_acceptance_recomputed']=ok==g['accepted'] and wins==g['improved_windows'] and np.isclose(avg,g['mean_paired_change_per_10000'],atol=1e-8)
                if ok:accepted.append(desc);eligible.append(trial)
            combination=[g for g in groups if g['group']=='combined']
            checks[base['run_id']+'_combined_trigger']=bool(combination)==(len(accepted)>=2)
            if combination:
                g=combination[0];trial=runs[g['run_id']];v,avg,wins,ok=change(trial,base)
                checks[g['run_id']+'_combined_union']=trial['features']==protocol['base_features']+[c for desc in accepted for c in desc['add']]
                checks[g['run_id']+'_combined_acceptance']=ok==g['accepted'] and np.isclose(avg,g['mean_paired_change_per_10000'],atol=1e-8)
                if ok:eligible.append(trial)
            winner=sorted(eligible,key=ranking)[0]
            checks[base['run_id']+'_branch_winner']=winner['run_id']==branch['final_run']
            eligible_final.append(winner)
        winner=sorted(eligible_final,key=ranking)[0]
        checks[family+'_final_winner']=winner['run_id']==selection['final_run']==lock['final_runs'][family]
        checks[family+'_final_features']=winner['features']==selection['final_features']
        checks[family+'_decisions_before_global_lock']=(HERE/f'{family}_selection.json').stat().st_mtime_ns<=(HERE/'selection_lock.json').stat().st_mtime_ns
    for rid in lock['june_diagnostic_runs']:
        run=runs[rid];family=run['family'];folder=HERE/'runs'/rid;j=read(folder/'June_result.json')
        pred=pd.read_csv(folder/'June_predictions.csv.gz',float_precision='round_trip')
        schema=read(folder/'D3_schema.json');model=CatBoostClassifier() if family=='catboost' else XGBClassifier()
        model.load_model(str(folder/('D3.cbm' if family=='catboost' else 'D3.ubj')))
        verify_prediction(run,'D3','June',pred,j['selected'],run['folds'][2]['threshold_decision']['threshold'],model,schema,True)
        checks[rid+'_June_complete_ids']=np.array_equal(pred.ID,data.loc[data.timestamp.ge('2025-06-01')&data.timestamp.lt('2025-07-01'),'ID'])
        checks[rid+'_June_lock_binding']=j['selection_lock_sha256']==result['selection_lock_sha256']
        checks[rid+'_June_generated_after_global_lock']=(folder/'June_predictions.csv.gz').stat().st_mtime_ns>=(HERE/'selection_lock.json').stat().st_mtime_ns
    # Verify the separate report tables against the already independently checked records.
    metrics=pd.read_csv(HERE/'tables/metrics.csv')
    metric_match=[]
    for rid,run in runs.items():
        for fold in run['folds']:
            for role,key in [('threshold_selection','threshold_metrics'),('development_selection','development_metrics')]:
                rows=metrics.loc[metrics.run_id.eq(rid)&metrics.fold.eq(fold['fold'])&metrics.role.eq(role)]
                metric_match.append(len(rows)==1 and all(np.isclose(rows.iloc[0][k],fold[key][k]) for k in ['n','TP','FP','FN','TN','utility','delta_per_10000']))
    checks['report_metric_table_matches_verified_records']=all(metric_match) and len(metrics)==6*len(runs)
    jt=pd.read_csv(HERE/'tables/june_diagnostic.csv').set_index('run_id')
    checks['report_june_table_matches_verified_records']=len(jt)==len(lock['june_diagnostic_runs']) and all(
        all(np.isclose(jt.loc[rid,k],read(HERE/'runs'/rid/'June_result.json')['selected'][k]) for k in ['n','TP','FP','FN','TN','utility','delta_vs_always_normal'])
        for rid in lock['june_diagnostic_runs'])
    gt=pd.read_csv(HERE/'tables/feature_groups.csv').set_index('run_id')
    expected_groups=[g for s in result['selections'].values() for g in s['group_results']]
    checks['report_group_table_matches_verified_decisions']=len(gt)==len(expected_groups) and all(
        bool(gt.loc[g['run_id'],'accepted'])==g['accepted'] and np.isclose(gt.loc[g['run_id'],'mean_paired_change_per_10000'],g['mean_paired_change_per_10000'])
        and all(np.isclose(gt.loc[g['run_id'],f],v) for f,v in g['changes_per_10000'].items()) for g in expected_groups)
    ct=pd.read_csv(HERE/'tables/realized_capacity_D3.csv').set_index('run_id')
    expected_capacity_ids={c['id'] for family in result['selections'] for c in protocol[family+'_grid']} | set(lock['final_runs'].values())
    checks['capacity_diagnostic_candidate_ids']=set(ct.index)==expected_capacity_ids
    checks['capacity_diagnostic_counts_and_size']=all(
        ct.loc[rid,'tree_count']==runs[rid]['folds'][2]['tree_count']
        and ct.loc[rid,'feature_count']==len(runs[rid]['features'])
        and 0<ct.loc[rid,'nonempty_leaves']<=ct.loc[rid,'total_leaves']
        and np.isclose(ct.loc[rid,'mean_leaves_per_tree'],ct.loc[rid,'total_leaves']/ct.loc[rid,'tree_count'])
        and np.isclose(ct.loc[rid,'model_megabytes'],runs[rid]['folds'][2]['model_bytes']/1024**2)
        and ct.loc[rid,'maximum_realized_depth']<=runs[rid]['config'].get('depth',runs[rid]['config'].get('max_depth'))
        for rid in ct.index)
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.images=[];self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='img':self.images.append(a.get('src',''))
            if tag=='a':self.links.append(a.get('href',''))
    parser=Parser();parser.feed((HERE/'report_zh.html').read_text(encoding='utf-8'))
    checks['all_nine_report_images_embedded']=len(parser.images)==9 and all(i.startswith('data:image/png;base64,') for i in parser.images)
    # The validation record itself is written below after checking the artifact links.
    checks['all_local_report_links_resolve']=all((HERE/link).exists() or link=='qa/independent_validation.json' for link in parser.links if not link.startswith(('http://','https://','#')))
    for p in (HERE/'figures').glob('*.png'):
        with Image.open(p) as im:im.verify()
    checks['all_plot_pngs_valid']=True
    checks={k:bool(v) for k,v in checks.items()}
    out={'checks':checks,'count':len(checks),'all_passed':all(checks.values()),'saved_prediction_rows_checked':count_rows,
         'no_refits':True,'failures':[k for k,v in checks.items() if not v],
         'post_selection_report_source_hashes':{name:digest(HERE/name) for name in ['report_comparison.py','narrative_comparison.py','capacity_diagnostics.py','verify_comparison.py']}}
    (HERE/'qa/independent_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:out[k] for k in ['count','all_passed','saved_prediction_rows_checked','failures']},ensure_ascii=False),flush=True)
    assert out['all_passed']


if __name__=='__main__':main()
