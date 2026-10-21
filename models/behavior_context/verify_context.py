"""Independent accounting, no-category allowlist, selection and saved-model audit."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
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
    p=read(HERE/'protocol.json');res=read(HERE/'results.json');lock=read(HERE/'selection_lock.json');m=read(HERE/'manifest.json')
    checks={};prediction_rows=0
    def put(k,v):checks[k]=bool(v)
    put('all_frozen_sources_baselines_and_cache_unchanged',all(digest(ROOT/k)==v for k,v in m['hashes'].items()))
    put('manifest_result_lock_binding',digest(HERE/'manifest.json')==res['manifest_sha256']==lock['manifest_sha256'])
    put('selection_lock_result_binding',digest(HERE/'selection_lock.json')==res['selection_lock_sha256'])
    put('group_lock_binding',digest(HERE/'group_lock.json')==lock['group_lock_sha256'])
    put('strict_past_feature_audit_all_passed',all(read(HERE/'qa/feature_validation.json')['checks'].values()))
    definition=read(HERE/'qa/score_definition_check.json')
    put('official_score_pdf_source_unchanged',digest(Path(definition['source']))==definition['source_sha256'])
    put('official_utility_weights',p['utility']=={'TP':50,'FN':-100,'FP':-5,'TN':0})
    data=pd.read_pickle(HERE/'feature_cache.pkl').set_index('ID',drop=False)
    put('one_hour_counts_subset_of_existing_day_counts',data.count_1h.le(data.count_24h).all())
    raw=pd.read_csv(ROOT/'fraudTrain.csv',usecols=['ID','unix_time','amt','is_fraud','category']).set_index('ID')
    put('all_cache_raw_values_and_labels_match',all(np.array_equal(data[c].astype(str) if c=='category' else data[c],raw.loc[data.ID,c].astype(str) if c=='category' else raw.loc[data.ID,c]) for c in raw.columns))
    old=pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl').set_index('ID',drop=False)
    put('existing_feature_values_preserved',all(np.array_equal(data[c].to_numpy(),old.loc[data.ID,c].to_numpy(),equal_nan=True) if pd.api.types.is_numeric_dtype(old[c]) else np.array_equal(data[c].astype(str),old.loc[data.ID,c].astype(str)) for c in old.columns))
    del old
    metrics=pd.read_csv(HERE/'tables/metrics.csv');group_table=pd.read_csv(HERE/'tables/feature_groups.csv')
    category=pd.read_csv(HERE/'tables/category_errors.csv');pairs=pd.read_csv(HERE/'tables/paired_changes.csv')
    scores=pd.read_csv(HERE/'tables/score_distributions.csv');summaries=pd.read_csv(HERE/'tables/run_summary.csv')
    entries={rid:(HERE/'runs'/rid) for rid in res['run_ids']}
    for b in p['baselines'].values():entries[b['source_run']]=HERE.parent/'category_ablation/runs'/b['source_run']
    runs={rid:read(folder/'result.json') for rid,folder in entries.items()}
    forbidden={'category','ID','unix_time','timestamp','customer_key','is_fraud','first','last','street','dob','cc_num'}
    for rid,run in runs.items():
        folder=entries[rid];family=run['family'];base=p['baselines'][family]
        put(rid+'_no_raw_category_or_leakage_inputs',not bool(set(run['features'])&forbidden))
        put(rid+'_fixed_hyperparameters',run['config']==base['config'])
        for fold,fp in zip(run['folds'],p['folds']):
            name=fold['fold'];prefix=rid+'_'+name
            fit=data.loc[data.timestamp.lt(fp['fit_end'])]
            schema=read(folder/f'{name}_schema.json');decision=read(folder/f'{name}_threshold.json');t=decision['threshold']
            put(prefix+'_fit_count',len(fit)==fold['fit_n'] and int(fit.is_fraud.sum())==fold['fit_fraud'])
            put(prefix+'_schema_hash',digest(folder/f'{name}_schema.json')==fold['schema_sha256'])
            put(prefix+'_schema_inputs',schema['features']==run['features'])
            put(prefix+'_fit_only_vocabulary',schema['categorical']=={c:sorted(fit[c].astype(str).unique().tolist()) for c in run['features'] if c in p['categorical_features']})
            mpath=folder/(name+('.cbm' if family=='catboost' else '.ubj'))
            put(prefix+'_model_hash',digest(mpath)==fold['model_sha256'])
            model=CatBoostClassifier() if family=='catboost' else XGBClassifier();model.load_model(str(mpath))
            put(prefix+'_actual_model_input_names',list(model.feature_names_ if family=='catboost' else model.get_booster().feature_names)==run['features'])
            expected={**p[family+'_fixed'],**{k:v for k,v in run['config'].items() if k!='id'},'random_seed' if family=='catboost' else 'random_state':p['seed']}
            put(prefix+'_fit_time_parameter_record',read(folder/f'{name}_model.json')['parameters']==expected)
            if family=='catboost':
                actual=model.get_all_params()
                put(prefix+'_actual_fixed_parameters',all(np.isclose(actual[k],v) if isinstance(v,(int,float)) else actual[k]==v for k,v in expected.items() if k not in ['thread_count','allow_writing_files']))
            else:put(prefix+'_actual_tree_budget',model.get_booster().num_boosted_rounds()==run['config']['n_estimators'])
            roles=[('threshold',name,f'{name}_threshold.csv.gz',fp['fit_end'],fp['threshold_end'],'threshold_metrics'),
                   ('development',name,f'{name}_development.csv.gz',fp['dev_start'],fp['dev_end'],'development_metrics')]
            if name=='D3':roles.append(('diagnostic','June','June_predictions.csv.gz','2025-06-01','2025-07-01',None))
            for role,window,filename,start,end,record_key in roles:
                pred=pd.read_csv(folder/filename,float_precision='round_trip');prediction_rows+=len(pred)
                key=prefix+'_'+role
                expected_ids=data.loc[data.timestamp.ge(start)&data.timestamp.lt(end),'ID']
                put(key+'_complete_window_ids',np.array_equal(pred.ID,expected_ids))
                put(key+'_raw_label_time_alignment',np.array_equal(pred.is_fraud,raw.loc[pred.ID,'is_fraud']) and np.array_equal(pred.unix_time,raw.loc[pred.ID,'unix_time']))
                put(key+'_scores_valid',np.isfinite(pred.score).all() and pred.score.between(0,1).all())
                put(key+'_threshold_decisions',np.array_equal(pred.score.ge(t),pred.decision.astype(bool)))
                y=pred.is_fraud.to_numpy();d=pred.decision.to_numpy(bool)
                tp=int(((y==1)&d).sum());fpn=int(((y==0)&d).sum());fn=int(((y==1)&~d).sum());tn=int(((y==0)&~d).sum())
                utility=50*tp-100*fn-5*fpn
                row=metrics.loc[metrics.run_id.eq(rid)&metrics.window.eq(window)&metrics.role.eq(role)].iloc[0]
                put(key+'_confusion_cost_accounting',tuple(row[c] for c in ['TP','FP','FN','TN','utility'])==(tp,fpn,fn,tn,utility))
                record=fold[record_key] if record_key else read(folder/'June_result.json')['selected']
                put(key+'_original_record_matches_saved_predictions',tuple(record[c] for c in ['TP','FP','FN','TN','utility'])==(tp,fpn,fn,tn,utility))
                put(key+'_baseline_normalization',row.always_normal_utility==-100*int(y.sum()) and row.delta_vs_always_normal==utility+100*int(y.sum()) and np.isclose(row.delta_per_10000,(utility+100*int(y.sum()))/len(y)*10000))
                put(key+'_rates',np.isclose(row.false_positive_rate,fpn/(fpn+tn)) and np.isclose(row.recall,tp/(tp+fn)))
                put(key+'_ranking_metrics',np.isclose(row.roc_auc,roc_auc_score(y,pred.score)) and np.isclose(row.average_precision,average_precision_score(y,pred.score)))
                ix=np.arange(len(pred)) if role=='diagnostic' else np.linspace(0,len(pred)-1,min(200,len(pred)),dtype=int)
                x=data.loc[pred.ID.iloc[ix],schema['features']].copy()
                for c,levels in schema['categorical'].items():x[c]=pd.Categorical(x[c].astype(str),categories=levels) if family=='xgboost' else x[c].astype(str)
                put(key+'_model_rescore',np.allclose(model.predict_proba(x)[:,1],pred.score.iloc[ix],rtol=0,atol=1e-12))
                if role=='threshold':
                    grouped=pred.assign(gain=np.where(y==1,150,-5)).groupby('score').agg(gain=('gain','sum'),n=('gain','size')).sort_index(ascending=False)
                    cum=grouped.gain.cumsum();best=max(0,int(cum.max()));flags=0 if best==0 else int(grouped.n.cumsum().loc[cum.eq(best)].iloc[0])
                    put(key+'_independent_exact_optimum',utility==-100*int(y.sum())+best and int(d.sum())==flags)
                    put(key+'_threshold_record',fold['threshold_decision']=={k:decision[k] for k in fold['threshold_decision']})
                if role=='diagnostic' and rid in res['run_ids']:
                    put(key+'_june_selection_lock',read(folder/'June_result.json')['selection_lock_sha256']==digest(HERE/'selection_lock.json'))
                    put(key+'_june_after_family_selection',(folder/filename).stat().st_mtime_ns>=(HERE/'selection_lock.json').stat().st_mtime_ns)
                cc=category.loc[category.run_id.eq(rid)&category.window.eq(window)&category.role.eq(role)]
                put(key+'_category_group_sums',tuple(cc[c].sum() for c in ['n','TP','FP','FN','utility'])==(len(pred),tp,fpn,fn,utility))
                normal_support=fit.loc[fit.is_fraud.eq(0)].groupby('category',observed=True).size()
                put(key+'_category_normal_fit_support',all(int(z.fit_normal_n)==int(normal_support.get(z.category,0)) for z in cc.itertuples()))
                for label in [0,1]:
                    z=scores.loc[scores.run_id.eq(rid)&scores.window.eq(window)&scores.role.eq(role)&scores.is_fraud.eq(label)].iloc[0]
                    values=pred.loc[pred.is_fraud.eq(label),'score']
                    put(key+f'_score_quantiles_{label}',all(np.isclose(z[k],values.quantile(q)) for k,q in [('p10',.1),('median',.5),('p90',.9)]))
    def changes(rid,baseline):
        a=metrics.loc[metrics.run_id.eq(rid)&metrics.role.eq('development')].set_index('window')
        b=metrics.loc[metrics.run_id.eq(baseline)&metrics.role.eq('development')].set_index('window')
        v=[(a.loc[w,'utility']-b.loc[w,'utility'])/a.loc[w,'n']*10000 for w in ['D1','D2','D3']]
        return float(np.mean(v)),sum(x>1e-9 for x in v)
    def rank(rid):
        run=runs[rid];cfg=run['config'];rows=metrics.loc[metrics.run_id.eq(rid)&metrics.role.eq('development')]
        return (-rows.delta_per_10000.mean(),-rows.delta_per_10000.min(),len(run['features']),cfg.get('depth',cfg.get('max_depth')),cfg.get('iterations',cfg.get('n_estimators')),-cfg.get('l2_leaf_reg',cfg.get('reg_lambda')),rid)
    for family,b in p['baselines'].items():
        selection=res['selections'][family];eligible=[b['source_run']];accepted=[]
        for g in b['groups']:
            rid=b['tag']+'__'+g['id'];avg,wins=changes(rid,b['source_run']);ok=avg>1e-9 and wins>=2
            recorded=next(z for z in selection['groups'] if z['run_id']==rid)
            row=group_table.loc[group_table.run_id.eq(rid)].iloc[0]
            put(rid+'_exact_independent_group_inputs',runs[rid]['features']==b['features']+g['add'])
            put(rid+'_acceptance_recomputed',ok==recorded['accepted']==row.accepted and wins==recorded['improved_windows'] and np.isclose(avg,recorded['mean_paired_change_per_10000']))
            if ok:eligible.append(rid);accepted.append(g)
        put(family+'_accepted_group_list',selection['accepted_groups']==[g['id'] for g in accepted])
        rid=b['tag']+'__combined'
        put(family+'_union_trigger',(rid in runs)==(len(accepted)>=2))
        if len(accepted)>=2:
            put(rid+'_exact_union_inputs',runs[rid]['features']==b['features']+[c for g in accepted for c in g['add']])
            avg,wins=changes(rid,b['source_run']);ok=avg>1e-9 and wins>=2
            put(rid+'_union_acceptance',ok==selection['combined']['accepted'] and wins==selection['combined']['improved_windows'] and np.isclose(avg,selection['combined']['mean_paired_change_per_10000']))
            if ok:eligible.append(rid)
        put(family+'_final_selection_recomputed',sorted(eligible,key=rank)[0]==selection['final_run'])
        put(family+'_final_features',selection['final_features']==runs[selection['final_run']]['features'])
    for row in pairs.itertuples():
        a=metrics.loc[metrics.run_id.eq(row.run_id)&metrics.window.eq(row.window)&metrics.role.ne('threshold')].iloc[0]
        b=metrics.loc[metrics.run_id.eq(row.baseline)&metrics.window.eq(row.window)&metrics.role.ne('threshold')].iloc[0]
        put(row.run_id+'_'+row.window+'_paired_cost_change',row.delta_utility==a.utility-b.utility and row.delta_FP==a.FP-b.FP and row.delta_FN==a.FN-b.FN and np.isclose(row.delta_per_10000,(a.utility-b.utility)/a.n*10000))
    rankings=pd.read_csv(HERE/'tables/score_rankings.csv')
    for row in rankings.itertuples():
        dev=metrics.loc[metrics.run_id.eq(row.run_id)&metrics.role.eq('development')]
        put(row.run_id+'_absolute_score_summary',row.pooled_score==int(dev.utility.sum()) and row.pooled_n==int(dev.n.sum()) and np.isclose(row.mean_score_per_10000,dev.utility_per_10000.mean()))
    for family in p['baselines']:
        rr=rankings.loc[rankings.family.eq(family)]
        put(family+'_equal_window_score_ranking',np.array_equal(rr.equal_window_score_rank,rr.mean_score_per_10000.rank(method='min',ascending=False).astype(int)))
        put(family+'_pooled_score_ranking',np.array_equal(rr.pooled_total_score_rank,rr.pooled_score.rank(method='min',ascending=False).astype(int)))
    ims=[]
    for path in sorted((HERE/'figures').glob('*.png')):
        im=Image.open(path).convert('RGB');put(path.name+'_valid_image',im.width>=800 and im.height>=400)
        im.thumbnail((900,600));canvas=Image.new('RGB',(920,640),'white');canvas.paste(im,((920-im.width)//2,30));ImageDraw.Draw(canvas).text((10,5),path.name,fill='black');ims.append(canvas)
    for start in [0,2]:
        batch=ims[start:(2 if start==0 else 5)];sheet=Image.new('RGB',(1840,640*((len(batch)+1)//2)),'#eeeeee')
        for i,im in enumerate(batch):sheet.paste(im,((i%2)*920,(i//2)*640))
        sheet.save(HERE/f'qa/contact_sheet_{start//2+1}.png')
    result=dict(checks=checks,total_checks=len(checks),passed=sum(checks.values()),failed=[k for k,v in checks.items() if not v],all_passed=all(checks.values()),prediction_rows_checked=prediction_rows,no_fits=True)
    (HERE/'qa/independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False),flush=True)
    assert result['all_passed'],result['failed']


if __name__=='__main__':main()
