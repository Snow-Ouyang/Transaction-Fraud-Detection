"""Independent saved-output audit. No fitting or model selection."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'capacity_comparison'
sys.path.insert(0,str(HERE.parent/'core_baseline/.deps'))
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    checks={};rows=0
    p=read(HERE/'protocol.json');manifest=read(HERE/'manifest.json');lock=read(HERE/'fit_lock.json')
    checks['all_frozen_sources_and_baseline_files_unchanged']=all(digest(ROOT/f)==h for f,h in manifest['hashes'].items())
    checks['all_fits_bound_before_june']=all(digest(HERE/'runs'/k/'result.json')==v for k,v in lock['runs'].items())
    checks['lock_manifest_binding']=lock['manifest_sha256']==digest(HERE/'manifest.json')
    data=pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl').set_index('ID',drop=False)
    raw=pd.read_csv(ROOT/'fraudTrain.csv',usecols=['ID','unix_time','is_fraud','category','amt']).set_index('ID')
    checks['cached_raw_fields_labels_align']=all(np.array_equal(data[c],raw.loc[data.ID,c]) for c in raw.columns)
    specs=[dict(run_id=rid,parent=None,family=family) for rid,family in [('CB5','catboost'),('X5__combined','xgboost')]]+p['branches']
    metrics=pd.read_csv(HERE/'tables/metrics.csv');pairs=pd.read_csv(HERE/'tables/paired_changes.csv')
    cats=pd.read_csv(HERE/'tables/category_support_errors.csv');distributions=pd.read_csv(HERE/'tables/score_distributions.csv')
    predictions={}
    for branch in specs:
        rid=branch['run_id'];family=branch['family'];folder=(HERE if branch['parent'] else PRIOR)/'runs'/rid
        run=read(folder/'result.json')
        if branch['parent']:
            parent=read(PRIOR/'runs'/branch['parent']/'result.json')
            checks[rid+'_fixed_parent_parameters']=run['config']==parent['config']
            checks[rid+'_exact_ablation_features']=run['features']==[c for c in parent['features'] if c not in branch['drop']]
        for fold,fspec in zip(run['folds'],p['folds']):
            name=fold['fold'];prefix=rid+'_'+name
            fit=data.loc[data.timestamp.lt(fspec['fit_end'])]
            schema=read(folder/f'{name}_schema.json');decision=read(folder/f'{name}_threshold.json');t=decision['threshold']
            checks[prefix+'_fit_rows']=len(fit)==fold['fit_n']
            checks[prefix+'_schema_hash']=digest(folder/f'{name}_schema.json')==fold['schema_sha256']
            checks[prefix+'_schema_features']=schema['features']==run['features']
            checks[prefix+'_fit_only_vocab']=schema['categorical']=={c:sorted(fit[c].astype(str).unique().tolist()) for c in run['features'] if c in p['categorical_features']}
            model=CatBoostClassifier() if family=='catboost' else XGBClassifier()
            mpath=folder/(name+('.cbm' if family=='catboost' else '.ubj'));model.load_model(str(mpath))
            checks[prefix+'_model_hash']=digest(mpath)==fold['model_sha256']
            actual=model.get_all_params() if family=='catboost' else None
            if family=='catboost':
                checks[prefix+'_actual_fixed_parameters']=all(np.isclose(actual[k],v) if isinstance(v,(int,float)) else actual[k]==v for k,v in {**p['catboost_fixed'],**{k:v for k,v in run['config'].items() if k!='id'},'random_seed':p['seed']}.items() if k not in ['thread_count','allow_writing_files'])
            else:
                # UBJSON reload restores trees but not all training configuration;
                # compare the frozen constructor path and saved fit-time parameter record.
                expected={**p['xgboost_fixed'],**{k:v for k,v in run['config'].items() if k!='id'},'random_state':p['seed']}
                checks[prefix+'_recorded_fixed_parameters']=read(folder/f'{name}_model.json')['parameters']==expected
                checks[prefix+'_actual_tree_budget']=model.get_booster().num_boosted_rounds()==expected['n_estimators']
            roles=[('threshold',name,f'{name}_threshold.csv.gz',fspec['fit_end'],fspec['threshold_end']),
                   ('development',name,f'{name}_development.csv.gz',fspec['dev_start'],fspec['dev_end'])]
            if name=='D3':roles.append(('diagnostic','June','June_predictions.csv.gz','2025-06-01','2025-07-01'))
            for role,window,filename,start,end in roles:
                pred=pd.read_csv(folder/filename,float_precision='round_trip');rows+=len(pred)
                key=prefix+'_'+role
                expected=data.loc[data.timestamp.ge(start)&data.timestamp.lt(end),'ID']
                checks[key+'_all_window_ids']=np.array_equal(pred.ID,expected)
                checks[key+'_raw_labels_times']=np.array_equal(pred.is_fraud,raw.loc[pred.ID,'is_fraud']) and np.array_equal(pred.unix_time,raw.loc[pred.ID,'unix_time'])
                checks[key+'_scores_valid']=np.isfinite(pred.score).all() and pred.score.between(0,1).all()
                checks[key+'_locked_threshold_decisions']=np.array_equal(pred.decision,pred.score.ge(t).astype(int))
                y=pred.is_fraud.to_numpy();d=pred.decision.to_numpy(bool)
                tp=int(((y==1)&d).sum());fp=int(((y==0)&d).sum());fn=int(((y==1)&~d).sum());tn=int(((y==0)&~d).sum())
                utility=50*tp-100*fn-5*fp
                row=metrics.loc[metrics.run_id.eq(rid)&metrics.window.eq(window)&metrics.role.eq(role)].iloc[0]
                checks[key+'_accounting']=tuple(row[c] for c in ['TP','FP','FN','TN','utility'])==(tp,fp,fn,tn,utility)
                checks[key+'_rates']=np.isclose(row.false_positive_rate,fp/(fp+tn)) and np.isclose(row.recall,tp/(tp+fn))
                checks[key+'_baseline']=row.delta_vs_always_normal==utility+100*int(y.sum())
                sample=np.arange(len(pred)) if role=='diagnostic' else np.linspace(0,len(pred)-1,min(300,len(pred)),dtype=int)
                inputs=data.loc[pred.ID.iloc[sample],schema['features']].copy()
                for c,levels in schema['categorical'].items():
                    inputs[c]=pd.Categorical(inputs[c].astype(str),categories=levels) if family=='xgboost' else inputs[c].astype(str)
                rescored=model.predict_proba(inputs)[:,1]
                checks[key+'_saved_model_rescore']=np.allclose(rescored,pred.score.iloc[sample],rtol=0,atol=1e-12)
                if role=='threshold':
                    groups=pred.assign(gain=np.where(y==1,150,-5)).groupby('score').agg(gain=('gain','sum'),n=('gain','size')).sort_index(ascending=False)
                    cumulative=groups.gain.cumsum();best=max(0,int(cumulative.max()))
                    flags=0 if best==0 else int(groups.n.cumsum().loc[cumulative.eq(best)].iloc[0])
                    checks[key+'_independent_optimal_threshold']=utility==-100*int(y.sum())+best and flags==int(d.sum())
                    checks[key+'_threshold_file_matches_record']=fold['threshold_decision']=={k:decision[k] for k in fold['threshold_decision']}
                if role=='diagnostic' and branch['parent']:
                    checks[key+'_june_bound_to_lock']=read(folder/'June_result.json')['fit_lock_sha256']==digest(HERE/'fit_lock.json')
                    checks[key+'_june_after_all_fits']=(folder/filename).stat().st_mtime_ns >= (HERE/'fit_lock.json').stat().st_mtime_ns
                fit_counts=fit.loc[fit.is_fraud.eq(0)].groupby('category').size()
                cc=cats.loc[cats.run_id.eq(rid)&cats.window.eq(window)&cats.role.eq(role)]
                checks[key+'_category_total_accounting']=tuple(cc[c].sum() for c in ['n','TP','FP','FN','TN','utility'])==(len(pred),tp,fp,fn,tn,utility)
                checks[key+'_category_fit_support']=all(int(r.fit_normal_n)==int(fit_counts.get(r.category,0)) for r in cc.itertuples())
                for label in [0,1]:
                    ss=distributions.loc[distributions.run_id.eq(rid)&distributions.window.eq(window)&distributions.role.eq(role)&distributions.is_fraud.eq(label)].iloc[0]
                    ps=pred.loc[pred.is_fraud.eq(label),'score']
                    checks[key+f'_score_quantiles_{label}']=all(np.isclose(ss[col],ps.quantile(q)) for col,q in [('score_p10',.1),('score_median',.5),('score_p90',.9)])
                predictions[(rid,window,role)]=pred
    for pair in pairs.itertuples():
        a=predictions[pair.run_id,pair.window,pair.role];b=predictions[pair.parent,pair.window,pair.role]
        y=a.is_fraud.to_numpy();ad=a.decision.to_numpy(bool);bd=b.decision.to_numpy(bool)
        checks[f'{pair.run_id}_{pair.window}_{pair.role}_paired_changes']=tuple(getattr(pair,c) for c in ['FP_fixed','FP_added','FN_fixed','FN_added'])==(
            int(((y==0)&bd&~ad).sum()),int(((y==0)&~bd&ad).sum()),int(((y==1)&~bd&ad).sum()),int(((y==1)&bd&~ad).sum()))
    ims=[]
    for f in sorted((HERE/'figures').glob('*.png')):
        im=Image.open(f).convert('RGB');checks[f.name+'_image_nonempty']=im.width>=800 and im.height>=400
        im.thumbnail((900,540));canvas=Image.new('RGB',(920,580),'white');canvas.paste(im,((920-im.width)//2,30));ImageDraw.Draw(canvas).text((10,5),f.name,fill='black');ims.append(canvas)
    sheet=Image.new('RGB',(1840,1160),'#eeeeee')
    for i,im in enumerate(ims):sheet.paste(im,((i%2)*920,(i//2)*580))
    sheet.save(HERE/'qa/figures_contact_sheet.png')
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(checks=checks,total_checks=len(checks),passed=sum(bool(x) for x in checks.values()),
                failed=[k for k,v in checks.items() if not v],prediction_rows_checked=rows,
                all_passed=all(checks.values()),no_fits=True)
    (HERE/'qa/independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False),flush=True)
    assert result['all_passed'],result['failed']


if __name__=='__main__':main()
