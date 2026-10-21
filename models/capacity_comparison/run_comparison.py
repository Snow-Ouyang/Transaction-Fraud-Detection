"""Frozen fixed10 search, two model families and feature-capacity controls."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V2=HERE.parent/'catboost_progression'
CORE=HERE.parent/'core_baseline'
sys.path.insert(0,str(CORE))
sys.path.insert(0,str(CORE/'.deps'))
from run_core import evaluate,select_threshold,sha256,write_json,safe,check_cost_logic
import catboost,xgboost
from model_io import schema_for,transform,parameters,make_model,suffix,load_model,predict,unknown_counts,importance

P=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def stamp():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def fingerprint(obj):return hashlib.sha256(json.dumps(safe(obj),sort_keys=True).encode()).hexdigest()


def log(family,message):
    line=stamp()+' '+message
    print(line,flush=True)
    with open(HERE/f'progress_{family}.log','a',encoding='utf-8') as f:f.write(line+'\n')


def prepare():
    for p in ['runs','tables','figures','qa']: (HERE/p).mkdir(exist_ok=True)
    assert P['base_features']==read(V2/'results.json')['final_features']
    assert all(read(V2/'qa/feature_validation.json')['checks'].values())
    sources=['fraudTrain.csv','fraudTest.csv','ISYE 7406 Project Description.pdf',
             'models/catboost_progression/results.json','models/catboost_progression/selection_lock.json',
             'models/catboost_progression/feature_cache.pkl','models/catboost_progression/features.py',
             'models/catboost_progression/qa/feature_validation.json','models/core_baseline/run_core.py']
    manifest={'source_hashes':{name:sha256(ROOT/name) for name in sources},
              'code_hashes':{name:sha256(HERE/name) for name in ['protocol.json','run_comparison.py','model_io.py']},
              'runtime':{'python':platform.python_version(),'catboost':catboost.__version__,'xgboost':xgboost.__version__,
                         'pandas':pd.__version__,'numpy':np.__version__,'executable':sys.executable},'started_utc':stamp()}
    assert manifest['source_hashes']['fraudTrain.csv']==read(V2/'results.json')['signature']['source_sha256']
    path=HERE/'manifest.json'
    if path.exists():
        old=read(path)
        assert all(old[k]==manifest[k] for k in ['source_hashes','code_hashes','runtime'])
    else:write_json(path,manifest)
    check_cost_logic()
    # Tests exercise only the encoding adapter, not a trial model or validation labels.
    sample=pd.DataFrame({'c':['a','b'],'x':[1.,np.nan]})
    s=schema_for(sample,['c','x'],['c'])
    future=pd.DataFrame({'c':['b','a','unknown'],'x':[2.,3.,np.nan]})
    z=transform(future,s,'xgboost')
    checks={'v2_history_audit_passed':True,'known_levels_consistent':z.c.cat.codes.tolist()==[1,0,-1],
            'future_unknown_mapped_missing':bool(z.c.isna().iloc[-1]),'numeric_missing_preserved':bool(np.isnan(z.x.iloc[-1])),
            'future_vocab_not_learned':'unknown' not in s['categorical']['c']}
    assert all(checks.values())
    write_json(HERE/'qa/encoding_checks.json',checks)
    log('prepare','Frozen protocol/code/source manifest; native-category encoding checks passed')


def check_manifest():
    m=read(HERE/'manifest.json')
    assert all(sha256(HERE/k)==v for k,v in m['code_hashes'].items()),'Frozen code changed'
    assert sha256(V2/'feature_cache.pkl')==m['source_hashes']['models/catboost_progression/feature_cache.pkl']
    return m


def rank(run):
    c=run['config']
    return (-run['mean_delta_per_10000'],-run['worst_delta_per_10000'],len(run['features']),
            c.get('depth',c.get('max_depth')),c.get('iterations',c.get('n_estimators')),
            -c.get('l2_leaf_reg',c.get('reg_lambda')),run['run_id'])


def paired(candidate,parent):
    p={f['fold']:f['development_metrics']['utility_per_10000'] for f in parent['folds']}
    values={f['fold']:f['development_metrics']['utility_per_10000']-p[f['fold']] for f in candidate['folds']}
    mean=float(np.mean(list(values.values())))
    wins=sum(v>1e-9 for v in values.values())
    return {'changes_per_10000':values,'mean_paired_change_per_10000':mean,'improved_windows':wins,
            'accepted':bool(mean>1e-9 and wins>=2)}


def fit_run(family,run_id,config,features,data,reuse=None):
    path=HERE/'runs'/run_id
    path.mkdir(exist_ok=True)
    run_signature=fingerprint({'manifest':sha256(HERE/'manifest.json'),'family':family,'config':config,'features':features})
    if (path/'result.json').exists():
        r=read(path/'result.json');assert r['run_signature']==run_signature
        assert all(sha256(path/(f['fold']+suffix(family)))==f['model_sha256'] for f in r['folds'])
        log(family,f'[{run_id}] resume completed run');return r
    folds=[]
    for f in P['folds']:
        name=f['name'];record_path=path/f'{name}_result.json';model_path=path/(name+suffix(family))
        if record_path.exists():
            record=read(record_path);assert record['run_signature']==run_signature
            assert sha256(model_path)==record['model_sha256'];folds.append(record);continue
        fit=data.loc[data.timestamp.lt(f['fit_end'])]
        cal=data.loc[data.timestamp.ge(f['fit_end']) & data.timestamp.lt(f['threshold_end'])]
        dev=data.loc[data.timestamp.ge(f['dev_start']) & data.timestamp.lt(f['dev_end'])]
        assert fit.timestamp.max()<cal.timestamp.min()<dev.timestamp.min()
        schema=schema_for(fit,features,P['categorical_features'])
        schema_path=path/f'{name}_schema.json';write_json(schema_path,schema)
        meta_path=path/f'{name}_model.json';tick=time.time()
        if meta_path.exists():
            meta=read(meta_path);assert meta['run_signature']==run_signature
            assert sha256(model_path)==meta['model_sha256']
            model=load_model(model_path,family);fit_seconds=meta['fit_seconds']
        else:
            if reuse:
                assert family=='catboost'
                old=read(V2/'runs'/reuse/'result.json');assert old['features']==features
                assert old['parameters']==parameters(P,family,config)
                prior=next(v for v in old['folds'] if v['fold']==name)
                src=V2/'runs'/reuse/(name+'.cbm');assert sha256(src)==prior['model_sha256']
                shutil.copy2(src,model_path);model=load_model(model_path,family);fit_seconds=0.
                log(family,f'[{run_id}/{name}] reuse v2/{reuse}')
            else:
                log(family,f'[{run_id}/{name}] FIT config={config["id"]}; inputs={len(features)}; rows={len(fit):,}')
                model=make_model(P,family,config,features)
                x=transform(fit,schema,family)
                if family=='xgboost': model.fit(x,fit.is_fraud,verbose=False)
                else:model.fit(x,fit.is_fraud)
                fit_seconds=time.time()-tick;model.save_model(str(model_path));del x
            meta={'run_signature':run_signature,'model_sha256':sha256(model_path),'fit_seconds':fit_seconds,
                  'reused_v2':reuse,'parameters':parameters(P,family,config),'schema_sha256':sha256(schema_path)}
            write_json(meta_path,meta)
        p_cal=predict(model,cal,schema,family);assert np.isfinite(p_cal).all()
        decision=select_threshold(cal.is_fraud,p_cal)
        write_json(path/f'{name}_threshold.json',{'run_signature':run_signature,'schema_sha256':sha256(schema_path),**decision})
        for role,frame,scores in [('threshold',cal,p_cal),('development',dev,None)]:
            if scores is None:scores=predict(model,frame,schema,family);p_dev=scores
            assert np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
            out=frame[['ID','unix_time','is_fraud']].copy()
            out['score']=scores;out['decision']=(scores>=decision['threshold']).astype('int8')
            out.to_csv(path/f'{name}_{role}.csv.gz',index=False,compression='gzip')
        loaded=load_model(model_path,family)
        subset=dev.iloc[::max(1,len(dev)//500)]
        difference=float(np.max(np.abs(predict(model,subset,schema,family)-predict(loaded,subset,schema,family))))
        assert difference<=1e-12
        importance(model,family,features).to_csv(path/f'{name}_importance.csv',index=False,encoding='utf-8-sig')
        if family=='catboost': tree_count=int(model.tree_count_)
        else:tree_count=int(model.get_booster().num_boosted_rounds())
        record={'fold':name,'run_signature':run_signature,'model_sha256':sha256(model_path),'schema_sha256':sha256(schema_path),
                'fit_n':len(fit),'fit_fraud':int(fit.is_fraud.sum()),'fit_seconds':fit_seconds,'reused_v2':reuse,
                'tree_count':tree_count,'model_bytes':model_path.stat().st_size,'reload_max_difference':difference,
                'threshold_decision':decision,'threshold_metrics':evaluate(cal.is_fraud,p_cal,decision['threshold']),
                'development_metrics':evaluate(dev.is_fraud,p_dev,decision['threshold']),
                'unknown_levels_threshold':unknown_counts(cal,schema),'unknown_levels_development':unknown_counts(dev,schema)}
        if reuse:
            oldfold=next(v for v in read(V2/'runs'/reuse/'result.json')['folds'] if v['fold']==name)
            assert record['development_metrics']['utility']==oldfold['development_metrics']['utility']
            assert record['threshold_decision']==oldfold['decision']
        write_json(record_path,record);folds.append(record)
        log(family,f'[{run_id}/{name}] DONE U={record["development_metrics"]["utility"]:,}; FP={record["development_metrics"]["FP"]:,}; fit={fit_seconds:.1f}s')
        del model,loaded,fit,cal,dev
    delta=[f['development_metrics']['delta_per_10000'] for f in folds]
    r={'run_id':run_id,'family':family,'config':config,'features':features,'run_signature':run_signature,'folds':folds,
       'mean_delta_per_10000':float(np.mean(delta)),'worst_delta_per_10000':float(np.min(delta)),
       'windows_above_always_normal':int(sum(v>0 for v in delta))}
    write_json(path/'result.json',r);return r


def run_family(family):
    check_manifest()
    result_path=HERE/f'{family}_selection.json'
    if result_path.exists():log(family,'Completed family exists; preserve selection');return
    data=pd.read_pickle(V2/'feature_cache.pkl')
    assert data.unix_time.is_monotonic_increasing
    grid=P[family+'_grid'];runs={};search=[]
    for c in grid:
        r=fit_run(family,c['id'],c,P['base_features'],data,reuse='E3_behavior_history' if c['id']=='CB0' else None)
        runs[r['run_id']]=r;search.append(r)
    ordered=sorted(search,key=rank);best=ordered[0]
    complex_best=sorted([r for r in search if r['config']['id']!='CB0'],key=rank)[0] if family=='catboost' else None
    decision={'family':family,'search_winner':best['run_id'],'best_complex_candidate':complex_best['run_id'] if complex_best else None,
              'ranked':[{k:r[k] for k in ['run_id','config','mean_delta_per_10000','worst_delta_per_10000','windows_above_always_normal']} for r in ordered],
              'locked_utc':stamp()}
    write_json(HERE/f'{family}_parameter_decision.json',decision)
    log(family,f'PARAMETERS LOCKED: overall={best["run_id"]}; complex={decision["best_complex_candidate"]}')
    bases=[runs['CB0'],complex_best] if family=='catboost' else [best]
    group_rows=[];branch_rows=[]
    for base in bases:
        accepted=[];eligible=[base];base_id=base['run_id']
        for g in P['groups']:
            run_id=base_id+'__'+g['id']
            reuse={'merchant':'E4_merchant','identity':'E5_identity_proxies'}.get(g['id']) if base_id=='CB0' else None
            trial=fit_run(family,run_id,base['config'],P['base_features']+g['add'],data,reuse=reuse)
            runs[run_id]=trial;change=paired(trial,base)
            row={'family':family,'branch':base_id,'group':g['id'],'group_name':g['name'],'run_id':run_id,
                 'feature_count':len(trial['features']),**change}
            group_rows.append(row)
            if change['accepted']:accepted.append(g);eligible.append(trial)
            log(family,f'GROUP {run_id}: {"ACCEPT" if change["accepted"] else "REJECT"}; mean/10k={change["mean_paired_change_per_10000"]:+.2f}; wins={change["improved_windows"]}/3')
        if len(accepted)>=2:
            names=P['base_features']+[n for g in accepted for n in g['add']]
            run_id=base_id+'__combined'
            trial=fit_run(family,run_id,base['config'],names,data);runs[run_id]=trial;change=paired(trial,base)
            group_rows.append({'family':family,'branch':base_id,'group':'combined','group_name':'通过组的合并检验','run_id':run_id,
                               'feature_count':len(names),**change})
            if change['accepted']:eligible.append(trial)
        winner=sorted(eligible,key=rank)[0]
        branch_rows.append({'base':base_id,'accepted_individual_groups':[g['id'] for g in accepted],
                            'eligible_runs':[r['run_id'] for r in eligible],'final_run':winner['run_id']})
    final=sorted([runs[b['final_run']] for b in branch_rows]+[best],key=rank)[0]
    selection={'family':family,'parameter_decision':decision,'branches':branch_rows,'group_results':group_rows,
               'final_run':final['run_id'],'final_features':final['features'],'all_run_ids':list(runs),
               'locked_utc':stamp(),'no_june_scores_generated':True,'manifest_sha256':sha256(HERE/'manifest.json')}
    write_json(result_path,selection)
    log(family,f'FAMILY SELECTION LOCKED final={final["run_id"]}; mean delta/10k={final["mean_delta_per_10000"]:.2f}')


def finalize():
    if (HERE/'results.json').exists():raise RuntimeError('Completed v3 preserved; report may be rebuilt, fits not overwritten')
    manifest=check_manifest();selections={f:read(HERE/f'{f}_selection.json') for f in ['catboost','xgboost']}
    locks={f:sha256(HERE/f'{f}_selection.json') for f in selections}
    runs={rid:read(HERE/'runs'/rid/'result.json') for s in selections.values() for rid in s['all_run_ids']}
    # Only relevant reference/selected base and extension candidates enter reused June diagnostics.
    june_ids=[]
    for s in selections.values():
        june_ids += [s['parameter_decision']['search_winner'],s['final_run']]
        for b in s['branches']:june_ids.append(b['base'])
        june_ids += [g['run_id'] for g in s['group_results']]
    june_ids=list(dict.fromkeys(june_ids))
    lock={'manifest_sha256':sha256(HERE/'manifest.json'),'family_selection_hashes':locks,
          'final_runs':{f:s['final_run'] for f,s in selections.items()},'june_diagnostic_runs':june_ids,
          'locked_utc':stamp(),'june_previously_observed':True,'no_new_independent_test':True}
    lock_path=HERE/'selection_lock.json'
    if lock_path.exists():
        existing=read(lock_path);assert existing['family_selection_hashes']==locks
    else:write_json(lock_path,lock)
    lock_hash=sha256(lock_path)
    log('final','BOTH FAMILIES LOCKED; beginning reused June diagnostics only now')
    data=pd.read_pickle(V2/'feature_cache.pkl');june=data.loc[data.timestamp.ge('2025-06-01') & data.timestamp.lt('2025-07-01')]
    diagnostics=[]
    for run_id in june_ids:
        run=runs[run_id];folder=HERE/'runs'/run_id;path=folder/'June_result.json';family=run['family']
        if path.exists():
            d=read(path);assert d['selection_lock_sha256']==lock_hash
        else:
            model=load_model(folder/('D3'+suffix(family)),family);schema=read(folder/'D3_schema.json')
            scores=predict(model,june,schema,family);t=run['folds'][2]['threshold_decision']['threshold']
            out=june[['ID','unix_time','is_fraud']].copy();out['score']=scores;out['decision']=(scores>=t).astype('int8')
            out.to_csv(folder/'June_predictions.csv.gz',index=False,compression='gzip')
            d={'run_id':run_id,'family':family,'selection_lock_sha256':lock_hash,'generated_utc':stamp(),
               'role':'reused_diagnostic_not_blind_test','selected':evaluate(june.is_fraud,scores,t),
               'unknown_categories':unknown_counts(june,schema)}
            write_json(path,d)
        diagnostics.append(d);log('final',f'JUNE {run_id}: U={d["selected"]["utility"]:,}; FP={d["selected"]["FP"]:,}')
    metrics=[];fits=[];grid_rows=[]
    for r in runs.values():
        for f in r['folds']:
            for role,key in [('threshold_selection','threshold_metrics'),('development_selection','development_metrics')]:
                metrics.append({'family':r['family'],'run_id':r['run_id'],'fold':f['fold'],'role':role,'feature_count':len(r['features']),**f[key]})
            fits.append({'family':r['family'],'run_id':r['run_id'],**{k:f[k] for k in ['fold','fit_n','fit_seconds','tree_count','model_bytes','reused_v2','model_sha256']}})
    for family,s in selections.items():
        for i,r in enumerate(s['parameter_decision']['ranked']):grid_rows.append({'family':family,'rank':i+1,**r})
    group_rows=[{**g,**g['changes_per_10000']} for s in selections.values() for g in s['group_results']]
    for name,rows in [('metrics',metrics),('parameter_grid',grid_rows),('feature_groups',group_rows),('fits',fits),
                       ('june_diagnostic',[{'run_id':d['run_id'],'family':d['family'],**d['selected']} for d in diagnostics])]:
        pd.DataFrame(rows).to_csv(HERE/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    checks={'sources_unchanged':all(sha256(ROOT/k)==v for k,v in manifest['source_hashes'].items()),
            'code_unchanged':all(sha256(HERE/k)==v for k,v in manifest['code_hashes'].items()),
            'selection_locks_unchanged':all(sha256(HERE/f'{f}_selection.json')==v for f,v in locks.items()),
            'global_selection_lock_unchanged':sha256(lock_path)==lock_hash,
            'all_model_reload_checks_passed':all(f['reload_max_difference']<=1e-12 for r in runs.values() for f in r['folds']),
            'both_six_candidate_searches_complete':all(len(s['parameter_decision']['ranked'])==6 for s in selections.values()),
            'no_official_test_features_loaded':True,'all_input_feature_sets_exclude_labels_dates_and_raw_identity':
                not any(set(r['features']) & {'is_fraud','ID','unix_time','timestamp','customer_key','first','last','street','dob','cc_num'} for r in runs.values())}
    assert all(checks.values())
    result={'experiment':P['experiment'],'selections':selections,'run_ids':list(runs),'selection_lock_sha256':lock_hash,
            'june':diagnostics,'runtime':manifest['runtime'],'checks':checks,'completed_utc':stamp(),
            'new_model_fits':sum(not bool(f['reused_v2']) for f in fits),'reused_model_fits':sum(bool(f['reused_v2']) for f in fits),
            'total_checkpoints':len(fits),'manifest_sha256':sha256(HERE/'manifest.json')}
    write_json(HERE/'results.json',result);log('final','V3 COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','catboost','xgboost','finalize','report'],required=True)
    stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage in ['catboost','xgboost']:run_family(stage)
    elif stage=='finalize':finalize()
    else:
        from report_comparison import render
        render(HERE)
