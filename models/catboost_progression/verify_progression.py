"""Independent source, threshold, selection-path and serialized-score verification.

This program never fits a model and never changes a parameter/feature decision.
"""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = HERE.parent/'core_baseline'
sys.path.insert(0, str(CORE/'.deps'))
from catboost import CatBoostClassifier
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''): h.update(part)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def confusion(y, decision):
    y, decision = np.asarray(y), np.asarray(decision, bool)
    tp, fp = int(((y == 1) & decision).sum()), int(((y == 0) & decision).sum())
    fn, tn = int(((y == 1) & ~decision).sum()), int(((y == 0) & ~decision).sum())
    return tp, fp, fn, tn, 50*tp-100*fn-5*fp


def main():
    result = load_json(HERE/'results.json')
    manifest = load_json(HERE/'experiment_manifest.json')
    lock = load_json(HERE/'selection_lock.json')
    protocol = load_json(HERE/'protocol.json')
    checks = {}
    checks['all_raw_and_previous_artifact_hashes_unchanged'] = all(digest(ROOT/name) == value for name,value in result['source_hashes'].items())
    checks['protocol_hash_frozen'] = digest(HERE/'protocol.json') == result['signature']['protocol_sha256']
    checks['runner_hash_frozen'] = digest(HERE/'run_progression.py') == result['signature']['runner_sha256']
    checks['feature_code_hash_frozen'] = digest(HERE/'features.py') == result['signature']['feature_code_sha256']
    checks['selection_lock_hash_unchanged'] = digest(HERE/'selection_lock.json') == result['selection_lock_sha256']
    feature_audit = load_json(HERE/'qa/feature_validation.json')
    checks['history_direct_checks_and_prefix_invariance'] = all(feature_audit['checks'].values())
    source = pd.read_csv(ROOT/'fraudTrain.csv', usecols=['ID','unix_time','amt','category','is_fraud']).set_index('ID')
    source['hour'] = pd.to_datetime(source.unix_time, unit='s').dt.hour.astype(str)
    data = pd.read_pickle(HERE/'feature_cache.pkl').set_index('ID',drop=False)
    aligned = source.loc[data.ID]
    checks['feature_cache_all_core_fields_and_labels_align_raw'] = all(np.array_equal(data[c],aligned[c]) for c in ['unix_time','amt','category','hour','is_fraud'])
    checks['numeric_inputs_have_no_infinite_values'] = not np.isinf(data.select_dtypes('number').drop(columns=['customer_key']).to_numpy(dtype=float)).any()
    forbidden = {'ID','unix_time','timestamp','is_fraud','customer_key','first','last','street','dob','cc_num'}
    summaries = {}
    fold_metadata = {f['name']:f for f in protocol['folds']}
    prediction_rows = 0
    for run_id in result['run_ids']:
        folder = HERE/'runs'/run_id
        run = load_json(folder/'result.json')
        summaries[run_id] = run
        checks[run_id+'_feature_allowlist'] = not (set(run['features']) & forbidden)
        for fold in run['folds']:
            name = fold['fold']
            model_path = folder/f'{name}.cbm'
            checks[run_id+'_'+name+'_model_hash'] = digest(model_path) == fold['model_sha256']
            model = CatBoostClassifier(); model.load_model(str(model_path))
            checks[run_id+'_'+name+'_input_names'] = model.feature_names_ == run['features']
            decision = load_json(folder/f'{name}_decision.json')
            t = decision['threshold']
            for role, key in [('threshold','threshold_metrics'),('development','development_metrics')]:
                pred = pd.read_csv(folder/f'{name}_{role}.csv.gz',float_precision='round_trip')
                prediction_rows += len(pred)
                raw = source.loc[pred.ID]
                core = data.loc[pred.ID]
                checks[run_id+'_'+name+'_'+role+'_source_alignment'] = (
                    np.array_equal(pred.unix_time,raw.unix_time) and np.array_equal(pred.is_fraud,raw.is_fraud))
                recorded = fold[key]
                tp,fp,fn,tn,u = confusion(pred.is_fraud,pred.decision)
                checks[run_id+'_'+name+'_'+role+'_confusion_and_utility'] = (
                    (tp,fp,fn,tn,u) == tuple(int(recorded[k]) for k in ['TP','FP','FN','TN','utility']))
                checks[run_id+'_'+name+'_'+role+'_score_threshold_decision'] = np.array_equal(pred.score >= t,pred.decision.astype(bool))
                sample = np.linspace(0,len(pred)-1,min(100,len(pred)),dtype=int)
                p = model.predict_proba(core.iloc[sample][run['features']])[:,1]
                checks[run_id+'_'+name+'_'+role+'_saved_model_scores'] = np.allclose(p,pred.score.to_numpy()[sample],rtol=0,atol=1e-12)
                times = pd.to_datetime(pred.unix_time,unit='s')
                f = fold_metadata[name]
                lower = f['fit_end_exclusive'] if role == 'threshold' else f['development_start']
                upper = f['threshold_end_exclusive'] if role == 'threshold' else f['development_end_exclusive']
                checks[run_id+'_'+name+'_'+role+'_window_bounds'] = times.ge(lower).all() and times.lt(upper).all()
                if role == 'threshold':
                    # Independently aggregate cost increments by tied score, rather
                    # than reusing the training runner's threshold implementation.
                    grouped = pred.assign(gain=np.where(pred.is_fraud.eq(1),150,-5)).groupby('score').agg(
                        gain=('gain','sum'),n=('gain','size')).sort_index(ascending=False)
                    cumulative = grouped.gain.cumsum()
                    best_gain = max(0,int(cumulative.max()))
                    best_u = -100*int(pred.is_fraud.sum())+best_gain
                    expected_flags = 0 if best_gain == 0 else int(grouped.n.cumsum().loc[cumulative.eq(best_gain)].iloc[0])
                    checks[run_id+'_'+name+'_independent_optimal_threshold_utility'] = best_u == decision['threshold_window_utility'] and expected_flags == decision['flags']

    ranked = sorted(result['parameter_grid'],key=lambda r:(-r['mean_delta_per_10000'],-r['worst_delta_per_10000'],r['depth'],-r['l2_leaf_reg']))
    checks['parameter_choice_recomputed'] = ranked[0]['run_id'] == result['selected_parameter_run']
    current = result['selected_parameter_run']
    telescoping = summaries[current]['mean_development_delta_per_10000']
    for group,step in zip(protocol['feature_groups'],result['feature_decisions']):
        candidate = summaries[group['id']]
        parent = summaries[current]
        checks[group['id']+'_parent_and_group_features'] = (
            step['parent_run'] == current and candidate['features'] == parent['features']+group['add'])
        old = {f['fold']:f['development_metrics'] for f in parent['folds']}
        changes = [f['development_metrics']['utility_per_10000']-old[f['fold']]['utility_per_10000'] for f in candidate['folds']]
        mean,wins = float(np.mean(changes)),sum(v > 1e-9 for v in changes)
        retained = mean > 1e-9 and wins >= 2
        checks[group['id']+'_retention_recomputed'] = retained == step['retained'] and np.isclose(mean,step['mean_paired_delta_per_10000'],atol=1e-9)
        if retained:
            current = group['id']; telescoping += mean
    checks['final_retained_path_matches_lock'] = current == result['final_retained_run'] == lock['final_retained_run']
    checks['final_features_match_retained_path'] = summaries[current]['features'] == result['final_features'] == lock['final_features']
    checks['retained_increments_telescope'] = np.isclose(telescoping,result['final_development_mean_delta_per_10000'],atol=1e-6)

    for run_id in lock['june_diagnostic_runs']:
        folder = HERE/'runs'/run_id
        summary = load_json(folder/'June_diagnostic.json')
        pred = pd.read_csv(folder/'June_diagnostic.csv.gz',float_precision='round_trip')
        prediction_rows += len(pred)
        raw = source.loc[pred.ID]
        checks[run_id+'_June_source_alignment'] = np.array_equal(pred.is_fraud,raw.is_fraud) and np.array_equal(pred.unix_time,raw.unix_time)
        checks[run_id+'_June_lock_binding'] = summary['selection_lock_sha256'] == result['selection_lock_sha256']
        tp,fp,fn,tn,u = confusion(pred.is_fraud,pred.decision)
        checks[run_id+'_June_cost_recomputed'] = (tp,fp,fn,tn,u) == tuple(int(summary['selected'][k]) for k in ['TP','FP','FN','TN','utility'])
        model = CatBoostClassifier();model.load_model(str(folder/'D3.cbm'))
        p = model.predict_proba(data.loc[pred.ID,summaries[run_id]['features']])[:,1]
        checks[run_id+'_June_full_model_rescore'] = np.allclose(p,pred.score,rtol=0,atol=1e-12)
        checks[run_id+'_June_predictions_after_selection_lock'] = (folder/'June_diagnostic.csv.gz').stat().st_mtime_ns >= (HERE/'selection_lock.json').stat().st_mtime_ns
    checks['all_feature_decisions_before_selection_lock'] = (HERE/'feature_decisions.json').stat().st_mtime_ns <= (HERE/'selection_lock.json').stat().st_mtime_ns
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.images=[];self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='img':self.images.append(a.get('src',''))
            if tag=='a':self.links.append(a.get('href',''))
    parser=Parser();parser.feed((HERE/'report_zh.html').read_text(encoding='utf-8'))
    checks['all_nine_figures_embedded'] = len(parser.images)==9 and all(x.startswith('data:image/png;base64,') for x in parser.images)
    checks['local_report_links_resolve'] = all((HERE/p).exists() for p in parser.links if not p.startswith(('http://','https://','#')))
    for path in (HERE/'figures').glob('*.png'):
        with Image.open(path) as im:im.verify()
    checks['all_png_figures_valid'] = True
    checks={k:bool(v) for k,v in checks.items()}
    out={'checks':checks,'count':len(checks),'all_passed':all(checks.values()),
         'saved_prediction_rows_checked':prediction_rows,'feature_cache_sha256':digest(HERE/'feature_cache.pkl'),
         'no_refits':True}
    (HERE/'qa/independent_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'checks':len(checks),'all_passed':out['all_passed'],'saved_prediction_rows_checked':prediction_rows,
                      'failures':[k for k,v in checks.items() if not v]},ensure_ascii=False),flush=True)
    assert out['all_passed']


if __name__ == '__main__':main()
