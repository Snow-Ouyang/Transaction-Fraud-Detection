"""Fixed-parameter category/proxy ablation. Original experiments stay immutable."""
from pathlib import Path
import argparse
import json
import sys
import platform
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'capacity_comparison'
ROOT = HERE.parents[1]
sys.path.insert(0, str(PRIOR))
import run_comparison as engine
from model_io import load_model, predict, suffix
from run_core import sha256, write_json, evaluate


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    if (HERE/'manifest.json').exists():
        check()
        return
    old = read(PRIOR/'protocol.json')
    p = {k: old[k] for k in ['seed', 'categorical_features', 'catboost_fixed', 'xgboost_fixed', 'folds', 'utility']}
    p.update(experiment='v4_category_proxy_ablation',
             threshold='Exact utility maximum on preceding threshold window only; reselected per ablation, never on development or June.',
             selection='Diagnostic experiment; no hyperparameter search or automatic promotion. Report all three branches.',
             june='Previously seen diagnostic; no independent test, no threshold or parameter tuning.',
             history='Immutable v2 strictly-past label-free feature cache; no new features.',
             official_test='Do not load, predict, or submit.',
             interpretation='Feature removal can change interactions and model fit; it does not establish a unique causal mechanism.')
    p['branches'] = []
    for run_id, parent, drop in [
        ('CB5_no_category', 'CB5', ['category']),
        ('X5_no_category', 'X5__combined', ['category']),
        ('X5_no_category_merchant', 'X5__combined', ['category', 'merchant'])]:
        r = read(PRIOR/'runs'/parent/'result.json')
        p['branches'].append(dict(run_id=run_id, parent=parent, family=r['family'], config=r['config'],
                                  drop=drop, features=[c for c in r['features'] if c not in drop]))
    write_json(HERE/'protocol.json', p)
    paths = [HERE/'protocol.json', HERE/'run_ablation.py',
             PRIOR/'protocol.json', PRIOR/'run_comparison.py', PRIOR/'model_io.py',
             HERE.parent/'core_baseline/run_core.py', HERE.parent/'catboost_progression/feature_cache.pkl',
             HERE.parent/'catboost_progression/features.py', HERE.parent/'catboost_progression/qa/feature_validation.json',
             ROOT/'fraudTrain.csv']
    for parent in ['CB5', 'X5__combined']:
        folder = PRIOR/'runs'/parent
        paths += [x for x in folder.iterdir() if x.is_file()]
    write_json(HERE/'manifest.json', dict(started_utc=engine.stamp(),
        hashes={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        runtime=dict(python=platform.python_version(), executable=sys.executable,
                     catboost=engine.catboost.__version__, xgboost=engine.xgboost.__version__)))
    for name in ['runs', 'tables', 'figures', 'qa']:
        (HERE/name).mkdir(exist_ok=True)
    check()
    print('Protocol and source hashes frozen: 3 branches x 3 folds = 9 new fits.', flush=True)


def check():
    m = read(HERE/'manifest.json')
    assert all(sha256(ROOT/p) == h for p, h in m['hashes'].items()), 'Frozen input/code changed'
    assert all(read(HERE.parent/'catboost_progression/qa/feature_validation.json')['checks'].values())
    engine.HERE = HERE
    engine.P = read(HERE/'protocol.json')
    return engine.P


def fit(family):
    p = check()
    data = pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl')
    for b in p['branches']:
        if b['family'] == family:
            engine.fit_run(family, b['run_id'], b['config'], b['features'], data)
    print(f'{family}: all predeclared folds complete.', flush=True)


def finalize():
    p = check()
    assert all((HERE/'runs'/b['run_id']/'result.json').exists() for b in p['branches'])
    # All fits and preceding-window thresholds exist before the reused June diagnostic.
    lock = HERE/'fit_lock.json'
    if not lock.exists():
        write_json(lock, dict(locked_utc=engine.stamp(), manifest_sha256=sha256(HERE/'manifest.json'),
            runs={b['run_id']: sha256(HERE/'runs'/b['run_id']/'result.json') for b in p['branches']},
            role='All predeclared fits locked; no selection and no June optimization.'))
    assert all(sha256(HERE/'runs'/k/'result.json') == v for k, v in read(lock)['runs'].items())
    data = pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl')
    june = data.loc[data.timestamp.ge('2025-06-01') & data.timestamp.lt('2025-07-01')]
    for b in p['branches']:
        folder = HERE/'runs'/b['run_id']
        path = folder/'June_result.json'
        if path.exists():
            assert read(path)['fit_lock_sha256'] == sha256(lock)
            continue
        r = read(folder/'result.json')
        model = load_model(folder/('D3'+suffix(b['family'])), b['family'])
        scores = predict(model, june, read(folder/'D3_schema.json'), b['family'])
        threshold = r['folds'][2]['threshold_decision']['threshold']
        out = june[['ID', 'unix_time', 'is_fraud']].copy()
        out['score'] = scores
        out['decision'] = (scores >= threshold).astype('int8')
        out.to_csv(folder/'June_predictions.csv.gz', index=False, compression='gzip')
        metrics = evaluate(june.is_fraud, scores, threshold)
        write_json(path, dict(run_id=b['run_id'], fit_lock_sha256=sha256(lock),
                             role='reused_diagnostic_not_blind_test', selected=metrics))
        print(f"June {b['run_id']}: U={metrics['utility']:,}; FP={metrics['FP']:,}", flush=True)
    write_json(HERE/'completion.json', dict(completed_utc=engine.stamp(),
        manifest_sha256=sha256(HERE/'manifest.json'), fit_lock_sha256=sha256(lock),
        branches=[b['run_id'] for b in p['branches']], new_fits=9))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--stage', choices=['prepare', 'catboost', 'xgboost', 'finalize'], required=True)
    stage = a.parse_args().stage
    if stage == 'prepare': prepare()
    elif stage == 'finalize': finalize()
    else: fit(stage)
