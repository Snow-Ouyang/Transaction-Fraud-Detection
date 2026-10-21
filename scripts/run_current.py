"""Clean-clone reproduction of frozen CatBoost16/XGBoost15; no reports or test predictions."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/core_baseline'))
sys.path.insert(0, str(ROOT / 'models/capacity_comparison'))
from run_core import evaluate, select_threshold, sha256, write_json
from model_io import schema_for, transform, make_model, predict, suffix
from verify_data import verify


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def prepare(folder):
    verify()
    old = module('past_history_builder', ROOT / 'models/catboost_progression/features.py')
    context = module('behavior_context_builder', ROOT / 'models/behavior_context/features.py')
    source = ROOT / 'fraudTrain.csv'
    signature = {
        'train': sha256(source),
        'history_code': sha256(ROOT / 'models/catboost_progression/features.py'),
        'context_code': sha256(ROOT / 'models/behavior_context/features.py'),
        'entry_code': sha256(Path(__file__)),
        'pandas': pd.__version__, 'numpy': np.__version__,
    }
    cache = folder / 'features.pkl'
    marker = folder / 'feature_signature.json'
    if cache.exists() or marker.exists():
        if not (cache.exists() and marker.exists()):
            raise RuntimeError('Incomplete cache; use a new --output directory')
        if json.loads(marker.read_text()) != signature:
            raise RuntimeError('Cache inputs changed; use a new --output directory')
        return cache
    data, _ = old.build_features(source, folder / 'qa/base_history')
    extra = context.behavior_features(data[context.FIELDS])
    audit = context.audit(data[context.FIELDS], extra, folder / 'qa/context_history')
    if not all(audit['checks'].values()):
        raise AssertionError(audit)
    assert np.array_equal(data.ID, extra.ID)
    for col in context.NEW:
        data[col] = extra[col].to_numpy()
    for col in data.select_dtypes(include=['object', 'string']).columns:
        data[col] = data[col].astype('category')
    data.to_pickle(cache)
    write_json(folder / 'qa/context_history/feature_validation.json', audit)
    write_json(marker, signature)
    return cache


def fit_job(job):
    cache, output, family, fold_name = job
    protocol = json.loads((ROOT / 'configs/current_models.json').read_text(encoding='utf-8'))
    entry = protocol['models'][family]
    fp = next(f for f in protocol['folds'] if f['name'] == fold_name)
    folder = Path(output) / family / fold_name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        raise FileExistsError(f'Results already exist: {folder}; use a new --output')
    data = pd.read_pickle(cache)
    fit = data.loc[data.timestamp.lt(fp['fit_end'])]
    cal = data.loc[data.timestamp.ge(fp['fit_end']) & data.timestamp.lt(fp['threshold_end'])]
    dev = data.loc[data.timestamp.ge(fp['dev_start']) & data.timestamp.lt(fp['dev_end'])]
    schema = schema_for(fit, entry['features'], protocol['categorical_features'])
    model = make_model(protocol, family, entry['config'], entry['features'])
    print(f'FIT {family}/{fold_name}: {len(fit):,} rows, {len(entry["features"])} inputs', flush=True)
    model.fit(transform(fit, schema, family), fit.is_fraud)
    model.save_model(str(folder / ('model' + suffix(family))))
    write_json(folder / 'schema.json', schema)
    cal_score = predict(model, cal, schema, family)
    choice = select_threshold(cal.is_fraud, cal_score)
    threshold = choice['threshold']
    dev_score = predict(model, dev, schema, family)
    for name, frame, score in [('calibration', cal, cal_score), ('development', dev, dev_score)]:
        saved = frame[['ID', 'unix_time', 'is_fraud']].copy()
        saved['score'] = score
        saved['decision'] = (score >= threshold).astype('int8')
        saved.to_csv(folder / f'{name}.csv.gz', index=False, compression='gzip')
    result = dict(family=family, fold=fold_name, features=entry['features'], config=entry['config'],
                  fit_n=len(fit), threshold_decision=choice,
                  calibration=evaluate(cal.is_fraud, cal_score, threshold),
                  development=evaluate(dev.is_fraud, dev_score, threshold))
    write_json(folder / 'result.json', result)
    print(f'DONE {family}/{fold_name}: score={result["development"]["utility"]}', flush=True)
    return result


def finalize(folder, families, records):
    p = json.loads((ROOT / 'configs/current_models.json').read_text())
    data = pd.read_pickle(folder / 'features.pkl')
    june = data.loc[data.timestamp.ge('2025-06-01') & data.timestamp.lt('2025-07-01')]
    from model_io import load_model
    summary = {}
    for family in families:
        rows = [r for r in records if r['family'] == family]
        d3 = next(r for r in rows if r['fold'] == 'D3')
        location = folder / family / 'D3'
        model = load_model(location / ('model' + suffix(family)), family)
        score = predict(model, june, json.loads((location / 'schema.json').read_text()), family)
        m = evaluate(june.is_fraud, score, d3['threshold_decision']['threshold'])
        write_json(location / 'June_result.json', dict(role='previously_observed_diagnostic_only', selected=m))
        dev_score = [r['development']['utility'] for r in rows]
        summary[family] = dict(mean_score_per_10000=float(np.mean([r['development']['utility_per_10000'] for r in rows])),
                               pooled_score=sum(dev_score), june_score=m['utility'],
                               matches_recorded_development=dev_score == p['models'][family]['expected_development_scores'],
                               matches_recorded_june=m['utility'] == p['models'][family]['expected_june_score'])
    write_json(folder / 'summary.json', summary)
    text = '# Frozen current-model reproduction\n\nJune is reused diagnosis, not independent validation. Official test is not scored.\n\n'
    text += '| Family | Mean score/10000 | Total development score | June | Matches recorded scores |\n|---|---:|---:|---:|---|\n'
    for family, s in summary.items():
        text += f'| {family} | {s["mean_score_per_10000"]:.6f} | {s["pooled_score"]} | {s["june_score"]} | {s["matches_recorded_development"] and s["matches_recorded_june"]} |\n'
    (folder / 'RESULTS.md').write_text(text, encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/current')
    parser.add_argument('--family', choices=['both', 'catboost', 'xgboost'], default='both')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cache = prepare(folder)
    if args.prepare_only:
        print(f'Feature preparation complete: {cache}', flush=True)
        return
    families = ['catboost', 'xgboost'] if args.family == 'both' else [args.family]
    protocol = json.loads((ROOT / 'configs/current_models.json').read_text())
    jobs = [(str(cache), str(folder), family, fold['name']) for family in families for fold in protocol['folds']]
    write_json(folder / 'config_snapshot.json', protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(fit_job, jobs))
    finalize(folder, families, records)


if __name__ == '__main__':
    main()
