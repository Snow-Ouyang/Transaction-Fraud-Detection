"""Small CatBoost grid, then development-selected forward feature groups.

No Logistic refits, XGBoost, official test predictions, or post-June decisions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = HERE.parent/'core_baseline'
sys.path.insert(0, str(CORE/'.deps'))
sys.path.insert(0, str(CORE))

import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from run_core import select_threshold, evaluate, utility, sha256, write_json, safe, check_cost_logic
from features import build_features

RUNS = HERE/'runs'
TABLES = HERE/'tables'
PROTOCOL = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
CORE_FEATURES = PROTOCOL['core_features']
CAT_COLUMNS = PROTOCOL['categorical_features_if_present']


def log(message):
    line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line, flush=True)
    with open(HERE/'progress.log', 'a', encoding='utf-8') as f:
        f.write(line+'\n')


def csv(name, records):
    if not isinstance(records, pd.DataFrame): records = pd.DataFrame(records)
    records.to_csv(TABLES/f'{name}.csv', index=False, encoding='utf-8-sig')
    return records


def hash_object(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True).encode()).hexdigest()


def prepare():
    for path in [RUNS, TABLES, HERE/'figures', HERE/'qa']:
        path.mkdir(parents=True, exist_ok=True)
    previous = json.loads((CORE/'results.json').read_text(encoding='utf-8'))
    raw_hash = sha256(ROOT/'fraudTrain.csv')
    assert raw_hash == previous['train_source_sha256']
    assert catboost.__version__ == previous['runtime']['catboost'] == '1.2.8'
    sources = {name: sha256(ROOT/name) for name in ['fraudTrain.csv', 'fraudTest.csv', 'ISYE 7406 Project Description.pdf',
                'eda/report_zh.md', 'eda/report_zh.html', 'models/core_baseline/results.json',
                'models/core_baseline/protocol.json', 'models/core_baseline/tables/metrics.csv',
                'models/core_baseline/report_zh.md', 'models/core_baseline/report_zh.html']}
    signature = {'protocol_sha256': sha256(HERE/'protocol.json'), 'source_sha256': raw_hash,
                 'feature_code_sha256': sha256(HERE/'features.py'), 'runner_sha256': sha256(Path(__file__))}
    marker = HERE/'experiment_manifest.json'
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        assert old['signature'] == signature, 'Frozen input/protocol/code changed; preserve run and start a new directory'
        assert old['source_hashes'] == sources
    else:
        write_json(marker, {'signature': signature, 'source_hashes': sources,
                           'started_utc': datetime.now(timezone.utc).isoformat(),
                           'runtime': {'python': platform.python_version(), 'executable': sys.executable,
                                       'pandas': pd.__version__, 'numpy': np.__version__, 'catboost': catboost.__version__}})
    cache = HERE/'feature_cache.pkl'
    if cache.exists():
        log('Reusing audited label-free feature cache')
        d = pd.read_pickle(cache)
        audit = json.loads((HERE/'qa/feature_validation.json').read_text(encoding='utf-8'))
        assert all(audit['checks'].values())
    else:
        log('Building strictly past features; all raw training records, no official test table')
        d, audit = build_features(ROOT/'fraudTrain.csv', HERE/'qa')
        d.to_pickle(cache)
        log(f'Features ready; {len(d):,} rows; independent prefix and direct-past checks passed')
    assert d.unix_time.is_monotonic_increasing
    check_cost_logic()
    return d, signature, sources


def model_parameters(config):
    return {**PROTOCOL['fixed_parameters'], 'depth': int(config['depth']), 'l2_leaf_reg': float(config['l2_leaf_reg']),
            'random_seed': PROTOCOL['seed'], 'allow_writing_files': False}


def fit_run(run_id, config, feature_names, data, signature):
    folder = RUNS/run_id
    folder.mkdir(parents=True, exist_ok=True)
    run_signature = hash_object({'signature': signature, 'parameters': model_parameters(config), 'features': feature_names})
    result_path = folder/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['run_signature'] == run_signature
        for fold in result['folds']:
            assert sha256(folder/f'{fold["fold"]}.cbm') == fold['model_sha256']
        log(f'[{run_id}] resuming completed development results')
        return result
    folds = []
    categoricals = [v for v in CAT_COLUMNS if v in feature_names]
    for fold in PROTOCOL['folds']:
        name, cutoff, t_end = fold['name'], fold['fit_end_exclusive'], fold['threshold_end_exclusive']
        fold_path = folder/f'{name}_result.json'
        if fold_path.exists():
            done = json.loads(fold_path.read_text(encoding='utf-8'))
            assert done['run_signature'] == run_signature
            assert sha256(folder/f'{name}.cbm') == done['model_sha256']
            folds.append(done)
            log(f'[{run_id}/{name}] resuming completed fold')
            continue
        fit = data.loc[data.timestamp < cutoff]
        tune = data.loc[data.timestamp.ge(cutoff) & data.timestamp.lt(t_end)]
        dev = data.loc[data.timestamp.ge(fold['development_start']) & data.timestamp.lt(fold['development_end_exclusive'])]
        assert fit.timestamp.max() < tune.timestamp.min() and tune.timestamp.max() < dev.timestamp.min()
        model_path = folder/f'{name}.cbm'
        meta_path = folder/f'{name}_model_meta.json'
        tick = time.time()
        reused = bool(config.get('reuse_core_v1_models', False))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            assert meta['run_signature'] == run_signature
            assert sha256(model_path) == meta['model_sha256']
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            fit_seconds = meta['fit_seconds']
        else:
            if reused:
                assert feature_names == CORE_FEATURES and config['depth'] == 6 and config['l2_leaf_reg'] == 3
                original = CORE/'artifacts'/f'{name}_catboost.cbm'
                shutil.copy2(original, model_path)
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                fit_seconds = 0.
                log(f'[{run_id}/{name}] reused original fixed core model')
            else:
                log(f'[{run_id}/{name}] fitting depth={config["depth"]}, L2={config["l2_leaf_reg"]}; {len(feature_names)} inputs; n={len(fit):,}')
                model = CatBoostClassifier(**model_parameters(config), cat_features=categoricals, verbose=100)
                model.fit(fit[feature_names], fit.is_fraud)
                model.save_model(str(model_path))
                fit_seconds = time.time()-tick
            meta = {'run_signature': run_signature, 'model_sha256': sha256(model_path), 'fit_seconds': fit_seconds,
                    'features': feature_names, 'parameters': model_parameters(config), 'reused_core_v1': reused}
            write_json(meta_path, meta)
        assert model.feature_names_ == feature_names
        p_tune = model.predict_proba(tune[feature_names])[:, 1]
        assert np.isfinite(p_tune).all() and np.all((p_tune >= 0) & (p_tune <= 1))
        decision = select_threshold(tune.is_fraud, p_tune)
        threshold = decision['threshold']
        write_json(folder/f'{name}_decision.json', {'run_signature': run_signature,
                   'fit_end_exclusive': cutoff, 'threshold_end_exclusive': t_end,
                   'model_sha256': sha256(model_path), **decision})
        tune_frame = tune[['ID', 'unix_time', 'is_fraud']].copy()
        tune_frame['score'], tune_frame['decision'] = p_tune, (p_tune >= threshold).astype('int8')
        tune_frame.to_csv(folder/f'{name}_threshold.csv.gz', index=False, compression='gzip')
        # No evaluation score is produced until the threshold decision has been written.
        p_dev = model.predict_proba(dev[feature_names])[:, 1]
        assert np.isfinite(p_dev).all() and np.all((p_dev >= 0) & (p_dev <= 1))
        dev_frame = dev[['ID', 'unix_time', 'is_fraud']].copy()
        dev_frame['score'], dev_frame['decision'] = p_dev, (p_dev >= threshold).astype('int8')
        dev_frame.to_csv(folder/f'{name}_development.csv.gz', index=False, compression='gzip')
        support = fit.groupby('category').is_fraud.agg(n='size', fraud='sum')
        support['normal'] = support.n-support.fraud
        support.to_csv(folder/f'{name}_category_support.csv', encoding='utf-8-sig')
        significance = model.get_feature_importance(type='PredictionValuesChange')
        pd.DataFrame({'feature': feature_names, 'importance': significance}).to_csv(
            folder/f'{name}_feature_importance.csv', index=False, encoding='utf-8-sig')
        reload = CatBoostClassifier()
        reload.load_model(str(model_path))
        sample = dev[feature_names].iloc[::max(1, len(dev)//1000)]
        difference = float(np.max(np.abs(model.predict_proba(sample)[:, 1]-reload.predict_proba(sample)[:, 1])))
        assert difference <= 1e-12
        record = {'fold': name, 'run_signature': run_signature, 'fit_n': len(fit), 'fit_fraud': int(fit.is_fraud.sum()),
                  'threshold_n': len(tune), 'development_n': len(dev), 'fit_seconds': fit_seconds,
                  'reused_core_v1': reused, 'model_sha256': sha256(model_path),
                  'reload_probability_max_difference': difference, 'decision': decision,
                  'threshold_metrics': evaluate(tune.is_fraud, p_tune, threshold),
                  'development_metrics': evaluate(dev.is_fraud, p_dev, threshold)}
        write_json(fold_path, record)
        folds.append(record)
        log(f'[{run_id}/{name}] development U={record["development_metrics"]["utility"]:,}; FP={record["development_metrics"]["FP"]:,}; threshold={threshold:.6g}')
    scores = [f['development_metrics']['delta_per_10000'] for f in folds]
    result = {'run_id': run_id, 'run_signature': run_signature, 'parameters': model_parameters(config),
              'features': feature_names, 'categorical_features': categoricals, 'folds': folds,
              'mean_development_delta_per_10000': float(np.mean(scores)),
              'worst_development_delta_per_10000': float(np.min(scores)),
              'development_windows_above_always_normal': int(sum(v > 0 for v in scores))}
    write_json(result_path, result)
    return result


def paired_change(candidate, parent):
    p = {f['fold']: f['development_metrics'] for f in parent['folds']}
    values = {f['fold']: f['development_metrics']['utility_per_10000']-p[f['fold']]['utility_per_10000']
              for f in candidate['folds']}
    mean = float(np.mean(list(values.values())))
    wins = sum(v > 1e-9 for v in values.values())
    return values, mean, wins, mean > 1e-9 and wins >= 2


def june_diagnostic(run, data):
    folder = RUNS/run['run_id']
    lock = json.loads((HERE/'selection_lock.json').read_text(encoding='utf-8'))
    assert run['run_id'] in lock['june_diagnostic_runs']
    path = folder/'June_diagnostic.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['selection_lock_sha256'] == sha256(HERE/'selection_lock.json')
        return result
    fold = next(f for f in run['folds'] if f['fold'] == 'D3')
    t = fold['decision']['threshold']
    june = data.loc[data.timestamp.ge('2025-06-01') & data.timestamp.lt('2025-07-01')].copy()
    model = CatBoostClassifier()
    model.load_model(str(folder/'D3.cbm'))
    p = model.predict_proba(june[run['features']])[:, 1]
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    exported = june[['ID', 'unix_time', 'is_fraud']].copy()
    exported['score'], exported['decision'] = p, (p >= t).astype('int8')
    exported.to_csv(folder/'June_diagnostic.csv.gz', index=False, compression='gzip')
    june['score'] = p
    support = pd.read_csv(folder/'D3_category_support.csv').set_index('category')
    june['support_group'] = np.where(june.category.map(support.normal).fillna(0).eq(0),
                                    'zero_normal_support_in_fit', 'normal_support_in_fit')
    june['amount_band'] = pd.cut(june.amt, [-np.inf, 10, 50, 100, 200, 500, 1000, 1500, 2000, np.inf]).astype(str)
    slices = []
    for column in ['category', 'hour', 'month', 'week', 'amount_band', 'support_group', 'has_customer_history']:
        for value, group in june.groupby(column, observed=True):
            slices.append({'run_id': run['run_id'], 'slice_feature': column, 'slice_value': str(value),
                           **evaluate(group.is_fraud, group.score, t)})
    pd.DataFrame(slices).to_csv(folder/'June_slices.csv', index=False, encoding='utf-8-sig')
    policies = {name: evaluate(june.is_fraud, p, threshold) for name, threshold in
                [('selected', t), ('default_0.5', .5), ('theoretical_5_over_155', 5/155)]}
    result = {'run_id': run['run_id'], 'selection_lock_sha256': sha256(HERE/'selection_lock.json'),
              'role': 'reused_diagnostic_not_selection_or_blind_test', **policies}
    write_json(path, result)
    log(f'[JUNE DIAGNOSTIC/{run["run_id"]}] U={policies["selected"]["utility"]:,}; FP={policies["selected"]["FP"]:,}; TP={policies["selected"]["TP"]:,}')
    return result


def run():
    if (HERE/'results.json').exists():
        raise RuntimeError('Completed experiment exists. Preserve results and use a new directory for new searches.')
    started = time.time()
    data, signature, sources = prepare()
    runs = {}
    tuning = []
    for config in PROTOCOL['parameter_candidates']:
        result = fit_run(config['id'], config, CORE_FEATURES, data, signature)
        runs[result['run_id']] = result
        tuning.append({'run_id': result['run_id'], 'depth': config['depth'], 'l2_leaf_reg': config['l2_leaf_reg'],
                       'mean_delta_per_10000': result['mean_development_delta_per_10000'],
                       'worst_delta_per_10000': result['worst_development_delta_per_10000'],
                       'windows_above_always_normal': result['development_windows_above_always_normal']})
        csv('parameter_grid', tuning)
    ranked = sorted(tuning, key=lambda r: (-r['mean_delta_per_10000'], -r['worst_delta_per_10000'], r['depth'], -r['l2_leaf_reg']))
    tuned_id = ranked[0]['run_id']
    config = next(p for p in PROTOCOL['parameter_candidates'] if p['id'] == tuned_id).copy()
    config.pop('reuse_core_v1_models', None)
    current = runs[tuned_id]
    write_json(HERE/'parameter_decision.json', {'selected_run': tuned_id, 'ranked_candidates': ranked,
                                               'signature': signature, 'selected_before_feature_trials': True})
    log(f'PARAMETER CHOICE LOCKED: {tuned_id}; mean development delta/10k={ranked[0]["mean_delta_per_10000"]:.2f}')
    progression = []
    for group in PROTOCOL['feature_groups']:
        parent = current
        names = parent['features']+group['add']
        assert len(names) == len(set(names))
        trial = fit_run(group['id'], config, names, data, signature)
        runs[trial['run_id']] = trial
        changes, mean, wins, retained = paired_change(trial, parent)
        row = {'step': group['id'], 'group_name': group['name_zh'], 'parent_run': parent['run_id'],
               'added_features': ', '.join(group['add']), 'candidate_feature_count': len(names),
               'mean_paired_delta_per_10000': mean, 'improved_windows': wins,
               'D1_change_per_10000': changes['D1'], 'D2_change_per_10000': changes['D2'], 'D3_change_per_10000': changes['D3'],
               'retained': retained, 'retained_run_after_step': trial['run_id'] if retained else parent['run_id']}
        progression.append(row)
        if retained: current = trial
        csv('feature_progression', progression)
        write_json(HERE/'feature_decisions.json', {'steps': progression, 'current_retained_run': current['run_id'],
                                                'signature': signature, 'no_june_predictions_generated': True})
        log(f'FEATURE DECISION {group["id"]}: {"RETAIN" if retained else "REJECT"}; paired mean/10k={mean:+.2f}; wins={wins}/3; current={current["run_id"]}')

    # All parameter and feature decisions precede the first June score in this experiment.
    diagnostic_ids = list(dict.fromkeys(['core_default', tuned_id]+[g['id'] for g in PROTOCOL['feature_groups']]))
    selection_lock = {'signature': signature, 'parameter_selected_run': tuned_id, 'final_retained_run': current['run_id'],
                      'final_features': current['features'], 'feature_decisions': progression,
                      'june_diagnostic_runs': diagnostic_ids, 'locked_utc': datetime.now(timezone.utc).isoformat(),
                      'june_not_used_for_current_run_selection': True, 'june_previously_viewed_and_evaluated': True}
    if (HERE/'selection_lock.json').exists():
        existing = json.loads((HERE/'selection_lock.json').read_text(encoding='utf-8'))
        assert existing['signature'] == signature and existing['final_retained_run'] == current['run_id']
    else:
        write_json(HERE/'selection_lock.json', selection_lock)
    lock_hash = sha256(HERE/'selection_lock.json')
    log('ALL SELECTIONS LOCKED; only now evaluating previously seen June as a reused diagnostic')
    diagnostics = [june_diagnostic(runs[name], data) for name in diagnostic_ids]

    metrics, fits, paired = [], [], []
    for run_id, result in runs.items():
        for fold in result['folds']:
            for role, key in [('threshold_selection_in_sample', 'threshold_metrics'), ('development_selection', 'development_metrics')]:
                metrics.append({'run_id': run_id, 'fold': fold['fold'], 'role': role,
                                'feature_count': len(result['features']), **fold[key]})
            fits.append({'run_id': run_id, 'fold': fold['fold'], 'fit_n': fold['fit_n'], 'fit_seconds': fold['fit_seconds'],
                         'feature_count': len(result['features']), 'reused_core_v1': fold['reused_core_v1'],
                         'model_sha256': fold['model_sha256'], 'reload_difference': fold['reload_probability_max_difference']})
    for diagnostic in diagnostics:
        for policy in ['selected', 'default_0.5', 'theoretical_5_over_155']:
            metrics.append({'run_id': diagnostic['run_id'], 'fold': 'D3_June', 'role': 'reused_june_diagnostic',
                            'policy': policy, 'feature_count': len(runs[diagnostic['run_id']]['features']), **diagnostic[policy]})
    csv('metrics', metrics)
    csv('fit_records', fits)
    csv('june_diagnostic', [{'run_id': v['run_id'], **v['selected']} for v in diagnostics])
    csv('run_features', [{'run_id': name, 'feature_count': len(r['features']), 'features': ', '.join(r['features']),
                          'depth': r['parameters']['depth'], 'l2_leaf_reg': r['parameters']['l2_leaf_reg'],
                          'mean_development_delta_per_10000': r['mean_development_delta_per_10000']} for name, r in runs.items()])
    previous = pd.read_csv(CORE/'tables/metrics.csv')
    default = runs['core_default']
    reuse_matches = []
    old_windows = {'D1': 'D1_NovDec', 'D2': 'D2_FebMar', 'D3': 'D3_May'}
    for fold in default['folds']:
        old = previous.loc[(previous.model == 'CatBoost') & (previous.policy == 'selected') & (previous.window == old_windows[fold['fold']])].iloc[0]
        reuse_matches.append(int(old.utility) == fold['development_metrics']['utility'])
    manifest = json.loads((HERE/'experiment_manifest.json').read_text(encoding='utf-8'))
    checks = {
        'frozen_protocol_and_codes_unchanged': signature == manifest['signature'] and signature['protocol_sha256'] == sha256(HERE/'protocol.json'),
        'all_source_and_prior_artifacts_unchanged': all(sha256(ROOT/name) == digest for name, digest in sources.items()),
        'reused_default_reproduces_v1_development_utilities': all(reuse_matches),
        'all_saved_model_reload_scores_match': all(f['reload_difference'] <= 1e-12 for f in fits),
        'selection_lock_unchanged_after_june': sha256(HERE/'selection_lock.json') == lock_hash,
        'only_four_parameter_candidates': len(tuning) == 4,
        'all_five_feature_groups_completed': len(progression) == 5,
        'history_validation_passed': all(json.loads((HERE/'qa/feature_validation.json').read_text(encoding='utf-8'))['checks'].values()),
        'no_xgboost_or_logistic_refits': True,
        'no_official_test_predictions': True,
    }
    assert all(checks.values())
    final_june = next(v for v in diagnostics if v['run_id'] == current['run_id'])
    result = {'experiment': PROTOCOL['experiment'], 'signature': signature, 'checks': checks,
              'selected_parameter_run': tuned_id, 'selected_parameters': model_parameters(config),
              'final_retained_run': current['run_id'], 'final_features': current['features'],
              'feature_decisions': progression, 'parameter_grid': ranked,
              'final_development_mean_delta_per_10000': current['mean_development_delta_per_10000'],
              'final_development_positive_windows': current['development_windows_above_always_normal'],
              'final_june_reused_diagnostic': final_june['selected'], 'selection_lock_sha256': lock_hash,
              'source_hashes': sources, 'run_ids': list(runs), 'new_model_fits': sum(not f['reused_core_v1'] for f in fits),
              'reused_model_fits': sum(f['reused_core_v1'] for f in fits), 'total_model_checkpoints': len(fits),
              'elapsed_seconds': time.time()-started, 'runtime': manifest['runtime'],
              'limitations': ['development windows used for parameter and feature selection',
                              'June already observed and evaluated in prior version, reused diagnostic only',
                              'no new independent test set, official test performance unknown']}
    write_json(HERE/'results.json', result)
    log('PROGRESSION EXPERIMENT COMPLETE')
    log(json.dumps(safe({'selected_parameters': result['selected_parameters'], 'final_retained_run': current['run_id'],
                        'final_features': current['features'], 'final_june': final_june['selected'],
                        'elapsed_seconds': result['elapsed_seconds']}), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'report', 'all'], default='all')
    args = parser.parse_args()
    if args.stage in ['train', 'all']: run()
    if args.stage in ['report', 'all']:
        from report_progression import render_report
        render_report(HERE)
