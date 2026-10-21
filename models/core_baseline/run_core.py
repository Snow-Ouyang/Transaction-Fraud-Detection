"""Frozen core-feature temporal baselines: L2 logistic and CatBoost.

Only amt, category, hour are model inputs. Official test data are not loaded.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'.deps'))

import joblib
import numpy as np
import pandas as pd
import sklearn
import catboost
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ['amt', 'category', 'hour']
TABLES = HERE/'tables'
ARTIFACTS = HERE/'artifacts'
PREDICTIONS = HERE/'predictions'


def log(text):
    print(text, flush=True)


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, data):
    if not isinstance(data, pd.DataFrame): data = pd.DataFrame(data)
    data.to_csv(TABLES/f'{name}.csv', index=False, encoding='utf-8-sig')
    return data


def utility(y, decision):
    y, decision = np.asarray(y, bool), np.asarray(decision, bool)
    tp, fp, fn = int(np.sum(y & decision)), int(np.sum(~y & decision)), int(np.sum(y & ~decision))
    return 50*tp-100*fn-5*fp


def select_threshold(y, p):
    """Exact score-tie-aware utility maximum; never receives evaluation labels."""
    y, p = np.asarray(y, dtype='int8'), np.asarray(p, dtype='float64')
    order = np.argsort(-p, kind='stable')
    ps, ys = p[order], y[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps)-1]
    count = np.r_[0, ends+1]
    gain = np.r_[0, np.cumsum(np.where(ys == 1, 150, -5), dtype='int64')[ends]]
    best = int(np.argmax(gain))  # first maximum = fewer flags
    k = int(count[best])
    if k == 0:
        threshold = float(np.nextafter(1., np.inf))
    elif k == len(ps):
        threshold = 0.
    else:
        threshold = float((ps[k-1]+ps[k])/2)
        if threshold <= ps[k]: threshold = float(ps[k-1])
    value = -100*int(y.sum())+int(gain[best])
    if utility(y, p >= threshold) != value:
        raise AssertionError('Threshold tie handling failed')
    return {'threshold': threshold, 'threshold_window_utility': value, 'flags': k,
            'delta_vs_always_normal': int(gain[best]), 'candidate_groups': len(count)}


def check_cost_logic():
    # Literal confusion cases and brute-force score boundaries independently check costs/ties.
    assert utility([1, 1, 0, 0], [1, 0, 1, 0]) == -55
    assert utility([1, 0], [0, 0]) == -100
    assert utility([1, 0], [1, 0]) == 50
    cases = [([0, 1, 0, 1], [.1, .1, .8, .9]), ([0, 0], [.2, .9]),
             ([1, 1], [.0, 1.]), ([0, 1, 1, 0, 0], [.5, .5, .5, .1, .9])]
    for y, p in cases:
        candidates = [0., np.nextafter(1., np.inf), *np.unique(p).tolist()]
        expected = max(utility(y, np.asarray(p) >= t) for t in candidates)
        assert select_threshold(y, p)['threshold_window_utility'] == expected
    return True


def evaluate(y, p, threshold):
    y = np.asarray(y, dtype='int8')
    decision = np.asarray(p) >= threshold
    tp = int(np.sum((y == 1) & decision)); fp = int(np.sum((y == 0) & decision))
    fn = int(np.sum((y == 1) & ~decision)); tn = int(np.sum((y == 0) & ~decision))
    pos, neg, n = int(y.sum()), int((y == 0).sum()), len(y)
    u = 50*tp-100*fn-5*fp
    return {'n': n, 'fraud': pos, 'normal': neg, 'threshold': threshold,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'flags': tp+fp,
            'utility': u, 'always_normal_utility': -100*pos, 'delta_vs_always_normal': u+100*pos,
            'utility_per_10000': u/n*10000, 'delta_per_10000': (u+100*pos)/n*10000,
            'precision': tp/(tp+fp) if tp+fp else np.nan, 'recall': tp/pos if pos else np.nan,
            'false_positive_rate': fp/neg if neg else np.nan,
            'false_positives_per_10000_normal': fp/neg*10000 if neg else np.nan,
            'accuracy': (tp+tn)/n,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            'average_precision': average_precision_score(y, p) if 0 < pos < n else np.nan,
            'roc_auc': roc_auc_score(y, p) if 0 < pos < n else np.nan,
            'logloss': log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[0, 1]),
            'brier_score': brier_score_loss(y, p),
            'true_positive_gain': 50*tp, 'false_negative_cost': -100*fn, 'false_positive_cost': -5*fp}


def make_logistic(protocol):
    p = protocol['logistic']
    amount = Pipeline([('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                       ('spline', SplineTransformer(n_knots=8, degree=3, knots='uniform',
                                                    extrapolation='linear', include_bias=False)),
                       ('scale', StandardScaler())])
    preprocess = ColumnTransformer([('amount', amount, ['amt']),
                                    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['category', 'hour'])])
    model = LogisticRegression(C=p['C'], penalty='l2', solver=p['solver'], max_iter=p['max_iter'],
                               tol=p['tol'], class_weight=None, random_state=protocol['seed'])
    return Pipeline([('preprocess', preprocess), ('classifier', model)])


def make_catboost(protocol):
    p = protocol['catboost']
    keys = ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'loss_function', 'boosting_type',
            'has_time', 'counter_calc_method', 'thread_count', 'task_type']
    return CatBoostClassifier(**{k: p[k] for k in keys}, cat_features=['category', 'hour'],
                             random_seed=protocol['seed'], allow_writing_files=False, verbose=100)


def load_training():
    d = pd.read_csv(ROOT/'fraudTrain.csv', usecols=['ID', 'unix_time', 'amt', 'category', 'is_fraud'],
                    dtype={'ID': 'int32', 'is_fraud': 'int8', 'category': 'string'})
    d['timestamp'] = pd.to_datetime(d.unix_time, unit='s')
    d['hour'] = d.timestamp.dt.hour.astype(str)
    d['category'] = d.category.astype(str)
    d['month'] = d.timestamp.dt.strftime('%Y-%m')
    d['week'] = d.timestamp.dt.to_period('W-SUN').astype(str)
    d['amount_band'] = pd.cut(d.amt, [-np.inf, 10, 50, 100, 200, 500, 1000, 1500, 2000, np.inf]).astype(str)
    return d.sort_values(['unix_time', 'ID'], kind='stable').reset_index(drop=True)


def run():
    start = time.time()
    for path in [TABLES, ARTIFACTS, PREDICTIONS, HERE/'figures']:
        path.mkdir(parents=True, exist_ok=True)
    if (HERE/'results.json').exists():
        raise RuntimeError('A completed v1 run exists. Preserve it; use a new experiment directory for new fits.')
    protocol_path = HERE/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol_digest = sha256(protocol_path)
    raw_digest = sha256(ROOT/'fraudTrain.csv')
    eda_manifest = json.loads((ROOT/'eda/input_manifest.json').read_text(encoding='utf-8'))
    assert raw_digest == eda_manifest['sources']['fraudTrain.csv']['sha256']
    source_files = [ROOT/'fraudTrain.csv', ROOT/'fraudTest.csv', ROOT/'ISYE 7406 Project Description.pdf',
                    ROOT/'eda/report_zh.md', ROOT/'eda/report_zh.html']
    input_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    check_cost_logic()
    log('Loading only training labels, three core inputs and alignment metadata ...')
    d = load_training()
    assert len(d) == 1402996 and set(d.is_fraud.unique()) == {0, 1}
    assert d.unix_time.is_monotonic_increasing
    assert set(FEATURES) == {'amt', 'category', 'hour'}
    metrics, slice_rows, monthly_rows, weekly_rows, threshold_rows = [], [], [], [], []
    fit_records, thresholds, support_records, calibration_rows, curve_rows = [], [], [], [], []
    response_rows, logit_coefficients, learning_rows, reload_checks, split_records = [], [], [], [], []
    all_eval_ids = set()
    final_eval = None
    for fold in protocol['folds']:
        name, cutoff, threshold_end = fold['name'], fold['fit_end_exclusive'], fold['threshold_end_exclusive']
        fit = d.loc[d.timestamp < cutoff]
        tune = d.loc[d.timestamp.ge(cutoff) & d.timestamp.lt(threshold_end)]
        assert fit.timestamp.max() < tune.timestamp.min()
        support = fit.groupby('category').is_fraud.agg(fit_n='size', fit_fraud='sum')
        support['fit_normal'] = support.fit_n-support.fit_fraud
        support_records.extend(support.reset_index().assign(fold=name).to_dict('records'))
        for role, data in [('fit', fit), ('threshold', tune)]:
            split_records.append({'fold': name, 'role': role, 'window': name+'_'+role,
                                  'n': len(data), 'fraud': int(data.is_fraud.sum()),
                                  'time_min': str(data.timestamp.min()), 'time_max': str(data.timestamp.max())})
        log(f'[{name}] fit={len(fit):,}; threshold={len(tune):,}; only {FEATURES}')
        models = {'Logistic': make_logistic(protocol), 'CatBoost': make_catboost(protocol)}
        chosen = {}
        tune_export = tune[['ID', 'unix_time', 'amt', 'category', 'hour', 'is_fraud']].copy()
        for model_name, model in models.items():
            tick = time.time()
            log(f'[{name}] fitting {model_name} ...')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                with threadpool_limits(limits=4):
                    model.fit(fit[FEATURES], fit.is_fraud)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence: raise RuntimeError('Logistic did not converge: '+str(convergence))
            duration = time.time()-tick
            if model_name == 'Logistic':
                path = ARTIFACTS/f'{name}_logistic.joblib'
                joblib.dump(model, path, compress=3)
                reloaded = joblib.load(path)
                names = model.named_steps['preprocess'].get_feature_names_out()
                coef = model.named_steps['classifier'].coef_[0]
                logit_coefficients.extend({'fold': name, 'term': term, 'coefficient': c} for term, c in zip(names, coef))
                iterations = int(model.named_steps['classifier'].n_iter_.max())
            else:
                path = ARTIFACTS/f'{name}_catboost.cbm'
                model.save_model(str(path))
                reloaded = CatBoostClassifier()
                reloaded.load_model(str(path))
                iterations = model.tree_count_
                learning_rows.extend({'fold': name, 'iteration': i+1, 'train_logloss': loss}
                                     for i, loss in enumerate(model.get_evals_result()['learn']['Logloss']))
            sample = tune[FEATURES].iloc[::max(1, len(tune)//1000)]
            reload_diff = float(np.max(np.abs(model.predict_proba(sample)[:, 1]-reloaded.predict_proba(sample)[:, 1])))
            if reload_diff > 1e-12: raise AssertionError('Saved model prediction mismatch')
            reload_checks.append({'fold': name, 'model': model_name, 'n': len(sample), 'maximum_abs_probability_difference': reload_diff})
            p = model.predict_proba(tune[FEATURES])[:, 1]
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            choice = select_threshold(tune.is_fraud, p)
            chosen[model_name] = choice
            record = {'fold': name, 'model': model_name, **choice}
            thresholds.append(record)
            # Write a decision record before evaluation-window predictions or metrics.
            write_json(ARTIFACTS/f'{name}_{model_name.lower()}_decision.json',
                       {'protocol_sha256': protocol_digest, 'fit_end_exclusive': cutoff,
                        'threshold_end_exclusive': threshold_end, 'model_sha256': sha256(path), **record})
            fit_records.append({'fold': name, 'model': model_name, 'fit_n': len(fit), 'fit_fraud': int(fit.is_fraud.sum()),
                                'fit_seconds': duration, 'iterations': iterations, 'model_file': str(path.relative_to(HERE)),
                                'model_sha256': sha256(path), 'converged': True})
            tune_export['p_'+model_name] = p
            tune_export['pred_'+model_name] = (p >= choice['threshold']).astype('int8')
            for policy, t in [('selected', choice['threshold']), ('default_0.5', .5), ('theoretical_5_over_155', 5/155)]:
                metrics.append({'fold': name, 'window': name+'_threshold', 'role': 'threshold_selection_in_sample',
                                'model': model_name, 'policy': policy, **evaluate(tune.is_fraud, p, t)})
            grid = np.unique(np.r_[np.geomspace(1e-5, .99, 161), .5, 5/155, choice['threshold'], 0., np.nextafter(1., np.inf)])
            for threshold in grid:
                threshold_rows.append({'fold': name, 'model': model_name, 'threshold': threshold,
                                       'utility': utility(tune.is_fraud, p >= threshold),
                                       'always_normal_utility': -100*int(tune.is_fraud.sum())})
            log(f'[{name}] {model_name} fitted in {duration:.1f}s; threshold locked at {choice["threshold"]:.8g}; threshold-window utility={choice["threshold_window_utility"]:,}')
        tune_export.to_csv(PREDICTIONS/f'{name}_threshold.csv.gz', index=False, compression='gzip')

        for window in fold['evaluations']:
            evaluation = d.loc[d.timestamp.ge(window['start']) & d.timestamp.lt(window['end_exclusive'])].copy()
            assert tune.timestamp.max() < evaluation.timestamp.min()
            assert not (set(evaluation.ID) & all_eval_ids)
            all_eval_ids.update(evaluation.ID.tolist())
            evaluation['fit_normal_support'] = evaluation.category.map(support.fit_normal).fillna(0).astype(int)
            evaluation['support_group'] = np.where(evaluation.fit_normal_support.eq(0), 'zero_normal_support_in_fit', 'normal_support_in_fit')
            split_records.append({'fold': name, 'role': window['role'], 'window': window['name'],
                                  'n': len(evaluation), 'fraud': int(evaluation.is_fraud.sum()),
                                  'time_min': str(evaluation.timestamp.min()), 'time_max': str(evaluation.timestamp.max())})
            exported = evaluation[['ID', 'unix_time', 'amt', 'category', 'hour', 'is_fraud', 'fit_normal_support']].copy()
            y = evaluation.is_fraud.to_numpy()
            for model_name, model in models.items():
                p = model.predict_proba(evaluation[FEATURES])[:, 1]
                assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                t = chosen[model_name]['threshold']
                exported['p_'+model_name] = p
                exported['pred_'+model_name] = (p >= t).astype('int8')
                for policy, threshold in [('selected', t), ('default_0.5', .5), ('theoretical_5_over_155', 5/155)]:
                    result = evaluate(y, p, threshold)
                    assert result['TP']+result['FP']+result['FN']+result['TN'] == len(evaluation)
                    assert result['utility'] == utility(y, p >= threshold)
                    metrics.append({'fold': name, 'window': window['name'], 'role': window['role'],
                                    'model': model_name, 'policy': policy, **result})
                evaluation['p'] = p
                for field in ['month', 'week', 'category', 'hour', 'amount_band', 'support_group']:
                    for value, group in evaluation.groupby(field, observed=True):
                        row = {'fold': name, 'window': window['name'], 'model': model_name,
                               'slice_feature': field, 'slice_value': str(value),
                               **evaluate(group.is_fraud, group.p, t)}
                        slice_rows.append(row)
                        if field == 'month': monthly_rows.append(row)
                        if field == 'week': weekly_rows.append(row)
                if name == 'D3':
                    for data_role, frame, values in [('threshold_April', tune, tune_export['p_'+model_name].to_numpy()),
                                                    (window['name'], evaluation, p)]:
                        # Repeated threshold rows across the two D3 evaluations are dropped on export.
                        bins = pd.cut(values, np.r_[0, np.geomspace(1e-4, .5, 10), 1.], include_lowest=True)
                        tmp = pd.DataFrame({'bin': bins, 'p': values, 'y': frame.is_fraud.to_numpy()})
                        for bin_name, group in tmp.groupby('bin', observed=True):
                            calibration_rows.append({'role': data_role, 'model': model_name, 'bin': str(bin_name),
                                                     'n': len(group), 'mean_score': group.p.mean(), 'fraud_rate': group.y.mean()})
                log(f'[{window["name"]}] {model_name}: U={utility(y, p >= t):,}; TP={int(np.sum((y == 1) & (p >= t))):,}; FP={int(np.sum((y == 0) & (p >= t))):,}')
            exported.to_csv(PREDICTIONS/f'{window["name"]}.csv.gz', index=False, compression='gzip')
            if window['name'] == 'D3_June':
                final_eval = exported
        if name == 'D3':
            amounts = np.geomspace(1, 10000, 160)
            for model_name, model in models.items():
                for category in ['gas_transport', 'grocery_pos', 'shopping_net']:
                    for hour in ['14', '23']:
                        features = pd.DataFrame({'amt': amounts, 'category': category, 'hour': hour})
                        scores = model.predict_proba(features[FEATURES])[:, 1]
                        response_rows.extend({'model': model_name, 'category': category, 'hour': hour, 'amt': a, 'score': p}
                                             for a, p in zip(amounts, scores))

    table('metrics', metrics)
    table('slices', slice_rows)
    table('monthly_metrics', monthly_rows)
    table('weekly_metrics', weekly_rows)
    table('thresholds', thresholds)
    table('threshold_curves', threshold_rows)
    table('fit_records', fit_records)
    table('fit_category_support', support_records)
    table('splits', split_records)
    table('logistic_coefficients', logit_coefficients)
    table('catboost_training_loss', learning_rows)
    table('model_reload_checks', reload_checks)
    table('calibration_diagnostic', pd.DataFrame(calibration_rows).drop_duplicates())
    table('amount_response', response_rows)
    assert final_eval is not None
    reload_metric_checks = []
    final_selected = pd.DataFrame(metrics).query("window == 'D3_June' and policy == 'selected'")
    for name in ['Logistic', 'CatBoost']:
        ref = final_selected.loc[final_selected.model == name].iloc[0]
        file_data = pd.read_csv(PREDICTIONS/'D3_June.csv.gz')
        decision = file_data['pred_'+name].to_numpy(dtype=bool)
        recomputed = utility(file_data.is_fraud, decision)
        reload_metric_checks.append({'model': name, 'utility_from_saved_decisions': recomputed,
                                     'utility_matches_report': recomputed == int(ref.utility)})
    table('saved_prediction_metric_checks', reload_metric_checks)
    checks = {
        'frozen_protocol_unchanged': sha256(protocol_path) == protocol_digest,
        'input_and_eda_artifacts_unchanged': all(sha256(ROOT/name) == value for name, value in input_hashes.items()),
        'utility_and_threshold_tie_checks': True,
        'all_models_converged': all(x['converged'] for x in fit_records),
        'all_model_reload_probabilities_match': all(x['maximum_abs_probability_difference'] <= 1e-12 for x in reload_checks),
        'saved_prediction_utilities_match': all(x['utility_matches_report'] for x in reload_metric_checks),
        'only_three_core_model_inputs': FEATURES == ['amt', 'category', 'hour'],
        'time_splits_strict_and_evaluations_disjoint': True,
        'no_official_test_predictions': not list(PREDICTIONS.glob('*test*')),
        'all_eval_metrics_reconcile': all(x['TP']+x['FP']+x['FN']+x['TN'] == x['n'] for x in metrics),
    }
    assert all(checks.values())
    results = {'experiment': protocol['experiment'], 'features': FEATURES, 'protocol_sha256': protocol_digest,
               'train_source_sha256': raw_digest, 'source_hashes': input_hashes, 'script_sha256': sha256(Path(__file__)),
               'runtime': {'python': platform.python_version(), 'executable': sys.executable,
                           'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'catboost': catboost.__version__},
               'elapsed_seconds': time.time()-start, 'checks': checks,
               'final_selected_metrics': final_selected.to_dict('records'),
               'selected_thresholds': thresholds,
               'scope': 'no expanded features, no XGBoost, no official test predictions or submissions; June already observed descriptively in EDA'}
    write_json(HERE/'results.json', results)
    log('CORE MODEL RUN COMPLETE')
    log(json.dumps(safe({'final_selected_metrics': results['final_selected_metrics'], 'elapsed_seconds': results['elapsed_seconds']}), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'report', 'all'], default='all')
    args = parser.parse_args()
    if args.stage in ['train', 'all']: run()
    if args.stage in ['report', 'all']:
        from report_core import render_report
        render_report(HERE)
