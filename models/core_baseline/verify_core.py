"""Independent raw-record, serialized-model and saved-decision checks; no fits."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'.deps'))
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from PIL import Image
from html.parser import HTMLParser


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''): h.update(part)
    return h.hexdigest()


def main():
    r = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    checks = {}
    checks['source_and_eda_hashes_match'] = all(digest(ROOT/name) == value for name, value in r['source_hashes'].items())
    checks['protocol_matches_frozen_run'] = digest(HERE/'protocol.json') == r['protocol_sha256']
    source = pd.read_csv(ROOT/'fraudTrain.csv', usecols=['ID', 'unix_time', 'amt', 'category', 'is_fraud']).set_index('ID')
    source['hour'] = pd.to_datetime(source.unix_time, unit='s').dt.hour.astype(str)
    metrics = pd.read_csv(HERE/'tables/metrics.csv')
    for path in sorted((HERE/'predictions').glob('*.csv.gz')):
        saved = pd.read_csv(path, dtype={'hour': str, 'category': str})
        original = source.loc[saved.ID]
        good = all(np.array_equal(saved[c].to_numpy(), original[c].to_numpy()) for c in ['unix_time', 'amt', 'category', 'hour', 'is_fraud'])
        checks[path.stem+'_all_raw_fields_aligned'] = good
        window = path.name.removesuffix('.csv.gz')
        for model in ['Logistic', 'CatBoost']:
            reference = metrics.loc[(metrics.window == window) & (metrics.model == model) & (metrics.policy == 'selected')]
            # Threshold windows are named consistently in metrics and prediction files.
            assert len(reference) == 1
            row = reference.iloc[0]
            y = saved.is_fraud.to_numpy()
            pred = saved['pred_'+model].to_numpy(dtype=bool)
            tp = int(((y == 1) & pred).sum())
            fp = int(((y == 0) & pred).sum())
            fn = int(((y == 1) & ~pred).sum())
            tn = int(((y == 0) & ~pred).sum())
            checks[window+'_'+model+'_independent_confusion_and_cost'] = (
                (tp, fp, fn, tn) == (int(row.TP), int(row.FP), int(row.FN), int(row.TN))
                and 50*tp-100*fn-5*fp == int(row.utility))

    june = pd.read_csv(HERE/'predictions/D3_June.csv.gz')
    x = source.loc[june.ID, ['amt', 'category', 'hour']].copy()
    thresholds = pd.read_csv(HERE/'tables/thresholds.csv').query("fold == 'D3'").set_index('model')
    for name in ['Logistic', 'CatBoost']:
        if name == 'Logistic':
            model = joblib.load(HERE/'artifacts/D3_logistic.joblib')
            checks[name+'_feature_names_only_core'] = list(model.feature_names_in_) == ['amt', 'category', 'hour']
        else:
            model = CatBoostClassifier()
            model.load_model(str(HERE/'artifacts/D3_catboost.cbm'))
            checks[name+'_feature_names_only_core'] = model.feature_names_ == ['amt', 'category', 'hour']
        p = model.predict_proba(x)[:, 1]
        checks[name+'_full_june_raw_source_rescore_matches'] = bool(np.allclose(p, june['p_'+name], rtol=0, atol=1e-12))
        checks[name+'_locked_threshold_decisions_match'] = bool(np.array_equal(p >= thresholds.loc[name, 'threshold'], june['pred_'+name].astype(bool)))
    for path in (HERE/'artifacts').glob('*_decision.json'):
        decision = json.loads(path.read_text(encoding='utf-8'))
        suffix = 'logistic.joblib' if decision['model'] == 'Logistic' else 'catboost.cbm'
        model_path = HERE/'artifacts'/f'{decision["fold"]}_{suffix}'
        checks[path.stem+'_hashes_match'] = (digest(model_path) == decision['model_sha256']
                                            and decision['protocol_sha256'] == r['protocol_sha256'])

    class Parser(HTMLParser):
        def __init__(self): super().__init__(); self.images=[]; self.links=[]
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == 'img': self.images.append(a.get('src', ''))
            if tag == 'a': self.links.append(a.get('href', ''))
    parser = Parser()
    parser.feed((HERE/'report_zh.html').read_text(encoding='utf-8'))
    checks['all_eight_report_images_embedded'] = len(parser.images) == 8 and all(p.startswith('data:image/png;base64,') for p in parser.images)
    checks['local_report_links_resolve'] = all((HERE/p).exists() for p in parser.links if not p.startswith(('http://', 'https://', '#')))
    for path in (HERE/'figures').glob('*.png'):
        with Image.open(path) as im: im.verify()
    checks['all_figure_pngs_valid'] = True
    checks['no_official_test_prediction_files'] = len(list((HERE/'predictions').glob('*.csv.gz'))) == 7
    checks = {k: bool(v) for k, v in checks.items()}
    output = {'checks': checks, 'count': len(checks), 'all_passed': all(checks.values()),
              'method': 'direct raw ID lookup for every saved row, independent confusion/cost, full June scoring from raw features using saved models; no refits'}
    (HERE/'qa_validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    assert output['all_passed']


if __name__ == '__main__': main()
