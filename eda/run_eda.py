"""Full-data exploratory analysis. No estimator fitting, predictions or submissions.

Run --mode analyze with bundled Python (pandas/numpy); --mode render with a
Python containing matplotlib/markdown. All outputs remain under this directory.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import platform
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / 'tables'
FIGURES = HERE / 'figures'
SEED = 7406
SOURCE_FILES = ['fraudTrain.csv', 'fraudTest.csv', 'ISYE 7406 Project Description.pdf']
Y = 'is_fraud'
CAT_COLUMNS = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city',
               'state', 'zip', 'job']
RAW_NUMERIC = ['ID', 'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
NUMERIC = ['amt', 'log_amt', 'age', 'city_pop', 'log_city_pop', 'lat', 'long',
           'merch_lat', 'merch_long', 'distance_km', 'hour', 'weekday', 'month_of_year']
SENSITIVE_LEVELS = {'cc_num', 'first', 'last', 'street', 'zip', 'customer_key'}


def log(message):
    print(message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_table(name, frame):
    frame.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')
    return frame


def wilson(positive, total):
    positive, total = np.asarray(positive, dtype=float), np.asarray(total, dtype=float)
    z = 1.959963984540054
    p = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
    den = 1 + z * z / np.maximum(total, 1)
    center = (p + z * z / (2 * np.maximum(total, 1))) / den
    half = z * np.sqrt(p * (1-p) / np.maximum(total, 1) + z*z / (4 * np.maximum(total, 1)**2)) / den
    return np.maximum(center-half, 0), np.minimum(center+half, 1)


def rates(frame, group, output='level'):
    result = frame.groupby(group, observed=True, dropna=False)[Y].agg(n='size', fraud='sum').reset_index()
    result = result.rename(columns={group: output})
    result['normal'] = result.n-result.fraud
    result['fraud_rate'] = result.fraud/result.n
    result['wilson_lower'], result['wilson_upper'] = wilson(result.fraud, result.n)
    result['fraud_share'] = result.fraud/frame[Y].sum()
    result['transaction_share'] = result.n/len(frame)
    result['lift'] = result.fraud_rate/frame[Y].mean()
    return result


def distribution_distance(a, b):
    """JS in natural-log units and total variation; counts aligned by caller."""
    p, q = np.asarray(a, float), np.asarray(b, float)
    p, q = p/p.sum(), q/q.sum()
    m = (p+q)/2
    left, right = p > 0, q > 0
    js = .5*np.sum(p[left]*np.log(p[left]/m[left])) + .5*np.sum(q[right]*np.log(q[right]/m[right]))
    return float(js), float(.5*np.abs(p-q).sum())


def ks_distance(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    values = np.unique(np.r_[a, b])
    return float(np.max(np.abs(np.searchsorted(a, values, side='right')/len(a)
                               -np.searchsorted(b, values, side='right')/len(b))))


def cramers_v(a, b):
    """Bias-corrected Cramer's V; descriptive association, not predictive value."""
    counts = pd.crosstab(a, b).to_numpy(dtype=float)
    n = counts.sum()
    r, k = counts.shape
    expected = counts.sum(axis=1)[:, None] @ counts.sum(axis=0)[None, :] / n
    chi2 = np.divide((counts-expected)**2, expected, out=np.zeros_like(counts), where=expected > 0).sum()
    phi = max(0., chi2/n - (k-1)*(r-1)/(n-1))
    rc, kc = r-(r-1)**2/(n-1), k-(k-1)**2/(n-1)
    den = min(rc-1, kc-1)
    return float(np.sqrt(phi/den)) if den > 0 else np.nan


def numeric_description(frame, columns, dataset, label=None):
    rows = []
    for col in columns:
        s = frame[col].replace([np.inf, -np.inf], np.nan)
        q = s.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
        row = {'dataset': dataset, 'label': label, 'feature': col, 'n': int(s.notna().sum()),
               'missing': int(s.isna().sum()), 'mean': s.mean(), 'std': s.std(), 'skew': s.skew()}
        row.update({f'q{int(v*100):02d}': q.loc[v] for v in q.index})
        rows.append(row)
    return rows


def load_data(filename):
    label = filename == 'fraudTrain.csv'
    types = {c: 'category' for c in CAT_COLUMNS if c != 'zip'}
    types.update({'cc_num': 'string', 'zip': 'string', 'trans_num': 'string',
                  'trans_date_trans_time': 'string', 'dob': 'string', 'ID': 'int32'})
    if label:
        types[Y] = 'int8'
    d = pd.read_csv(ROOT/filename, dtype=types)
    original_columns = d.columns.tolist()
    profile = []
    for col in original_columns:
        s = d[col]
        blank = int(s.astype('string').str.strip().eq('').fillna(False).sum()) if col not in RAW_NUMERIC+[Y] else 0
        profile.append({'dataset': 'train' if label else 'test', 'feature': col, 'dtype_loaded': str(s.dtype),
                        'rows': len(d), 'missing': int(s.isna().sum()), 'blank_strings': blank,
                        'unique': int(s.nunique(dropna=True))})
    date_format = '%Y-%m-%d %H:%M:%S' if label else '%m/%d/%Y %H:%M'
    dob_format = '%Y-%m-%d' if label else '%m/%d/%Y'
    text_time = pd.to_datetime(d.trans_date_trans_time, format=date_format, errors='coerce')
    d['timestamp'] = pd.to_datetime(d.unix_time, unit='s', errors='coerce')
    birth = pd.to_datetime(d.dob, format=dob_format, errors='coerce')
    d['dob_iso'] = birth.dt.strftime('%Y-%m-%d')
    d['age'] = (d.timestamp-birth).dt.total_seconds()/(365.2425*86400)
    d['hour'] = d.timestamp.dt.hour.astype('int8')
    d['weekday'] = d.timestamp.dt.dayofweek.astype('int8')
    d['month_of_year'] = d.timestamp.dt.month.astype('int8')
    d['month'] = d.timestamp.dt.strftime('%Y-%m').astype('category')
    d['date'] = d.timestamp.dt.floor('D')
    d['log_amt'] = np.log1p(d.amt)
    d['log_city_pop'] = np.log1p(d.city_pop)
    a1, a2 = np.radians(d.lat), np.radians(d.merch_lat)
    aa = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin(np.radians(d.merch_long-d.long)/2)**2
    d['distance_km'] = 6371.0088*2*np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
    identity = d[['first', 'last', 'street', 'zip', 'dob_iso']].astype('string')
    d['customer_key'] = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype='uint64')
    if identity.drop_duplicates().shape[0] != d.customer_key.nunique():
        raise ValueError('Customer-key hash collision detected')
    numeric_epoch = text_time.to_numpy(dtype='datetime64[s]').astype('int64')
    differences = np.diff(d.unix_time.to_numpy(dtype='int64'))
    text_differences = np.diff(numeric_epoch)
    check = {
        'rows': len(d), 'columns': len(original_columns), 'missing_cells': int(d[original_columns].isna().sum().sum()),
        'duplicate_full_rows': int(d[original_columns].duplicated().sum()),
        'duplicate_business_rows_excluding_ID': int(d[[c for c in original_columns if c != 'ID']].duplicated().sum()),
        'duplicate_trans_num': int(d.trans_num.duplicated().sum()), 'duplicate_ID': int(d.ID.duplicated().sum()),
        'time_min': str(d.timestamp.min()), 'time_max': str(d.timestamp.max()),
        'text_time_parse_failures': int(text_time.isna().sum()), 'dob_parse_failures': int(birth.isna().sum()),
        'unix_time_parse_failures': int(d.timestamp.isna().sum()),
        'text_unix_second_mismatch': int((numeric_epoch != d.unix_time).sum()),
        'text_unix_minute_mismatch': int((numeric_epoch//60 != d.unix_time//60).sum()),
        'out_of_order_unix_steps': int((differences < 0).sum()),
        'out_of_order_text_steps': int((text_differences < 0).sum()),
        'largest_backward_seconds': int(differences.min(initial=0)),
        'duplicate_timestamp_rows': int(d.unix_time.duplicated().sum()),
        'negative_or_zero_amount_rows': int((d.amt <= 0).sum()),
        'negative_population_rows': int((d.city_pop < 0).sum()),
        'nonfinite_numeric_cells': int((~np.isfinite(d[RAW_NUMERIC].to_numpy(dtype=float))).sum()),
        'age_outside_0_110_rows': int((~d.age.between(0, 110)).sum()),
        'invalid_latitude_rows': int((~d.lat.between(-90, 90) | ~d.merch_lat.between(-90, 90)).sum()),
        'invalid_longitude_rows': int((~d.long.between(-180, 180) | ~d.merch_long.between(-180, 180)).sum()),
        'cc_num_scientific_rows': int(d.cc_num.str.contains('[eE]', regex=True).sum()),
        'cc_num_non_digit_rows': int((~d.cc_num.str.fullmatch(r'\d+')).sum()),
        'zip_less_than_5_char_rows': int((d.zip.str.len() < 5).sum()),
        'trans_num_not_32_hex_rows': int((~d.trans_num.str.fullmatch(r'[0-9a-fA-F]{32}')).sum()),
        'merchant_fraud_prefix_rows': int(d.merchant.str.startswith('fraud_').sum()),
        'unique_customers_static_key': int(d.customer_key.nunique()),
        'memory_mib_loaded_and_derived': float(d.memory_usage(deep=True).sum()/1024**2),
    }
    if label:
        check.update({'fraud': int(d[Y].sum()), 'normal': int((d[Y] == 0).sum()),
                      'fraud_rate': float(d[Y].mean()), 'label_values': sorted(d[Y].unique().tolist())})
    return d, profile, check


def analyze():
    started = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sources = {name: {'sha256': sha256(ROOT/name), 'bytes': (ROOT/name).stat().st_size} for name in SOURCE_FILES}
    log('Loading training data and auditing every original column ...')
    train, profile_train, quality_train = load_data('fraudTrain.csv')
    log('Loading unlabeled test data and auditing formats ...')
    test, profile_test, quality_test = load_data('fraudTest.csv')
    save_table('01_feature_inventory', pd.DataFrame(profile_train+profile_test))
    quality = {'train': quality_train, 'test': quality_test}
    save_table('02_data_quality', pd.DataFrame([{'check': k, 'train': quality_train.get(k), 'test': quality_test.get(k)}
                                              for k in dict.fromkeys([*quality_train, *quality_test])]))
    train_keys, test_keys = set(train.customer_key.unique()), set(test.customer_key.unique())
    train_cards = train[['cc_num', 'customer_key']].drop_duplicates()
    rounded = train_cards.cc_num.map(lambda x: format(Decimal(x), '.5E'))
    rounding_sizes = train_cards.assign(rounded=rounded).groupby('rounded').cc_num.nunique()
    entity = {
        'train_cards': int(train.cc_num.nunique()), 'test_raw_card_tokens': int(test.cc_num.nunique()),
        'train_customers': len(train_keys), 'test_customers': len(test_keys),
        'shared_customers': len(train_keys & test_keys), 'new_test_customers': len(test_keys-train_keys),
        'new_test_customer_rows': int((~test.customer_key.isin(train_keys)).sum()),
        'new_test_merchant_rows': int((~test.merchant.isin(train.merchant.unique())).sum()),
        'raw_card_token_shared': len(set(train.cc_num.unique()) & set(test.cc_num.unique())),
        'rounded_6_significant_digit_collision_buckets': int((rounding_sizes > 1).sum()),
        'static_key_multiple_training_cards': int((train_cards.groupby('customer_key').cc_num.nunique() > 1).sum()),
        'trans_num_overlap': len(set(train.trans_num.unique()) & set(test.trans_num.unique())),
        'ID_overlap': len(set(train.ID.unique()) & set(test.ID.unique())),
    }
    save_table('03_entity_overlap', pd.DataFrame([{'metric': k, 'value': v} for k, v in entity.items()]))

    log('Computing full-data numeric distributions, class differences and feature dependence ...')
    numeric_rows = []
    for name, frame in [('train', train), ('test', test)]:
        numeric_rows.extend(numeric_description(frame, list(dict.fromkeys(RAW_NUMERIC+NUMERIC)), name))
    for y in [0, 1]:
        numeric_rows.extend(numeric_description(train.loc[train[Y] == y], NUMERIC, 'train', y))
    numeric = save_table('04_numeric_summary', pd.DataFrame(numeric_rows))
    pearson = train[NUMERIC+['ID', 'unix_time']].corr(method='pearson')
    spearman = train[NUMERIC+['ID', 'unix_time']].corr(method='spearman')
    save_table('05_pearson', pearson.rename_axis('feature').reset_index())
    save_table('06_spearman', spearman.rename_axis('feature').reset_index())
    association_cols = ['category', 'merchant', 'gender', 'state', 'city', 'zip', 'job', 'customer_key']
    assoc = pd.DataFrame(np.eye(len(association_cols)), index=association_cols, columns=association_cols)
    for i, a in enumerate(association_cols):
        for b in association_cols[:i]:
            assoc.loc[a, b] = assoc.loc[b, a] = cramers_v(train[a], train[b])
    save_table('07_categorical_cramers_v', assoc.rename_axis('feature').reset_index())
    dependencies = []
    dependency_pairs = ([('cc_num', v) for v in ['first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob']]
                        + [('customer_key', 'cc_num'), ('merchant', 'category'), ('zip', 'city'), ('zip', 'state'), ('city', 'state'), ('street', 'cc_num')])
    for a, b in dependency_pairs:
        counts = train.groupby(a, observed=True)[b].nunique()
        bad = set(counts.index[counts > 1])
        dependencies.append({'determinant': a, 'dependent': b, 'n_levels': len(counts),
                             'ambiguous_levels': len(bad), 'affected_rows': int(train[a].isin(bad).sum()),
                             'max_distinct_dependents': int(counts.max())})
    save_table('08_functional_dependencies', pd.DataFrame(dependencies))
    cat_association = []
    for col in ['category', 'merchant', 'gender', 'state', 'city', 'zip', 'job', 'cc_num', 'first', 'last', 'street']:
        sizes = train.groupby(col, observed=True).size()
        cat_association.append({'feature': col, 'levels': int(len(sizes)), 'target_cramers_v_bias_corrected': cramers_v(train[col], train[Y]),
                                'levels_below_100_rows': int((sizes < 100).sum()), 'minimum_support': int(sizes.min()),
                                'median_support': float(sizes.median())})
        if col not in SENSITIVE_LEVELS:
            save_table(f'rates_{col}', rates(train, col).sort_values('n', ascending=False))
    save_table('09_categorical_target_association', pd.DataFrame(cat_association))
    effects = []
    for col in NUMERIC:
        normal, fraud = train.loc[train[Y] == 0, col], train.loc[train[Y] == 1, col]
        effects.append({'feature': col, 'ks_class_difference': ks_distance(normal, fraud),
                        'median_normal': normal.median(), 'median_fraud': fraud.median(),
                        'spearman_with_label': train[[col, Y]].corr(method='spearman').iloc[0, 1]})
    save_table('10_numeric_target_association', pd.DataFrame(effects))

    histograms = []
    hist_features = ['amt', 'log_amt', 'age', 'log_city_pop', 'distance_km', 'lat', 'long']
    for col in hist_features:
        pooled_min = min(train[col].min(), test[col].min())
        pooled_max = max(train[col].max(), test[col].max())
        edges = np.linspace(pooled_min, pooled_max, 71)
        if col == 'amt':
            edges = np.linspace(0, 1500, 76)
        for label, values in [('normal', train.loc[train[Y] == 0, col]), ('fraud', train.loc[train[Y] == 1, col]),
                              ('train', train[col]), ('test', test[col])]:
            counts, _ = np.histogram(values, edges)
            for i, count in enumerate(counts):
                histograms.append({'feature': col, 'group': label, 'left': edges[i], 'right': edges[i+1],
                                   'count': int(count), 'share_of_all_group_rows': count/len(values),
                                   'density': count/len(values)/(edges[i+1]-edges[i]),
                                   'outside_plot_range': int(((values < edges[0]) | (values > edges[-1])).sum())})
    save_table('11_histograms', pd.DataFrame(histograms))

    log('Analyzing time, category/amount interactions, demographics and geography ...')
    monthly = []
    daily = []
    for label, frame in [('train', train), ('test', test)]:
        if label == 'train':
            m = rates(frame, 'month', 'month')
            dd = rates(frame, 'date', 'date')
        else:
            m = frame.groupby('month', observed=True).size().rename('n').reset_index()
            dd = frame.groupby('date', observed=True).size().rename('n').reset_index()
        m['dataset'], dd['dataset'] = label, label
        monthly.append(m)
        daily.append(dd)
    monthly = save_table('12_monthly', pd.concat(monthly, ignore_index=True))
    save_table('13_daily', pd.concat(daily, ignore_index=True))
    for col in ['hour', 'weekday', 'month_of_year']:
        save_table(f'rates_{col}', rates(train, col))
    amount_edges = [-np.inf, 10, 50, 100, 200, 500, 1000, 1500, 2000, np.inf]
    train['amount_band'] = pd.cut(train.amt, amount_edges)
    save_table('14_amount_bands', rates(train, 'amount_band'))
    amount_category = train.groupby(['category', 'amount_band'], observed=True)[Y].agg(n='size', fraud='sum').reset_index()
    amount_category['fraud_rate'] = amount_category.fraud/amount_category.n
    save_table('15_category_amount', amount_category)
    category_hour = train.groupby(['category', 'hour'], observed=True)[Y].agg(n='size', fraud='sum').reset_index()
    category_hour['fraud_rate'] = category_hour.fraud/category_hour.n
    save_table('16_category_hour', category_hour)
    category_month = train.groupby(['category', 'month'], observed=True)[Y].agg(n='size', fraud='sum').reset_index()
    category_month['fraud_rate'] = category_month.fraud/category_month.n
    save_table('17_category_month', category_month)
    monthly_composition = category_month.copy()
    monthly_composition['normal'] = monthly_composition.n-monthly_composition.fraud
    monthly_composition['dataset'] = 'train'
    test_composition = test.groupby(['category', 'month'], observed=True).size().rename('n').reset_index()
    test_composition['dataset'] = 'test'
    save_table('35_category_month_composition', pd.concat([monthly_composition, test_composition], ignore_index=True))
    active_categories, active_merchants = [], []
    for name, data in [('normal', train.loc[train[Y] == 0]), ('fraud', train.loc[train[Y] == 1]), ('test_unlabeled', test)]:
        for field, destination in [('category', active_categories), ('merchant', active_merchants)]:
            active = data.groupby(field, observed=True).agg(n=('unix_time', 'size'), first_time=('timestamp', 'min'),
                        last_time=('timestamp', 'max'), active_months=('month', 'nunique'), active_days=('date', 'nunique')).reset_index()
            active['span_days'] = (active.last_time-active.first_time).dt.total_seconds()/86400
            active['group'] = name
            destination.append(active)
    active_categories = save_table('36_category_active_periods', pd.concat(active_categories, ignore_index=True))
    active_merchants = save_table('37_merchant_active_periods', pd.concat(active_merchants, ignore_index=True))
    zero_normal = monthly_composition.normal.eq(0)
    temporal_structure = {
        'total_category_month_cells_train': len(monthly_composition),
        'normal_nonempty_category_month_cells': int(monthly_composition.normal.gt(0).sum()),
        'fraud_nonempty_category_month_cells': int(monthly_composition.fraud.gt(0).sum()),
        'zero_normal_positive_fraud_cells': int((zero_normal & monthly_composition.fraud.gt(0)).sum()),
        'fraud_rows_in_zero_normal_cells': int(monthly_composition.loc[zero_normal, 'fraud'].sum()),
        'fraud_share_in_zero_normal_cells': float(monthly_composition.loc[zero_normal, 'fraud'].sum()/train[Y].sum()),
        'normal_category_month_cramers_v': cramers_v(train.loc[train[Y] == 0, 'category'], train.loc[train[Y] == 0, 'month']),
        'fraud_category_month_cramers_v': cramers_v(train.loc[train[Y] == 1, 'category'], train.loc[train[Y] == 1, 'month']),
        'test_category_month_cramers_v': cramers_v(test.category, test.month),
    }
    save_table('38_time_structure_summary', pd.DataFrame([{'metric': k, 'value': v} for k, v in temporal_structure.items()]))
    save_table('39_category_amount_by_label', train.groupby(['category', Y], observed=True).amt.agg(
        n='size', mean='mean', median='median', q95=lambda a: a.quantile(.95), maximum='max').reset_index())
    age_edges = [0, 18, 25, 35, 45, 55, 65, 75, 85, 111]
    train['age_band'] = pd.cut(train.age, age_edges, right=False)
    save_table('18_age_bands', rates(train, 'age_band'))
    pop_edges = [-np.inf, 1000, 10000, 100000, 1000000, np.inf]
    train['population_band'] = pd.cut(train.city_pop, pop_edges)
    save_table('19_population_bands', rates(train, 'population_band'))
    train['distance_band'] = pd.cut(train.distance_km, [-np.inf, 25, 50, 75, 100, 125, 150, np.inf])
    save_table('20_distance_bands', rates(train, 'distance_band'))

    customer = train.groupby('customer_key', observed=True)[Y].agg(n='size', fraud='sum').reset_index()
    customer['fraud_rate'] = customer.fraud/customer.n
    customer['profile'] = np.select([customer.fraud == 0, customer.fraud == customer.n], ['normal_only', 'fraud_only'], default='mixed')
    customer['customer_anonymous'] = [f'C{i:04d}' for i in range(len(customer))]
    customer_export = customer.drop(columns='customer_key')
    save_table('21_customer_profiles', customer_export)
    profile_counts = customer.groupby('profile').agg(customers=('n', 'size'), transactions=('n', 'sum'), fraud=('fraud', 'sum')).reset_index()
    save_table('22_customer_profile_summary', profile_counts)
    entity.update({'fraud_only_customers': int((customer.profile == 'fraud_only').sum()),
                   'fraud_only_customer_transactions': int(customer.loc[customer.profile == 'fraud_only', 'n'].sum()),
                   'fraud_only_customer_median_support': float(customer.loc[customer.profile == 'fraud_only', 'n'].median()),
                   'mixed_customers': int((customer.profile == 'mixed').sum()),
                   'normal_only_customers': int((customer.profile == 'normal_only').sum())})
    positive_customer_counts = customer.sort_values('fraud', ascending=False).fraud.to_numpy()
    concentration = [{'top_customers': k, 'fraud_share': positive_customer_counts[:k].sum()/train[Y].sum()}
                     for k in [1, 5, 10, 25, 50, 100]]
    save_table('23_fraud_concentration', pd.DataFrame(concentration))
    unseen_windows = []
    for start, end in [('2025-01-01', '2025-04-01'), ('2025-04-01', '2025-06-01'), ('2025-06-01', '2025-07-01')]:
        past = train.loc[train.timestamp < start]
        window = train.loc[train.timestamp.ge(start) & train.timestamp.lt(end)]
        unseen = ~window.customer_key.isin(past.customer_key.unique())
        unseen_windows.append({'start': start, 'end_exclusive': end, 'n': len(window), 'fraud': int(window[Y].sum()),
                               'new_customers': int(window.loc[unseen, 'customer_key'].nunique()),
                               'new_customer_rows': int(unseen.sum()),
                               'fraud_in_new_customer_rows': int(window.loc[unseen, Y].sum()),
                               'new_customer_row_share': float(unseen.mean())})
    save_table('24_new_customers_over_time', pd.DataFrame(unseen_windows))

    log('Computing strictly earlier, label-free customer-history descriptors ...')
    sequence = pd.concat([train[['customer_key', 'unix_time', 'amt', 'merchant', Y]].assign(dataset='train'),
                          test[['customer_key', 'unix_time', 'amt', 'merchant']].assign(dataset='test', is_fraud=-1)],
                         ignore_index=True)
    sequence['source_row'] = np.arange(len(sequence))
    aggregated = sequence.groupby(['customer_key', 'unix_time'], sort=True, observed=True).agg(events=('amt', 'size'),
                                                                                           amount_sum=('amt', 'sum')).reset_index()
    group = aggregated.groupby('customer_key', sort=False)
    aggregated['past_count'] = group.events.cumsum()-aggregated.events
    aggregated['past_sum'] = group.amount_sum.cumsum()-aggregated.amount_sum
    aggregated['past_mean_amt'] = aggregated.past_sum/aggregated.past_count.replace(0, np.nan)
    aggregated['gap_seconds'] = group.unix_time.diff()
    count24 = np.zeros(len(aggregated), dtype='int64')
    for positions in group.indices.values():
        times = aggregated.unix_time.to_numpy()[positions]
        counts = aggregated.events.to_numpy()[positions]
        prefix = np.r_[0, np.cumsum(counts)]
        lower = np.searchsorted(times, times-86400, side='left')
        count24[positions] = prefix[:-1]-prefix[lower]
    aggregated['count_24h'] = count24
    sequence = sequence.merge(aggregated[['customer_key', 'unix_time', 'past_count', 'past_mean_amt', 'gap_seconds', 'count_24h']],
                              on=['customer_key', 'unix_time'], how='left', validate='many_to_one').sort_values('source_row')
    first_merchant_time = sequence.groupby(['customer_key', 'merchant'], observed=True).unix_time.transform('min')
    sequence['new_merchant'] = (sequence.unix_time == first_merchant_time).astype('int8')
    sequence['amount_to_past_mean'] = sequence.amt/sequence.past_mean_amt
    sequence_train = sequence.loc[sequence.dataset == 'train'].copy()
    sequence_summary = []
    for y in [0, 1]:
        sequence_summary.extend(numeric_description(sequence_train.loc[sequence_train[Y] == y],
                                 ['past_count', 'past_mean_amt', 'gap_seconds', 'count_24h', 'amount_to_past_mean', 'new_merchant'], 'train', y))
    save_table('25_sequence_summary', pd.DataFrame(sequence_summary))
    sequence_train['count24_band'] = pd.cut(sequence_train.count_24h, [-.5, .5, 1.5, 2.5, 5.5, 10.5, 20.5, 50.5, np.inf])
    sequence_train['relative_amount_band'] = pd.cut(sequence_train.amount_to_past_mean, [-np.inf, .25, .5, 1, 2, 5, 10, np.inf])
    sequence_train['gap_band'] = pd.cut(sequence_train.gap_seconds, [-np.inf, 30, 60, 300, 1800, 3600, 86400, np.inf])
    save_table('26_history_count24_bands', rates(sequence_train, 'count24_band'))
    save_table('27_history_relative_amount_bands', rates(sequence_train, 'relative_amount_band'))
    save_table('28_history_gap_bands', rates(sequence_train, 'gap_band'))
    save_table('29_new_merchant', rates(sequence_train, 'new_merchant'))
    first_rows = sequence_train.past_count.eq(0)
    entity['first_customer_transaction_rows'] = int(first_rows.sum())
    entity['first_customer_transaction_fraud'] = int(sequence_train.loc[first_rows, Y].sum())
    entity['first_customer_transaction_fraud_rate'] = float(sequence_train.loc[first_rows, Y].mean())
    sequence_checks = {
        'row_count_preserved': len(sequence) == len(train)+len(test),
        'strict_past_count_nonnegative': bool(sequence.past_count.ge(0).all()),
        'count24_within_past_count': bool(sequence.count_24h.le(sequence.past_count).all()),
        'distinct_timestamp_previous_gap_positive': bool(sequence.gap_seconds.dropna().gt(0).all()),
        'all_first_events_have_no_past_mean': bool(sequence.loc[sequence.past_count == 0, 'past_mean_amt'].isna().all()),
    }
    # Independent direct filtering checks the aggregated implementation, including
    # same-second records, first events, positive records and train/test boundary.
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(len(sequence), size=12, replace=False).tolist())
    chosen.update([0, len(train), len(sequence)-1])
    same_time_positions = np.flatnonzero(sequence.duplicated(['customer_key', 'unix_time'], keep=False).to_numpy())
    chosen.update(same_time_positions[:2].tolist())
    chosen.update(np.flatnonzero(sequence[Y].eq(1).to_numpy())[:2].tolist())
    spot_checks = []
    for position in sorted(chosen):
        row = sequence.iloc[position]
        past = sequence.loc[(sequence.customer_key == row.customer_key) & (sequence.unix_time < row.unix_time)]
        expected_mean = past.amt.mean() if len(past) else np.nan
        expected_gap = row.unix_time-past.unix_time.max() if len(past) else np.nan
        direct_checks = {
            'past_count_matches': int(row.past_count) == len(past),
            'past_mean_matches': bool(np.isclose(row.past_mean_amt, expected_mean, rtol=1e-10, atol=1e-9, equal_nan=True)),
            'gap_matches': bool(np.isclose(row.gap_seconds, expected_gap, equal_nan=True)),
            'count24_matches': int(row.count_24h) == int(past.unix_time.ge(row.unix_time-86400).sum()),
            'new_merchant_matches': bool(row.new_merchant) == bool(~past.merchant.eq(row.merchant).any()),
        }
        spot_checks.append({'source_row': int(row.source_row), 'dataset': row.dataset, **direct_checks})
    spot = save_table('40_history_independent_spot_checks', pd.DataFrame(spot_checks))
    sequence_checks['independent_history_spot_checks'] = bool(spot[[c for c in spot if c.endswith('_matches')]].all().all())
    positive_sequence = train.loc[train[Y] == 1, ['customer_key', 'unix_time']].sort_values(['customer_key', 'unix_time'])
    fraud_gap = positive_sequence.groupby('customer_key').unix_time.diff().dropna()
    entity['repeat_fraud_events'] = len(fraud_gap)
    entity['repeat_fraud_within_1h_share'] = float(fraud_gap.le(3600).mean())
    entity['repeat_fraud_within_24h_share'] = float(fraud_gap.le(86400).mean())
    entity['median_gap_between_fraud_events_seconds'] = float(fraud_gap.median())
    del sequence, aggregated, sequence_train, positive_sequence

    log('Comparing unlabeled test distributions and within-training stability ...')
    drift = []
    for col in NUMERIC:
        a, b = train[col].dropna().to_numpy(), test[col].dropna().to_numpy()
        quantile_edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        inner = quantile_edges[1:-1]
        bins = np.r_[-np.inf, inner, np.inf]
        ca, _ = np.histogram(a, bins)
        cb, _ = np.histogram(b, bins)
        # Jeffreys 0.5 pseudo-count, documented; avoids log(0) at empty bins.
        pa, pb = (ca+.5)/(ca.sum()+.5*len(ca)), (cb+.5)/(cb.sum()+.5*len(cb))
        js, tvd = distribution_distance(ca, cb)
        drift.append({'feature': col, 'ks_distance': ks_distance(a, b),
                      'psi_train_quantile_bins': float(np.sum((pb-pa)*np.log(pb/pa))),
                      'js_divergence_nats_binned': js, 'tvd_binned': tvd,
                      'train_median': np.median(a), 'test_median': np.median(b),
                      'test_outside_train_range_share': float(((b < a.min()) | (b > a.max())).mean()),
                      'bins': len(ca)})
    drift = save_table('30_numeric_train_test_shift', pd.DataFrame(drift))
    cat_drift = []
    for col in CAT_COLUMNS+['cc_num', 'customer_key', 'hour', 'weekday', 'month_of_year']:
        a, b = train[col].value_counts(), test[col].value_counts()
        levels = a.index.union(b.index)
        ca, cb = a.reindex(levels, fill_value=0), b.reindex(levels, fill_value=0)
        js, tvd = distribution_distance(ca, cb)
        cat_drift.append({'feature': col, 'train_levels': len(a), 'test_levels': len(b),
                          'js_divergence_nats': js, 'tvd': tvd,
                          'test_unseen_level_row_share': float((~test[col].isin(a.index)).mean())})
    save_table('31_categorical_train_test_shift', pd.DataFrame(cat_drift))
    for col in ['category', 'hour', 'weekday', 'month_of_year', 'gender', 'state']:
        a, b = train[col].value_counts(normalize=True), test[col].value_counts(normalize=True)
        levels = a.index.union(b.index)
        save_table(f'compare_{col}', pd.DataFrame({'level': levels, 'train_share': a.reindex(levels, fill_value=0).to_numpy(),
                                                  'test_share': b.reindex(levels, fill_value=0).to_numpy()}))

    # Fixed category/amount cells provide drift context; not used as model input.
    train['quarter'] = train.timestamp.dt.to_period('Q').astype(str)
    stability = train.groupby(['quarter', 'category'], observed=True)[Y].agg(n='size', fraud='sum').reset_index()
    stability['fraud_rate'] = stability.fraud/stability.n
    save_table('32_quarter_category_stability', stability)

    roles = {
        'ID': ('行标识', '排除预测输入', '重置编号且高度代理时间，仅用于输出对齐'),
        'trans_date_trans_time': ('交易时间', '统一解析后派生', '测试文本缺秒；与 unix_time 重复'),
        'cc_num': ('客户标识', '先修复身份一致性', '测试精度损失，不能按原始值连接或当连续数值'),
        'merchant': ('商户类别变量', '基础候选', '与 category 重叠但不完全一一对应'),
        'category': ('交易类别变量', '基础候选', '与金额和时段联合检查'),
        'amt': ('金额，美元', '基础候选', '非单调风险；长尾不能简单视为欺诈'),
        'first': ('客户姓名', '仅身份匹配', '不把样本内客户差异当普适规律'),
        'last': ('客户姓名', '仅身份匹配', '与客户键高度冗余'),
        'gender': ('客户属性', '候选并做增量验证', '交易加权差异不代表客户层面的因果影响'),
        'street': ('客户地址', '仅身份匹配', '几乎是客户标识；不直接生成全期目标编码'),
        'city': ('地理类别', '与位置字段成组验证', '城市名不一定唯一确定州'),
        'state': ('地理类别', '基础候选', '与城市/邮编/位置存在依赖'),
        'zip': ('邮编标识', '按字符串处理并成组验证', '保留前导零；与客户位置冗余'),
        'lat': ('客户纬度', '地理特征组候选', '与商户纬度高度相关'),
        'long': ('客户经度', '地理特征组候选', '与商户经度高度相关'),
        'city_pop': ('城市人口', '基础候选', '偏态；线性模型可用 log1p'),
        'job': ('职业类别', '候选并检验身份记忆', '支持量不均且与客户键关联'),
        'dob': ('出生日期', '转换为交易时年龄', '格式不同；避免同时保留冗余日期表达'),
        'trans_num': ('唯一交易标识', '排除预测输入', '仅用于去重与追踪'),
        'unix_time': ('秒级交易时间', '排序和派生时间特征', '不依赖文件原始行序；不推断客户当地时区'),
        'merch_lat': ('商户纬度', '地理特征组候选', '商户名称不能自动解释成单一固定门店'),
        'merch_long': ('商户经度', '地理特征组候选', '需检验绝对位置与相对距离的增量'),
        'is_fraud': ('训练标签', '仅训练期描述与未来评估', '测试未知，禁止补零或推断测试效用'),
    }
    inventory = pd.DataFrame(profile_train).drop(columns='dataset')
    inventory['meaning'] = inventory.feature.map(lambda x: roles[x][0])
    inventory['candidate_action'] = inventory.feature.map(lambda x: roles[x][1])
    inventory['caution'] = inventory.feature.map(lambda x: roles[x][2])
    save_table('33_feature_dictionary_and_actions', inventory)
    baselines = []
    for month, grp in train.groupby('month', observed=True):
        pos, n = int(grp[Y].sum()), len(grp)
        baselines.append({'month': month, 'n': n, 'fraud': pos, 'always_normal_accuracy': 1-pos/n,
                          'always_normal_utility': -100*pos, 'always_fraud_utility': 50*pos-5*(n-pos),
                          'oracle_utility_upper_bound': 50*pos})
    save_table('34_utility_reference_by_month', pd.DataFrame(baselines))

    checks = {'expected_train_rows': len(train) == 1402996, 'expected_test_rows': len(test) == 469843,
              'train_label_binary': set(train[Y].unique()) == {0, 1}, 'test_has_no_label': Y not in test.columns,
              'monthly_train_rows_reconcile': int(monthly.loc[monthly.dataset == 'train', 'n'].sum()) == len(train),
              'monthly_test_rows_reconcile': int(monthly.loc[monthly.dataset == 'test', 'n'].sum()) == len(test),
              'category_fraud_reconciles': int(pd.read_csv(TABLES/'rates_category.csv').fraud.sum()) == int(train[Y].sum()),
              'customer_profile_rows_reconcile': int(customer.n.sum()) == len(train),
              'customer_profile_fraud_reconcile': int(customer.fraud.sum()) == int(train[Y].sum()),
              'train_text_and_unix_match': quality_train['text_unix_second_mismatch'] == 0,
              'test_minute_and_unix_match': quality_test['text_unix_minute_mismatch'] == 0,
              'train_test_transactions_disjoint': entity['trans_num_overlap'] == 0,
              'joint_category_month_counts_reconcile': int(monthly_composition.n.sum()) == len(train),
              'joint_category_month_fraud_reconcile': int(monthly_composition.fraud.sum()) == int(train[Y].sum()),
              'finite_raw_numeric_fields': quality_train['nonfinite_numeric_cells'] == quality_test['nonfinite_numeric_cells'] == 0,
              **sequence_checks}
    checks['sources_unchanged'] = all(sha256(ROOT/name) == sources[name]['sha256'] for name in sources)
    if not all(checks.values()):
        raise AssertionError(f'Analysis reconciliation failed: {checks}')
    results = {'quality': quality, 'entities': entity, 'temporal_structure': temporal_structure,
               'fraud_rate_min_month': float(monthly.loc[monthly.dataset == 'train', 'fraud_rate'].min()),
               'fraud_rate_max_month': float(monthly.loc[monthly.dataset == 'train', 'fraud_rate'].max()),
               'theoretical_calibrated_probability_threshold': 5/155,
               'always_normal_utility_train': -100*int(train[Y].sum()),
               'always_normal_accuracy_train': 1-float(train[Y].mean()),
               'analysis_seconds': time.time()-started,
               'analysis_runtime': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                                    'executable': sys.executable},
               'random_seed': SEED, 'checks': checks}
    write_json(HERE/'results.json', results)
    write_json(HERE/'input_manifest.json', {'sources': sources, 'analysis_script_sha256': sha256(Path(__file__)),
                                          'checks': checks, 'seed': SEED})
    log(f'Analysis complete: {len(list(TABLES.glob("*.csv")))} aggregate tables; {results["analysis_seconds"]:.1f}s')
    log(json.dumps(clean_json({'quality': quality, 'entities': entity}), ensure_ascii=False))


def render():
    # Imported only in render mode because the bundled analysis runtime lacks matplotlib.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import markdown
    from report_content import write_reports

    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [fm.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': 165,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': .15})
    blue, red, muted = '#316a9a', '#c55345', '#697886'
    FIGURES.mkdir(exist_ok=True)
    r = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    read = lambda name: pd.read_csv(TABLES/f'{name}.csv')
    figures = []

    def finish(fig, name, title, caption):
        fig.suptitle(title, fontsize=15, x=.03, ha='left')
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(FIGURES/f'{name}.png', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append({'file': f'figures/{name}.png', 'title': title, 'caption': caption})

    def rate_bars(ax, data, title, horizontal=False):
        lo, hi, p = data.wilson_lower.to_numpy()*100, data.wilson_upper.to_numpy()*100, data.fraud_rate.to_numpy()*100
        x = np.arange(len(data))
        labels = data.level.astype(str)
        if horizontal:
            ax.barh(x, p, color=blue)
            ax.errorbar(p, x, xerr=[np.maximum(p-lo, 0), np.maximum(hi-p, 0)], fmt='none', ecolor=muted, capsize=2)
            ax.set_yticks(x, labels)
            ax.set_xlabel('欺诈率 (%)')
        else:
            ax.bar(x, p, color=blue)
            ax.errorbar(x, p, yerr=[np.maximum(p-lo, 0), np.maximum(hi-p, 0)], fmt='none', ecolor=muted, capsize=2)
            ax.set_xticks(x, labels, rotation=45, ha='right')
            ax.set_ylabel('欺诈率 (%)')
        line = r['quality']['train']['fraud_rate']*100
        (ax.axvline if horizontal else ax.axhline)(line, color=red, ls='--', lw=1, label='整体欺诈率')
        ax.set_title(title)

    monthly = read('12_monthly')
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    xx = np.arange(len(monthly))
    ax[0].bar(xx, monthly.n/1000, color=[blue if a == 'train' else muted for a in monthly.dataset])
    ax[0].set_ylabel('交易数 (千笔)')
    observed = monthly.fraud_rate.notna()
    ax[1].plot(xx[observed], monthly.loc[observed, 'fraud_rate']*100, 'o-', color=red)
    ax[1].set_ylabel('训练集欺诈率 (%)')
    for a in ax:
        a.axvline(17.5, color='#222', ls='--', lw=1)
        a.axvspan(17.5, 23.5, color='#e8edf2', alpha=.6)
    ax[1].set_xticks(xx, monthly.month, rotation=60, ha='right')
    finish(fig, '01_monthly', '交易量与欺诈率随时间变化', '灰色区域为无标签测试期；测试期欺诈率未知，没有补零。')

    hist = read('11_histograms')
    def histogram(ax, feature, comparison=('normal', 'fraud')):
        for group, color in zip(comparison, [blue, red]):
            h = hist.loc[(hist.feature == feature) & (hist.group == group)]
            ax.stairs(h.density, np.r_[h.left.to_numpy(), h.right.iloc[-1]], label=group, color=color, linewidth=1.6)
        ax.set_xlabel(feature)
        ax.set_ylabel('各组内归一化密度')
        ax.legend(frameon=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    histogram(ax[0], 'amt'); histogram(ax[1], 'log_amt')
    finish(fig, '02_amount', '金额分布：欺诈并非简单等同于极端大额', '左图显示 0–1500 美元，范围外交易仍计入分母；右图 log1p 金额覆盖全范围。')

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [2, 1]})
    rate_bars(ax[0], read('rates_hour'), '按数据记录时钟的小时')
    days = read('rates_weekday'); days['level'] = ['一', '二', '三', '四', '五', '六', '日']
    rate_bars(ax[1], days, '星期')
    finish(fig, '03_time_rates', '时段与星期的欺诈率', '误差线为逐笔 Bernoulli 假设下的 Wilson 区间，仅作描述；未校正客户/欺诈事件聚集。')

    category = read('rates_category').sort_values('fraud_rate')
    fig, ax = plt.subplots(figsize=(10, 6.5))
    rate_bars(ax, category, '类别差异', horizontal=True)
    finish(fig, '04_category', '商户类别与欺诈率', '类别支持量和正例数见 rates_category.csv；描述性高风险不等于稳定的预测规则。')

    def heatmap_rate(table, row, col, name, title, support=200, figsize=(13, 6)):
        data = read(table)
        values = data.pivot(index=row, columns=col, values='fraud_rate')*100
        counts = data.pivot(index=row, columns=col, values='n')
        if col == 'amount_band':
            desired = ['(-inf, 10.0]', '(10.0, 50.0]', '(50.0, 100.0]', '(100.0, 200.0]', '(200.0, 500.0]',
                       '(500.0, 1000.0]', '(1000.0, 1500.0]', '(1500.0, 2000.0]', '(2000.0, inf]']
            order = [v for v in desired if v in values]
            values, counts = values[order], counts[order]
        masked = np.ma.masked_where(counts.to_numpy() < support, values.to_numpy())
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.colormaps['YlOrRd'].copy(); cmap.set_bad('#e1e6eb')
        im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0)
        ax.set_yticks(np.arange(len(values)), values.index)
        ax.set_xticks(np.arange(len(values.columns)), values.columns, rotation=55, ha='right')
        ax.grid(False)
        fig.colorbar(im, ax=ax, label='欺诈率 (%)', fraction=.03, pad=.02)
        finish(fig, name, title, f'灰色格表示支持量小于 {support} 或没有样本，不表示零欺诈率；完整计数见对应 CSV。')
    heatmap_rate('15_category_amount', 'category', 'amount_band', '05_category_amount', '商户类别与金额区间的联合分布')
    heatmap_rate('16_category_hour', 'category', 'hour', '06_category_hour', '商户类别与时段的联合分布')
    heatmap_rate('17_category_month', 'category', 'month', '07_category_month', '类别欺诈率在不同月份的稳定性')

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    for a, feature in zip(ax, ['age', 'log_city_pop', 'distance_km']):
        histogram(a, feature)
    finish(fig, '08_numeric_other', '年龄、城市人口与相对位置', '距离由 Haversine 公式计算。边际分布相近不排除交互作用，也不证明该字段无用。')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    rate_bars(ax[0], read('rates_gender'), '性别（交易加权）')
    rate_bars(ax[1], read('18_age_bands'), '年龄区间（交易加权）')
    finish(fig, '09_demographics', '客户属性与欺诈率', '客户交易频率不同，交易加权的差异不能解释成客户层面或因果效应。')

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    for a, col in zip(ax, ['state', 'job']):
        dat = read(f'rates_{col}').nlargest(15, 'n').sort_values('fraud_rate')
        rate_bars(a, dat, f'{col}：交易数最多的 15 组', horizontal=True)
    finish(fig, '10_state_job', '地理与职业的分组差异', '按支持量选组，避免仅展示最高欺诈率的小样本组；其余组保留在完整表中。')

    matrix_columns = ['amt', 'age', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'distance_km', 'hour', 'weekday', 'ID', 'unix_time']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, name, title in zip(axes, ['05_pearson', '06_spearman'], ['Pearson', 'Spearman']):
        m = read(name).set_index('feature').loc[matrix_columns, matrix_columns]
        im = ax.imshow(m, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(m)), m.columns, rotation=65, ha='right')
        ax.set_yticks(range(len(m)), m.index)
        ax.set_title(title); ax.grid(False)
    # Reserve an explicit colorbar axis so the legend cannot cover matrix cells.
    fig.suptitle('数值相关性：原始位置、时间与编号存在冗余', x=.03, ha='left', fontsize=15)
    fig.subplots_adjust(left=.09, right=.87, bottom=.25, top=.88, wspace=.38)
    colorbar_axis = fig.add_axes([.91, .25, .012, .63])
    fig.colorbar(im, cax=colorbar_axis)
    fig.savefig(FIGURES/'11_numeric_correlations.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    figures.append({'file': 'figures/11_numeric_correlations.png', 'title': '数值相关性：原始位置、时间与编号存在冗余',
                    'caption': 'ID 与 unix_time 仅作为审计对照展示；不建议把 ID 输入预测模型。完整派生特征相关矩阵见 CSV。'})

    cat_matrix = read('07_categorical_cramers_v').set_index('feature')
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cat_matrix, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(len(cat_matrix)), cat_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(cat_matrix)), cat_matrix.index); ax.grid(False)
    for i in range(len(cat_matrix)):
        for j in range(len(cat_matrix)):
            v = cat_matrix.iloc[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white' if v > .6 else '#222', fontsize=9)
    fig.colorbar(im, ax=ax, label='Bias-corrected Cramer’s V', fraction=.04)
    finish(fig, '12_categorical_dependence', '分类字段之间的依赖关系', 'V 可识别非线性类别依赖；不是因果关系，也不直接衡量对未知交易的预测价值。')

    customers = read('21_customer_profiles')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    palette = {'normal_only': muted, 'mixed': blue, 'fraud_only': red}
    for profile, data in customers.groupby('profile'):
        ax[0].scatter(data.n, data.fraud_rate*100, s=16, alpha=.6, label=profile, color=palette[profile])
    ax[0].set_xscale('log'); ax[0].set_xlabel('客户训练交易数（对数）'); ax[0].set_ylabel('样本内欺诈率 (%)'); ax[0].legend(fontsize=8)
    concentration = read('23_fraud_concentration')
    ax[1].plot(concentration.top_customers, concentration.fraud_share*100, 'o-', color=red)
    ax[1].set_xlabel('按欺诈笔数排名的前 N 位客户'); ax[1].set_ylabel('累计覆盖欺诈交易 (%)')
    finish(fig, '13_customer_structure', '客户支持量、样本内标签纯度与欺诈集中程度', '“仅欺诈客户”是对当前样本的描述，不能证明该客户所有未来交易均为欺诈。')

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    rate_bars(ax[0], read('26_history_count24_bands'), '严格此前 24 小时交易次数')
    relative = read('27_history_relative_amount_bands'); relative['level'] = relative.level.fillna('无历史')
    rate_bars(ax[1], relative, '当前金额 / 严格此前历史均值')
    finish(fig, '14_customer_history', '无标签客户历史描述与欺诈率', '同秒交易互不读取；可使用此前已发生的无标签交易。第一笔交易的历史均值保持缺失。')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    gaps = read('28_history_gap_bands'); gaps['level'] = gaps.level.fillna('无历史')
    rate_bars(ax[0], gaps, '距此前不同时间戳交易的秒数')
    rate_bars(ax[1], read('29_new_merchant'), '是否首次在该商户交易（0/1）')
    finish(fig, '15_history_gap_merchant', '交易间隔与首次商户', '客户身份使用静态组合键；新商户特征只依据此前交易，不读取标签。')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, feature in zip(ax, ['log_amt', 'age']):
        histogram(a, feature, ('train', 'test'))
    finish(fig, '16_numeric_shift', '训练与测试的无标签数值分布', '不同时间窗口和季节组成可能造成分布变化；无法据此判断测试期欺诈率或模型效用。')

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, col in zip(ax, ['category', 'hour']):
        comp = read(f'compare_{col}').sort_values('level')
        pos = np.arange(len(comp))
        a.bar(pos-.2, comp.train_share*100, width=.4, color=blue, label='train')
        a.bar(pos+.2, comp.test_share*100, width=.4, color=red, label='test')
        a.set_xticks(pos, comp.level, rotation=60, ha='right'); a.set_ylabel('交易占比 (%)'); a.set_title(col); a.legend(frameon=False)
    finish(fig, '17_categorical_shift', '类别与时段在训练、测试中的组成变化', '比例均按各自文件归一化；不存在测试标签。')

    drift = read('30_numeric_train_test_shift').sort_values('ks_distance')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    ax[0].barh(drift.feature, drift.ks_distance, color=blue); ax[0].set_xlabel('经验 KS 距离（0–1）')
    ax[1].barh(drift.feature, drift.psi_train_quantile_bins, color=muted); ax[1].set_xlabel('PSI（训练分位点分箱）')
    finish(fig, '18_shift_overview', '无标签特征分布差异的效应量', '不报告大样本下容易显著的 p 值；月份差异包含明确的窗口/季节效应，PSI 不设通用好坏阈值。')

    effects = read('10_numeric_target_association').sort_values('ks_class_difference')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(effects.feature, effects.ks_class_difference, color=blue)
    ax.set_xlabel('正常与欺诈的经验 KS 距离（0–1）')
    finish(fig, '19_marginal_association', '数值特征与标签的边际分布差异', '这不是特征选择结果：弱边际差异仍可能存在交互，强边际差异也未必能跨时间泛化。')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    rate_bars(ax[0], read('19_population_bands'), '城市人口区间')
    rate_bars(ax[1], read('20_distance_bands'), '客户与商户球面距离 (km)')
    finish(fig, '20_geography_population', '位置距离与人口分组', '结合支持量判断，不能把稀疏尾部的高/低欺诈率直接作为规则。')

    composition = read('35_category_month_composition')
    train_comp = composition.loc[composition.dataset == 'train']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, label, title in zip(axes, ['normal', 'fraud'], ['正常交易：每月类别占比', '欺诈交易：每月类别占比']):
        m = train_comp.pivot(index='category', columns='month', values=label).fillna(0)
        share = m/m.sum(axis=0)*100
        im = ax.imshow(share, aspect='auto', cmap='Blues', vmin=0, vmax=100)
        ax.set_yticks(range(len(m)), m.index, fontsize=8)
        ax.set_xticks(range(len(m.columns)), m.columns, rotation=65, ha='right', fontsize=8)
        ax.set_title(title); ax.grid(False)
        fig.colorbar(im, ax=ax, label='该类标签当月交易占比 (%)', fraction=.025)
    finish(fig, '21_label_specific_category_time', '正常与欺诈交易具有不同的类别时间结构',
           '正常仅在 50/252 个类别×月份格中出现，欺诈在全部 252 格出现。这里展示组成占比，零表示真实零计数，不采用风险图的低支持量遮罩。')

    comp = composition.pivot(index='category', columns='month', values='n').fillna(0)
    comp = comp.reindex(sorted(comp.columns), axis=1)
    share = comp/comp.sum(axis=0)*100
    fig, ax = plt.subplots(figsize=(15, 6.5))
    im = ax.imshow(share, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(comp.columns)), comp.columns, rotation=60, ha='right')
    ax.set_yticks(range(len(comp)), comp.index); ax.grid(False)
    ax.axvline(17.5, ls='--', color=red, lw=2)
    fig.colorbar(im, ax=ax, label='当月全部交易中该类别占比 (%)', fraction=.025)
    finish(fig, '22_train_test_category_schedule', '训练与无标签测试的类别时间集中区间',
           '虚线后是无标签测试期。测试同样存在类别集中月份，但覆盖周期与训练不同；不能据此推断测试标签，也不能把全期平均分布接近解释为条件关系稳定。')

    write_json(HERE/'figure_index.json', figures)
    write_reports(HERE, r, figures, markdown)
    write_json(HERE/'render_manifest.json', {'python': platform.python_version(), 'executable': sys.executable,
                                             'matplotlib': matplotlib.__version__, 'markdown': markdown.__version__,
                                             'figures': len(figures), 'report_script_sha256': sha256(HERE/'report_content.py'),
                                             'run_script_sha256': sha256(Path(__file__))})
    log(f'Render complete: {len(figures)} figures, report_zh.md and standalone report_zh.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['analyze', 'render', 'all'], default='all')
    args = parser.parse_args()
    if args.mode in ['analyze', 'all']:
        analyze()
    if args.mode in ['render', 'all']:
        render()
