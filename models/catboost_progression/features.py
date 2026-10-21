"""Strictly past, label-free feature construction for the progression experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HISTORY_COLUMNS = ['past_count', 'past_mean_amt', 'gap_seconds', 'count_24h',
                   'new_merchant', 'amount_to_past_mean', 'has_customer_history']


def history_features(events):
    required = ['ID', 'unix_time', 'amt', 'merchant', 'customer_key']
    if set(events.columns) != set(required):
        raise ValueError('History function accepts only five label-free fields')
    d = events[required].sort_values(['unix_time', 'ID'], kind='stable').copy()
    agg = d.groupby(['customer_key', 'unix_time'], sort=True, observed=True).agg(
        events=('amt', 'size'), amount_sum=('amt', 'sum')).reset_index()
    group = agg.groupby('customer_key', sort=False)
    agg['past_count'] = group.events.cumsum()-agg.events
    agg['past_mean_amt'] = (group.amount_sum.cumsum()-agg.amount_sum)/agg.past_count.replace(0, np.nan)
    agg['gap_seconds'] = group.unix_time.diff()
    count24 = np.zeros(len(agg), dtype='int64')
    times, counts = agg.unix_time.to_numpy(), agg.events.to_numpy()
    for positions in group.indices.values():
        t, c = times[positions], counts[positions]
        prefix = np.r_[0, np.cumsum(c)]
        lower = np.searchsorted(t, t-86400, side='left')
        count24[positions] = prefix[:-1]-prefix[lower]
    agg['count_24h'] = count24
    out = d.merge(agg[['customer_key', 'unix_time', 'past_count', 'past_mean_amt', 'gap_seconds', 'count_24h']],
                  on=['customer_key', 'unix_time'], how='left', validate='many_to_one')
    first_time = out.groupby(['customer_key', 'merchant'], observed=True).unix_time.transform('min')
    out['new_merchant'] = out.unix_time.eq(first_time).astype('int8')
    out['amount_to_past_mean'] = out.amt/out.past_mean_amt
    out['has_customer_history'] = out.past_count.gt(0).astype('int8')
    return out.sort_values(['unix_time', 'ID'], kind='stable').reset_index(drop=True)


def build_features(source_path, audit_folder):
    audit_folder = Path(audit_folder)
    audit_folder.mkdir(parents=True, exist_ok=True)
    columns = ['ID', 'unix_time', 'amt', 'category', 'merchant', 'dob', 'job', 'city', 'zip',
               'first', 'last', 'street', 'is_fraud']
    cat = ['category', 'merchant', 'job', 'city', 'first', 'last', 'street']
    d = pd.read_csv(source_path, usecols=columns, dtype={**{k: 'category' for k in cat},
                   'ID': 'int32', 'is_fraud': 'int8', 'zip': 'string', 'dob': 'string'})
    d['timestamp'] = pd.to_datetime(d.unix_time, unit='s')
    birth = pd.to_datetime(d.dob, format='%Y-%m-%d', errors='raise')
    d['age'] = (d.timestamp-birth).dt.total_seconds()/(365.2425*86400)
    d['hour'] = d.timestamp.dt.hour.astype(str)
    d['weekday'] = d.timestamp.dt.dayofweek.astype(str)
    d['month'] = d.timestamp.dt.strftime('%Y-%m')
    d['week'] = d.timestamp.dt.to_period('W-SUN').astype(str)
    d['zip'] = d.zip.str.zfill(5)
    identity = d[['first', 'last', 'street', 'zip']].astype('string')
    identity['dob_iso'] = birth.dt.strftime('%Y-%m-%d')
    d['customer_key'] = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype='uint64')
    assert identity.drop_duplicates().shape[0] == d.customer_key.nunique() == 1099
    d = d.sort_values(['unix_time', 'ID'], kind='stable').reset_index(drop=True)
    events = d[['ID', 'unix_time', 'amt', 'merchant', 'customer_key']]
    hist = history_features(events)
    assert np.array_equal(d.ID, hist.ID)
    for col in HISTORY_COLUMNS:
        d[col] = hist[col].to_numpy()

    rng = np.random.default_rng(7406)
    chosen = set(rng.choice(len(d), size=30, replace=False).tolist())
    chosen.update([0, len(d)-1])
    chosen.update(np.flatnonzero(d.duplicated(['customer_key', 'unix_time'], keep=False).to_numpy())[:2].tolist())
    chosen.update(np.flatnonzero(d.is_fraud.eq(1).to_numpy())[:2].tolist())
    spot = []
    for index in sorted(chosen):
        row = d.iloc[index]
        past = events.loc[events.customer_key.eq(row.customer_key) & events.unix_time.lt(row.unix_time)]
        mean = past.amt.mean() if len(past) else np.nan
        gap = row.unix_time-past.unix_time.max() if len(past) else np.nan
        record = {'ID': int(row.ID),
                  'count_matches': int(row.past_count) == len(past),
                  'mean_matches': bool(np.isclose(row.past_mean_amt, mean, rtol=1e-10, atol=1e-9, equal_nan=True)),
                  'gap_matches': bool(np.isclose(row.gap_seconds, gap, equal_nan=True)),
                  'count24_matches': int(row.count_24h) == int(past.unix_time.ge(row.unix_time-86400).sum()),
                  'new_merchant_matches': bool(row.new_merchant) == (not bool(past.merchant.eq(row.merchant).any()))}
        spot.append(record)
    spot = pd.DataFrame(spot)
    spot.to_csv(audit_folder/'history_independent_spots.csv', index=False, encoding='utf-8-sig')
    cutoff = pd.Timestamp('2024-10-01').timestamp()
    # Use datetime64 rather than host timezone to derive the exact cutoff.
    cutoff = int(np.datetime64('2024-10-01', 's').astype('int64'))
    prefix = events.loc[events.unix_time < cutoff]
    prefix_result = history_features(prefix).set_index('ID')
    full_prefix = d.loc[d.unix_time < cutoff].set_index('ID')
    prefix_checks = {name: bool(np.allclose(prefix_result[name], full_prefix.loc[prefix_result.index, name],
                                          rtol=1e-10, atol=1e-9, equal_nan=True)) for name in HISTORY_COLUMNS}
    checks = {
        'all_rows_preserved': len(d) == 1402996,
        'customer_key_unique_matches_static_combinations': True,
        'strict_past_counts_nonnegative': bool(d.past_count.ge(0).all()),
        'count24_no_more_than_past_count': bool(d.count_24h.le(d.past_count).all()),
        'previous_distinct_time_strictly_earlier': bool(d.gap_seconds.dropna().gt(0).all()),
        'first_event_mean_missing': bool(d.loc[d.past_count.eq(0), 'past_mean_amt'].isna().all()),
        'independent_direct_spots': bool(spot[[c for c in spot if c.endswith('_matches')]].all().all()),
        'future_append_does_not_change_prefix_features': all(prefix_checks.values()),
        'history_input_has_no_labels': 'is_fraud' not in events,
    }
    audit = {'checks': checks, 'prefix_invariance': prefix_checks, 'spot_rows': len(spot),
             'first_history_missing_rows': int(d.past_mean_amt.isna().sum()),
             'normal_relative_amount_median': float(d.loc[d.is_fraud.eq(0), 'amount_to_past_mean'].median()),
             'fraud_relative_amount_median': float(d.loc[d.is_fraud.eq(1), 'amount_to_past_mean'].median()),
             'future_append_prefix_n': len(prefix)}
    (audit_folder/'feature_validation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    assert all(checks.values())
    d = d.drop(columns=['first', 'last', 'street', 'dob'])
    for column in ['category', 'merchant', 'job', 'city', 'zip']:
        d[column] = d[column].astype(str)
    return d, audit
