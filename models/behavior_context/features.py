"""Past-only behavior context; accepts six fields and never any labels."""
from collections import defaultdict
import numpy as np
import pandas as pd

FIELDS=['ID','unix_time','amt','merchant','customer_key','category']
NEW=['count_1h','distinct_merchants_1h','amount_to_mean_7d','mean_7d_to_past_mean',
     'hour_unusualness','category_past_share','amount_to_shrunk_category_mean']
ALPHA=5.0


def behavior_features(events):
    if set(events.columns)!=set(FIELDS):
        raise ValueError('Only the six declared label-free event fields are accepted')
    d=events[FIELDS].sort_values(['unix_time','ID'],kind='stable').reset_index(drop=True)
    n=len(d);output={k:np.full(n,np.nan) for k in NEW}
    times=d.unix_time.to_numpy();amount=d.amt.to_numpy(float)
    merchants=pd.Categorical(d.merchant).codes;categories=pd.Categorical(d.category).codes
    assert (merchants>=0).all() and (categories>=0).all()
    for pos in d.groupby('customer_key',sort=False,observed=True).indices.values():
        t=times[pos];a=amount[pos];mc=merchants[pos];cc=categories[pos]
        end=np.searchsorted(t,t,side='left')
        start1=np.searchsorted(t,t-3600,side='left')
        start7=np.searchsorted(t,t-7*86400,side='left')
        sums=np.r_[0.,a.cumsum()]
        prev=np.divide(sums[end],end,out=np.full(len(pos),np.nan),where=end>0)
        counts7=end-start7
        mean7=np.divide(sums[end]-sums[start7],counts7,out=np.full(len(pos),np.nan),where=counts7>0)
        output['count_1h'][pos]=end-start1
        output['amount_to_mean_7d'][pos]=a/mean7
        output['mean_7d_to_past_mean'][pos]=mean7/prev
        boundaries=np.r_[np.flatnonzero(np.r_[True,t[1:]!=t[:-1]]),len(t)]
        counts=defaultdict(int);left=right=0
        cat_count=np.zeros(int(categories.max())+1,dtype='int64')
        cat_sum=np.zeros(len(cat_count));hour_count=np.zeros(24,dtype='int64')
        for begin,finish in zip(boundaries[:-1],boundaries[1:]):
            while right<begin:
                counts[int(mc[right])]+=1;right+=1
            while left<right and t[left]<t[begin]-3600:
                key=int(mc[left]);counts[key]-=1
                if counts[key]==0:del counts[key]
                left+=1
            ix=pos[begin:finish];c=cc[begin:finish]
            h=((t[begin:finish]//3600)%24).astype(int)
            output['distinct_merchants_1h'][ix]=len(counts)
            output['hour_unusualness'][ix]=-np.log((hour_count[h]+1)/(begin+24))
            if begin>0:
                output['category_past_share'][ix]=cat_count[c]/begin
                mean=(cat_sum[c]+ALPHA*prev[begin])/(cat_count[c]+ALPHA)
                output['amount_to_shrunk_category_mean'][ix]=a[begin:finish]/mean
            # Apply updates only after every transaction at this same second is scored.
            if finish-begin==1:
                cat_count[c[0]]+=1;cat_sum[c[0]]+=a[begin];hour_count[h[0]]+=1
            else:
                np.add.at(cat_count,c,1);np.add.at(cat_sum,c,a[begin:finish]);np.add.at(hour_count,h,1)
    return pd.DataFrame({'ID':d.ID,**output})


def audit(events,out,folder):
    """Independent literal-history recomputation plus prefix/synthetic boundary checks."""
    folder.mkdir(exist_ok=True,parents=True)
    d=events.sort_values(['unix_time','ID'],kind='stable').reset_index(drop=True)
    assert np.array_equal(d.ID,out.ID)
    rng=np.random.default_rng(7406)
    chosen=set(rng.choice(len(d),36,replace=False).tolist())|{0,len(d)-1}
    chosen.update(np.flatnonzero(d.duplicated(['customer_key','unix_time'],keep=False).to_numpy())[:6].tolist())
    spots=[]
    for i in sorted(chosen):
        r=d.iloc[i];past=d.loc[d.customer_key.eq(r.customer_key)&d.unix_time.lt(r.unix_time)]
        one=past.loc[past.unix_time.ge(r.unix_time-3600)]
        seven=past.loc[past.unix_time.ge(r.unix_time-7*86400)]
        same=past.loc[past.category.eq(r.category)]
        mean=past.amt.mean();recent=seven.amt.mean();h=int(r.unix_time//3600%24)
        vals=[len(one),one.merchant.nunique(),r.amt/recent,recent/mean,
              -np.log((int((past.unix_time//3600%24).eq(h).sum())+1)/(len(past)+24)),
              len(same)/len(past) if len(past) else np.nan,
              r.amt/((same.amt.sum()+ALPHA*mean)/(len(same)+ALPHA)) if len(past) else np.nan]
        spots.append({'ID':int(r.ID),**{k:bool(np.isclose(out.iloc[i][k],v,rtol=1e-9,atol=1e-9,equal_nan=True)) for k,v in zip(NEW,vals)}})
    spots=pd.DataFrame(spots);spots.to_csv(folder/'independent_history_spots.csv',index=False,encoding='utf-8-sig')
    cutoff=int(np.datetime64('2024-10-01','s').astype('int64'))
    mask=d.unix_time.lt(cutoff)
    prefix=behavior_features(d.loc[mask,FIELDS])
    prefix_checks={k:bool(np.allclose(prefix[k],out.loc[mask,k],rtol=1e-9,atol=1e-9,equal_nan=True)) for k in NEW}
    toy=pd.DataFrame({'ID':[1,2,3,4,5,6], 'unix_time':[0,3600,3600,3601,7*86400,7*86400+1],
                      'amt':[10.,20.,30.,40.,50.,60.],'merchant':['a','b','c','a','d','a'],
                      'customer_key':[1]*6,'category':['a','b','b','a','b','a']})
    z=behavior_features(toy)
    boundary={'exact_one_hour_inclusive':z.count_1h.iloc[1:3].tolist()==[1.,1.],
              'same_second_excluded':z.category_past_share.iloc[1:3].tolist()==[0.,0.],
              'just_older_than_one_hour_excluded':z.count_1h.iloc[3]==2,
              'seven_day_lower_boundary_inclusive':bool(np.isclose(z.amount_to_mean_7d.iloc[4],50/25)),
              'seven_day_oldest_excluded_after_one_second':bool(np.isclose(z.amount_to_mean_7d.iloc[5],60/35))}
    checks={'label_free_function_input':set(events.columns)==set(FIELDS),
            'direct_independent_spots':bool(spots[NEW].all().all()),
            'future_append_prefix_invariance':all(prefix_checks.values()),
            'all_boundary_tests_passed':all(boundary.values()),
            'no_infinite_features':not np.isinf(out[NEW].to_numpy()).any(),
            'distinct_no_more_than_count':bool(out.distinct_merchants_1h.le(out.count_1h).all()),
            'share_in_unit_interval':bool(out.category_past_share.dropna().between(0,1).all())}
    return dict(checks=checks,independent_spots=len(spots),prefix_rows=len(prefix),
                prefix_checks=prefix_checks,boundary_checks=boundary,
                new_features=NEW,category_shrinkage_alpha=ALPHA)
