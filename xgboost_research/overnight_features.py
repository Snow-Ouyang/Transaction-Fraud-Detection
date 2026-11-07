"""Label-free histories; every query uses times strictly less than current time."""
import numpy as np
import pandas as pd

FIELDS=['ID','unix_time','customer_key','amt','merchant','category','hour','weekday','lat','long','merch_lat','merch_long']
CONTEXTS={
    'person_band':['customer_key','band'],
    'person_weekday':['customer_key','weekday'],
    'person_weekend':['customer_key','weekend'],
    'person_merchant':['customer_key','merchant'],
    'person_cat_band':['customer_key','category','band'],
    'person_cat_weekend':['customer_key','category','weekend'],
    'person_cell':['customer_key','cell'],
}
WINDOWS={'30m':1800,'3h':10800,'6h':21600,'3d':259200,'30d':2592000}
GLOBAL={'merchant':['merchant'],'cat_band':['category','band'],'merchant_band':['merchant','band']}
CONTEXT_METRICS=['logcount','shrunk_logratio','shrunk_z','log_recency','logcount7','share7','shrunk_logratio7']
WINDOW_METRICS=['logcount','mean_logratio','z','std_relative','sum_daily_relative','count_rate_relative','mean_regime']
LAST_METRICS=['mean_logratio','z','cv','max_ratio','range_relative','mean_regime','gap_relative']
GLOBAL_METRICS=['mean_logratio7','z7','mean_logratio30','z30','rate7_vs23']
SEQUENCE=['previous_amount_logratio','previous_amount_z','previous_amount_trend','previous_gap_acceleration',
          'category_changed','merchant_changed','last5_category_share','last5_merchant_share','last5_category_diversity','last5_merchant_diversity',
          'last20_category_share','last20_merchant_share','last20_category_diversity','last20_merchant_diversity']
GEO=['home_distance_logratio','home_distance_z','centroid_distance','centroid_scaled_distance','centroid7_distance','location_dispersion']

def catalog():
    specs=[]
    for name,keys in CONTEXTS.items():
        for metric in CONTEXT_METRICS:specs.append(dict(name=f'nf_{name}_{metric}',family='context',context=name,metric=metric,keys=keys))
    for name,seconds in WINDOWS.items():
        for metric in WINDOW_METRICS:specs.append(dict(name=f'nf_window_{name}_{metric}',family='window',seconds=seconds,metric=metric))
    for k in [3,5,10,20]:
        for metric in LAST_METRICS:specs.append(dict(name=f'nf_last{k}_{metric}',family='last',k=k,metric=metric))
    for name,keys in GLOBAL.items():
        for metric in GLOBAL_METRICS:specs.append(dict(name=f'nf_global_{name}_{metric}',family='global',context=name,metric=metric,keys=keys))
    specs += [dict(name=f'nf_seq_{name}',family='sequence',metric=name) for name in SEQUENCE]
    specs += [dict(name=f'nf_geo_{name}',family='geo',metric=name) for name in GEO]
    return specs

def enrich(frame):
    assert set(frame.columns)==set(FIELDS),'Only whitelisted label-free fields accepted'
    d=frame.reset_index(drop=True).copy();assert d.unix_time.is_monotonic_increasing and d.ID.is_unique
    d['hour']=d.hour.astype(int);d['weekday']=d.weekday.astype(int)
    d['band']=d.hour//6;d['weekend']=(d.weekday>=5).astype(int)
    d['cell']=list(zip(np.floor(d.merch_lat*2).astype(int),np.floor(d.merch_long*2).astype(int)))
    return d

def grouped(d,keys,value=None):
    """Composite sorted keys support strict same-second exclusion without row loops."""
    a=d.amt.to_numpy(dtype='float64') if value is None else np.asarray(value,dtype='float64')
    ids=pd.factorize(pd.MultiIndex.from_frame(d[keys]),sort=False)[0].astype('int64')
    t=d.unix_time.to_numpy(dtype='int64');order=np.lexsort((t,ids));inv=np.empty(len(order),dtype='int64');inv[order]=np.arange(len(order))
    tt=t[order];gg=ids[order];aa=a[order];span=int(t.max()-t.min())+400*86400+1
    codes=gg*span+(tt-int(t.min()))+100*86400
    hi=np.searchsorted(codes,codes,side='left');start=np.r_[0,np.flatnonzero(gg[1:]!=gg[:-1])+1]
    group_start=np.repeat(start,np.diff(np.r_[start,len(order)]))
    cumulative=pd.Series(aa).groupby(gg,sort=False).cumsum().to_numpy()
    cumulative_sq=pd.Series(aa*aa).groupby(gg,sort=False).cumsum().to_numpy()
    def prefix(values,end):return np.where(end>group_start,values[np.maximum(end-1,0)],0.)
    def stats(seconds=None):
        lo=group_start if seconds is None else np.maximum(group_start,np.searchsorted(codes,codes-seconds,side='left'))
        count=hi-lo;total=prefix(cumulative,hi)-prefix(cumulative,lo);square=prefix(cumulative_sq,hi)-prefix(cumulative_sq,lo)
        mean=np.divide(total,count,out=np.full(len(d),np.nan),where=count>0)
        std=np.sqrt(np.maximum(0,np.divide(square,count,out=np.full(len(d),np.nan),where=count>0)-mean*mean));std[count==1]=0.
        return tuple(v[inv] for v in [count,total,mean,std])
    count=hi-group_start;previous=np.maximum(hi-1,0)
    recency=np.where(count>0,tt-tt[previous],np.nan)[inv]
    tenure=(tt-tt[group_start])[inv]
    return stats,recency,tenure

def distance(a,b,c,e):
    a,b,c,e=[np.radians(np.asarray(v,dtype='float64')) for v in [a,b,c,e]]
    z=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((e-b)/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(z,0,1)))

def build(frame):
    d=enrich(frame);n=len(d);amt=d.amt.to_numpy(dtype='float64');out={}
    personal,gap,tenure=grouped(d,['customer_key']);pn,ps,pm,pstd=personal();cn7,_,cm7,_=personal(7*86400)
    daily=np.maximum(tenure/86400,1.)
    for label,keys in CONTEXTS.items():
        stats,recency,_=grouped(d,keys);count,total,mean,std=stats();count7,total7,mean7,std7=stats(7*86400)
        shrunk=(total+5*pm)/(count+5);second=(count*(std*std+mean*mean))
        second=np.nan_to_num(second,nan=0.)
        shrunkstd=np.sqrt(np.maximum(0,(second+5*(pstd*pstd+pm*pm))/(count+5)-shrunk*shrunk))
        values=[np.log1p(count),np.log1p(amt)-np.log1p(shrunk),(amt-shrunk)/(shrunkstd+1),np.log1p(recency),np.log1p(count7),
                (count7+.5)/(cn7+1),np.log1p(amt)-np.log1p((total7+5*pm)/(count7+5))]
        for metric,v in zip(CONTEXT_METRICS,values):out[f'nf_{label}_{metric}']=v
    for label,seconds in WINDOWS.items():
        count,total,mean,std=personal(seconds)
        values=[np.log1p(count),np.log1p(amt)-np.log1p(mean),(amt-mean)/(std+1),std/(pstd+1),
                total/(ps/daily+1), (count+.5)/((pn/daily)*(seconds/86400)+.5),np.log1p(mean)-np.log1p(pm)]
        for metric,v in zip(WINDOW_METRICS,values):out[f'nf_window_{label}_{metric}']=v
    for label,keys in GLOBAL.items():
        stats,_,_=grouped(d,keys);n7,_,mu7,sd7=stats(7*86400);n30,_,mu30,sd30=stats(30*86400)
        values=[np.log1p(amt)-np.log1p(mu7),(amt-mu7)/(sd7+1),np.log1p(amt)-np.log1p(mu30),(amt-mu30)/(sd30+1),
                np.log((n7+.5)/((n30-n7)*7/23+.5))]
        for metric,v in zip(GLOBAL_METRICS,values):out[f'nf_global_{label}_{metric}']=v
    for name in LAST_METRICS:
        for k in [3,5,10,20]:out[f'nf_last{k}_{name}']=np.full(n,np.nan)
    for metric in SEQUENCE:out[f'nf_seq_{metric}']=np.full(n,np.nan)
    all_t=d.unix_time.to_numpy();all_cc=d.category.to_numpy();all_mm=d.merchant.to_numpy()
    for idx in d.groupby('customer_key',sort=False,observed=True).indices.values():
        t=all_t[idx];a=amt[idx];cc=all_cc[idx];mm=all_mm[idx]
        hi=np.searchsorted(t,t,side='left');cs=np.r_[0.,np.cumsum(a)];sq=np.r_[0.,np.cumsum(a*a)]
        for k in [3,5,10,20]:
            lo=np.maximum(0,hi-k);count=hi-lo
            mean=np.divide(cs[hi]-cs[lo],count,out=np.full(len(idx),np.nan),where=count>0)
            std=np.sqrt(np.maximum(0,np.divide(sq[hi]-sq[lo],count,out=np.full(len(idx),np.nan),where=count>0)-mean*mean));std[count==1]=0.
            mx=pd.Series(a).rolling(k,min_periods=1).max().to_numpy()[np.maximum(hi-1,0)];mn=pd.Series(a).rolling(k,min_periods=1).min().to_numpy()[np.maximum(hi-1,0)]
            mx=np.where(count>0,mx,np.nan);mn=np.where(count>0,mn,np.nan)
            # Cadence spans previous distinct-time event and earliest retained event.
            cadence=np.divide(t[np.maximum(hi-1,0)]-t[lo],np.maximum(count-1,1))
            values=[np.log1p(a)-np.log1p(mean),(a-mean)/(std+1),std/(mean+1),a/(mx+1),(mx-mn)/(mean+1),
                    np.log1p(mean)-np.log1p(pm[idx]),gap[idx]/(cadence+1)]
            for metric,v in zip(LAST_METRICS,values):out[f'nf_last{k}_{metric}'][idx]=v
        starts=np.r_[0,np.flatnonzero(t[1:]!=t[:-1])+1];ends=np.r_[starts[1:],len(idx)]
        for j,(s,e) in enumerate(zip(starts,ends)):
            if not j:continue
            ps0,pe0=starts[j-1],ends[j-1];prev=a[ps0:pe0];ix=idx[s:e]
            out['nf_seq_previous_amount_logratio'][ix]=np.log1p(a[s:e])-np.log1p(prev.mean())
            out['nf_seq_previous_amount_z'][ix]=(a[s:e]-prev.mean())/(prev.std()+1)
            out['nf_seq_category_changed'][ix]=[float(v not in cc[ps0:pe0]) for v in cc[s:e]]
            out['nf_seq_merchant_changed'][ix]=[float(v not in mm[ps0:pe0]) for v in mm[s:e]]
            if j>=2:
                pp=a[starts[j-2]:ends[j-2]].mean()
                out['nf_seq_previous_amount_trend'][ix]=np.log1p(prev.mean())-np.log1p(pp)
                out['nf_seq_previous_gap_acceleration'][ix]=np.log1p(t[s]-t[ps0])-np.log1p(t[ps0]-t[starts[j-2]])
            for k in [5,20]:
                lo=max(0,s-k);cprev=cc[lo:s];mprev=mm[lo:s]
                for dim,values,current in [('category',cprev,cc[s:e]),('merchant',mprev,mm[s:e])]:
                    out[f'nf_seq_last{k}_{dim}_share'][ix]=[np.mean(values==v) for v in current]
                    out[f'nf_seq_last{k}_{dim}_diversity'][ix]=len(set(values))/len(values)
    hd=distance(d.lat,d['long'],d.merch_lat,d.merch_long)
    ds,_,_=grouped(d,['customer_key'],hd);_,_,dm,dstd=ds()
    la,_,_=grouped(d,['customer_key'],d.merch_lat);_,_,lm,ls=la();_,_,lm7,_=la(7*86400)
    ln,_,_=grouped(d,['customer_key'],d.merch_long);_,_,nm,ns=ln();_,_,nm7,_=ln(7*86400)
    cd=distance(lm,nm,d.merch_lat,d.merch_long);disp=111.2*np.sqrt(ls*ls+(ns*np.cos(np.radians(lm)))**2)
    values=[np.log1p(hd)-np.log1p(dm),(hd-dm)/(dstd+1),cd,cd/(disp+1),distance(lm7,nm7,d.merch_lat,d.merch_long),disp]
    for metric,v in zip(GEO,values):out[f'nf_geo_{metric}']=v
    result=pd.DataFrame(out);assert set(result)=={s['name'] for s in catalog()}
    assert not np.isinf(result.to_numpy()).any()
    return result

def literal(frame,i):
    """Independent direct subset-history formulas for every candidate at one event."""
    d=enrich(frame);r=d.iloc[i];a=float(r.amt);t=int(r.unix_time);past=d.loc[(d.customer_key==r.customer_key)&(d.unix_time<t)]
    pm=past.amt.mean();sd=past.amt.std(ddof=0);n=len(past);total=past.amt.sum();p7=past.loc[past.unix_time>=t-7*86400]
    age=max((t-past.unix_time.min())/86400,1.) if n else 1.;out={}
    def stats(z):return len(z),z.amt.sum(),z.amt.mean(),z.amt.std(ddof=0)
    for label,keys in CONTEXTS.items():
        z=past
        for key in keys:
            if key!='customer_key':z=z.loc[z[key].map(lambda v:v==r[key])]
        count,s,mean,std=stats(z);z7=z.loc[z.unix_time>=t-7*86400];n7,s7,_,_=stats(z7)
        sm=(s+5*pm)/(count+5);v=(np.square(z.amt).sum()+5*(sd*sd+pm*pm))/(count+5)-sm*sm
        values=[np.log1p(count),np.log1p(a)-np.log1p(sm),(a-sm)/(np.sqrt(max(v,0))+1),np.log1p(t-z.unix_time.max()) if count else np.nan,
                np.log1p(n7),(n7+.5)/(len(p7)+1),np.log1p(a)-np.log1p((s7+5*pm)/(n7+5))]
        for metric,value in zip(CONTEXT_METRICS,values):out[f'nf_{label}_{metric}']=value
    for label,seconds in WINDOWS.items():
        z=past.loc[past.unix_time>=t-seconds];count,s,mean,std=stats(z)
        values=[np.log1p(count),np.log1p(a)-np.log1p(mean),(a-mean)/(std+1),std/(sd+1),s/(total/age+1),
                (count+.5)/((n/age)*seconds/86400+.5),np.log1p(mean)-np.log1p(pm)]
        for metric,value in zip(WINDOW_METRICS,values):out[f'nf_window_{label}_{metric}']=value
    for label,keys in GLOBAL.items():
        z=d.loc[d.unix_time<t]
        for key in keys:z=z.loc[z[key].map(lambda v:v==r[key])]
        n7,_,m7,s7=stats(z.loc[z.unix_time>=t-7*86400]);n30,_,m30,s30=stats(z.loc[z.unix_time>=t-30*86400])
        values=[np.log1p(a)-np.log1p(m7),(a-m7)/(s7+1),np.log1p(a)-np.log1p(m30),(a-m30)/(s30+1),np.log((n7+.5)/((n30-n7)*7/23+.5))]
        for metric,value in zip(GLOBAL_METRICS,values):out[f'nf_global_{label}_{metric}']=value
    gap=t-past.unix_time.max() if n else np.nan
    for k in [3,5,10,20]:
        z=past.tail(k);cnt,s,mean,std=stats(z);mx=z.amt.max();mn=z.amt.min()
        cadence=(z.unix_time.max()-z.unix_time.min())/max(cnt-1,1) if cnt else 0.
        values=[np.log1p(a)-np.log1p(mean),(a-mean)/(std+1),std/(mean+1),a/(mx+1),(mx-mn)/(mean+1),np.log1p(mean)-np.log1p(pm),gap/(cadence+1)]
        for metric,value in zip(LAST_METRICS,values):out[f'nf_last{k}_{metric}']=value
    for metric in SEQUENCE:out[f'nf_seq_{metric}']=np.nan
    if n:
        prev=past.loc[past.unix_time==past.unix_time.max()];earlier=past.loc[past.unix_time<prev.unix_time.iloc[0]]
        out['nf_seq_previous_amount_logratio']=np.log1p(a)-np.log1p(prev.amt.mean())
        out['nf_seq_previous_amount_z']=(a-prev.amt.mean())/(prev.amt.std(ddof=0)+1)
        out['nf_seq_category_changed']=float(r.category not in set(prev.category));out['nf_seq_merchant_changed']=float(r.merchant not in set(prev.merchant))
        if len(earlier):
            pp=earlier.loc[earlier.unix_time==earlier.unix_time.max()]
            out['nf_seq_previous_amount_trend']=np.log1p(prev.amt.mean())-np.log1p(pp.amt.mean())
            out['nf_seq_previous_gap_acceleration']=np.log1p(t-prev.unix_time.iloc[0])-np.log1p(prev.unix_time.iloc[0]-pp.unix_time.iloc[0])
        for k in [5,20]:
            z=past.tail(k)
            for dim in ['category','merchant']:
                out[f'nf_seq_last{k}_{dim}_share']=float((z[dim]==r[dim]).mean())
                out[f'nf_seq_last{k}_{dim}_diversity']=z[dim].nunique()/len(z)
    hd=float(distance(r.lat,r['long'],r.merch_lat,r.merch_long));dist=distance(past.lat,past['long'],past.merch_lat,past.merch_long)
    lm=past.merch_lat.mean();nm=past.merch_long.mean();cd=float(distance(lm,nm,r.merch_lat,r.merch_long))
    disp=111.2*np.sqrt(past.merch_lat.std(ddof=0)**2+(past.merch_long.std(ddof=0)*np.cos(np.radians(lm)))**2)
    values=[np.log1p(hd)-np.log1p(np.mean(dist)),(hd-np.mean(dist))/(np.std(dist)+1),cd,cd/(disp+1),
            float(distance(p7.merch_lat.mean(),p7.merch_long.mean(),r.merch_lat,r.merch_long)),disp]
    for metric,value in zip(GEO,values):out[f'nf_geo_{metric}']=value
    return out

