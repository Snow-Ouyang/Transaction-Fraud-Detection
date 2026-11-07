"""New label-free behavior innovations, persistence, and conditional changes.

All historical gathers use searchsorted(time, side='left'): the complete
current second is excluded. Input rows have a frozen, label-free whitelist.
"""
from functools import lru_cache
import numpy as np
import pandas as pd

SIGNALS=['nf_seq_last5_category_diversity','nf_last5_mean_regime',
 'nf_seq_previous_amount_z','amount_to_mean_7d','category_past_share',
 'merchant_speed_kmh','personal_night_share','peer__rel_category_1d']
PAIRS=[(0,1),(0,2),(0,5),(0,7),(1,2),(1,5),(1,6),(2,5),(2,7),(3,4),(3,5),(4,7)]
FIELDS=['ID','unix_time','customer_key','category','merchant','hour']+SIGNALS+['gap_seconds','count_24h']

def catalog(batch):
    out=[]
    def add(name,op,**kw):out.append(dict(name=f'bc{batch}_{name}',op=op,**kw))
    if batch==1:
        for s in SIGNALS:
            for op in ['median_delta10','z10','z30','regime5_vs20']:
                add(s+'_'+op,op,signal=s)
    elif batch==2:
        for s in SIGNALS:
            for op in ['stress_rate5','stress_shift5_vs20','stress_current_x5']:
                add(s+'_'+op,op,signal=s)
        for a,b in PAIRS:add(f'innovation_product_{a}_{b}','innovation_product',signal=SIGNALS[a],other=SIGNALS[b])
    elif batch==3:
        for s in SIGNALS[:5]:
            for context in ['category','merchant','category_band','transition']:
                for op in ['context_delta','context_z']:
                    add(s+'_'+context+'_'+op,op,signal=s,context=context)
    elif batch==4:
        for s in SIGNALS:
            for op in ['stress_age','signed_persistence20','tail_asymmetry30','innovation_acceleration']:
                add(s+'_'+op,op,signal=s)
    elif batch==5:
        for a,b in PAIRS:
            for op in ['corr20','corr_shift5_vs20','current_x_other_history']:
                add(f'{op}_{a}_{b}',op,signal=SIGNALS[a],other=SIGNALS[b])
    elif batch==6:
        for s in SIGNALS:
            for op in ['stress_share_1h','stress_share_1d','weighted_innovation20','sign_reversal20']:
                add(s+'_'+op,op,signal=s)
    else:raise ValueError(batch)
    return out

def descriptions():
    return {
      'median_delta10':'Current existing behavior signal minus prior10 median (min5).',
      'z10':'Current signal standardized by prior10 mean/population std (min5).',
      'z30':'Current signal standardized by prior30 mean/population std (min5).',
      'regime5_vs20':'Mean of last5 signal values minus disjoint preceding20, scaled by older std; min3/5.',
      'stress_rate5':'Prior5 fraction of historical z10 >1.5 (min3).',
      'stress_shift5_vs20':'Prior5 stress rate minus disjoint preceding20 stress rate (min3/5).',
      'stress_current_x5':'tanh(current z10/3) times prior5 stress fraction.',
      'innovation_product':'Product of two tanh(z10/3) innovations.',
      'context_delta':'Current signal minus same-person contextual past30d mean, shrunk by20 all-person-history pseudo-observations.',
      'context_z':'As context_delta standardized by shrunk contextual second moment.',
      'stress_age':'log1p(seconds since most recent strictly prior z10>1.5 event).',
      'signed_persistence20':'Prior20 mean sign of historical z10 (min5).',
      'tail_asymmetry30':'(q90+q10-2*median)/(q90-q10+1e-6) of prior30 signal values (min10).',
      'innovation_acceleration':'Current z10 minus mean of prior5 z10 values (min3).',
      'corr20':'Prior20 finite paired signal correlation (min5).',
      'corr_shift5_vs20':'Prior5 correlation minus disjoint preceding20 correlation (min3/5).',
      'current_x_other_history':'tanh(current signal z10/3) times mean tanh(other historical z10/3) in prior10 (min5).',
      'stress_share_1h':'Strict prior1h fraction of finite historical z10 exceeding1.5 (min3).',
      'stress_share_1d':'Strict prior1d fraction of finite historical z10 exceeding1.5 (min3).',
      'weighted_innovation20':'Prior20 z10 mean with linear recency weights (min5 finite).',
      'sign_reversal20':'Prior20 adjacent finite historical z10 sign-reversal fraction (min5 pairs).',
    }

def _hist(v,hi,k,op='mean',minimum=5,offset=0):
    series=pd.Series(v,dtype='float64')
    if op=='std':r=series.rolling(k,min_periods=minimum).std(ddof=0)
    elif op=='median':r=series.rolling(k,min_periods=minimum).median()
    elif op.startswith('q'):r=series.rolling(k,min_periods=minimum).quantile(float(op[1:]))
    else:r=series.rolling(k,min_periods=minimum).mean()
    end=hi-offset;ix=np.maximum(end-1,0)
    return np.where(end>0,r.to_numpy()[ix],np.nan)

def _z(v,hi,k=10):
    return (v-_hist(v,hi,k))/(_hist(v,hi,k,'std')+1e-6)

def _corr(a,b,hi,k,minimum,offset=0):
    v=pd.Series(a).rolling(k,min_periods=minimum).corr(pd.Series(b)).to_numpy()
    end=hi-offset
    return np.where(end>0,v[np.maximum(end-1,0)],np.nan)

def _context_keys(d,context):
    if context=='category':return list(zip(d.customer_key,d.category.astype(str)))
    if context=='merchant':return list(zip(d.customer_key,d.merchant.astype(str)))
    if context=='category_band':return list(zip(d.customer_key,d.category.astype(str),d.hour.astype(int)//6))
    time_all=d.unix_time.to_numpy();category_all=d.category.astype(str).to_numpy()
    previous=np.full(len(d),'__AMBIGUOUS_OR_EMPTY__',dtype=object)
    for ix in d.groupby('customer_key',sort=False).indices.values():
        t=time_all[ix];c=category_all[ix]
        starts=np.r_[0,np.flatnonzero(np.diff(t)!=0)+1];ends=np.r_[starts[1:],len(ix)]
        last='__AMBIGUOUS_OR_EMPTY__'
        for lo,hi in zip(starts,ends):
            previous[ix[lo:hi]]=last;u=set(c[lo:hi]);last=next(iter(u)) if len(u)==1 else '__AMBIGUOUS_OR_EMPTY__'
    return list(zip(d.customer_key,previous,d.category.astype(str)))

def build(frame,batch):
    assert set(frame)==set(FIELDS),'Label-free input whitelist only'
    d=frame.reset_index(drop=True);assert d.ID.is_unique and d.unix_time.is_monotonic_increasing
    d=d.sort_values(['unix_time','ID'],kind='stable');restore=d.index.to_numpy();d=d.reset_index(drop=True)
    time_all=d.unix_time.to_numpy();signal_arrays={s:d[s].to_numpy(dtype='float64') for s in SIGNALS}
    specs=catalog(batch);out={s['name']:np.full(len(d),np.nan) for s in specs}
    if batch==3:
        person_mu={};person_sd={}
        for s in SIGNALS[:5]:
            mu=np.full(len(d),np.nan);sd=mu.copy()
            for ix in d.groupby('customer_key',sort=False).indices.values():
                hi=np.searchsorted(time_all[ix],time_all[ix],side='left');v=signal_arrays[s][ix]
                mu[ix]=_hist(v,hi,len(ix),minimum=1);sd[ix]=_hist(v,hi,len(ix),'std',minimum=1)
            person_mu[s]=mu;person_sd[s]=sd
        for context in ['category','merchant','category_band','transition']:
            keys=_context_keys(d,context);codes=pd.factorize(keys,sort=False)[0]
            for ix in pd.Series(np.arange(len(d))).groupby(codes,sort=False).indices.values():
                tt=time_all[ix];hi=np.searchsorted(tt,tt,'left');lo=np.searchsorted(tt,tt-30*86400,'left')
                for s in SIGNALS[:5]:
                    v=signal_arrays[s][ix];ok=np.isfinite(v);c=np.r_[0,np.cumsum(ok)];su=np.r_[0,np.cumsum(np.where(ok,v,0.))];sq=np.r_[0,np.cumsum(np.where(ok,v*v,0.))]
                    n=c[hi]-c[lo];total=su[hi]-su[lo];square=sq[hi]-sq[lo];pm=person_mu[s][ix];ps=person_sd[s][ix]
                    mean=(total+20*pm)/(n+20);var=np.maximum(0,(square+20*(ps*ps+pm*pm))/(n+20)-mean*mean)
                    out[f'bc3_{s}_{context}_context_delta'][ix]=v-mean;out[f'bc3_{s}_{context}_context_z'][ix]=(v-mean)/(np.sqrt(var)+1e-6)
    else:
        for ix in d.groupby('customer_key',sort=False).indices.values():
            tt=time_all[ix];hi=np.searchsorted(tt,tt,'left')
            vals={s:signal_arrays[s][ix] for s in SIGNALS};zs={s:_z(v,hi) for s,v in vals.items()}
            for spec in specs:
                op=spec['op'];s=spec['signal'];v=vals[s];z=zs[s];stress=np.where(np.isfinite(z),(z>1.5).astype(float),np.nan)
                if op=='median_delta10':r=v-_hist(v,hi,10,'median')
                elif op in ['z10','z30']:r=_z(v,hi,int(op[1:]))
                elif op=='regime5_vs20':r=(_hist(v,hi,5,minimum=3)-_hist(v,hi,20,minimum=5,offset=5))/(_hist(v,hi,20,'std',minimum=5,offset=5)+1e-6)
                elif op=='stress_rate5':r=_hist(stress,hi,5,minimum=3)
                elif op=='stress_shift5_vs20':r=_hist(stress,hi,5,minimum=3)-_hist(stress,hi,20,minimum=5,offset=5)
                elif op=='stress_current_x5':r=np.tanh(z/3)*_hist(stress,hi,5,minimum=3)
                elif op=='innovation_product':r=np.tanh(z/3)*np.tanh(zs[spec['other']]/3)
                elif op=='stress_age':
                    last=np.maximum.accumulate(np.where(np.isfinite(z)&(z>1.5),tt,-np.inf));r=np.where(hi>0,np.log1p(tt-last[np.maximum(hi-1,0)]),np.nan)
                elif op=='signed_persistence20':r=_hist(np.sign(z),hi,20,minimum=5)
                elif op=='tail_asymmetry30':
                    q1=_hist(v,hi,30,'q0.1',10);q9=_hist(v,hi,30,'q0.9',10);q5=_hist(v,hi,30,'median',10);r=(q9+q1-2*q5)/(q9-q1+1e-6)
                elif op=='innovation_acceleration':r=z-_hist(z,hi,5,minimum=3)
                elif op=='corr20':r=_corr(v,vals[spec['other']],hi,20,5)
                elif op=='corr_shift5_vs20':r=_corr(v,vals[spec['other']],hi,5,3)-_corr(v,vals[spec['other']],hi,20,5,5)
                elif op=='current_x_other_history':r=np.tanh(z/3)*_hist(np.tanh(zs[spec['other']]/3),hi,10,minimum=5)
                elif op.startswith('stress_share'):
                    sec=3600 if op.endswith('1h') else 86400;lo=np.searchsorted(tt,tt-sec,'left');ok=np.isfinite(stress);c=np.r_[0,np.cumsum(ok)];su=np.r_[0,np.cumsum(np.nan_to_num(stress))];n=c[hi]-c[lo];r=np.divide(su[hi]-su[lo],n,out=np.full(len(ix),np.nan),where=n>=3)
                elif op=='weighted_innovation20':
                    n=len(ix);pos=np.arange(n,dtype=float);ok=np.isfinite(z);cnt=np.r_[0,np.cumsum(ok)];su=np.r_[0,np.cumsum(np.where(ok,z,0))];sz=np.r_[0,np.cumsum(np.where(ok,z*(pos+1),0))];wt=np.r_[0,np.cumsum(np.where(ok,pos+1,0))];lo=np.maximum(hi-20,0)
                    num=sz[hi]-sz[lo]-lo*(su[hi]-su[lo]);den=wt[hi]-wt[lo]-lo*(cnt[hi]-cnt[lo]);r=np.divide(num,den,out=np.full(n,np.nan),where=(cnt[hi]-cnt[lo])>=5)
                elif op=='sign_reversal20':
                    prev=np.r_[np.nan,z[:-1]];turn=np.where(np.isfinite(z)&np.isfinite(prev),(z*prev<0).astype(float),np.nan);r=_hist(turn,hi,20,minimum=5)
                else:raise ValueError(op)
                out[spec['name']][ix]=r
    return pd.DataFrame(out,index=restore).sort_index().replace([np.inf,-np.inf],np.nan)

def literal(frame,i,batch):
    """Independent scalar reference: explicit prior-row masks, direct NumPy stats."""
    row=frame.iloc[i];hist=frame.loc[frame.customer_key.eq(row.customer_key)&frame.unix_time.lt(row.unix_time)].sort_values(['unix_time','ID'],kind='stable')
    def finite(x):return np.asarray(x,dtype='float64')[np.isfinite(np.asarray(x,dtype='float64'))]
    def stat(v,k,op='mean',minimum=5,offset=0):
        end=max(0,len(v)-offset);a=finite(np.asarray(v)[max(0,end-k):end])
        if len(a)<minimum:return np.nan
        if op=='std':return float(np.std(a))
        if op=='median':return float(np.median(a))
        if op.startswith('q'):return float(np.quantile(a,float(op[1:])))
        return float(np.mean(a))
    def innovation(f,j):
        r=f.iloc[j];h=f.loc[f.unix_time.lt(r.unix_time)];v=h[s].to_numpy(dtype='float64')
        return (float(r[s])-stat(v,10))/(stat(v,10,'std')+1e-6)
    zcache={}
    def historical_z(signal):
        if signal not in zcache:
            a=hist[signal].to_numpy(dtype='float64');tt=hist.unix_time.to_numpy();hi=np.searchsorted(tt,tt,'left');zz=np.full(len(hist),np.nan)
            for j,h in enumerate(hi):
                v0=a[max(0,h-10):h];v0=v0[np.isfinite(v0)]
                if len(v0)>=5:zz[j]=(a[j]-np.mean(v0))/(np.std(v0)+1e-6)
            zcache[signal]=zz
        return zcache[signal]
    contexts={}
    def context_history(context):
        if context not in contexts:
            if context=='category':mask=hist.category.astype(str).eq(str(row.category)).to_numpy()
            elif context=='merchant':mask=hist.merchant.astype(str).eq(str(row.merchant)).to_numpy()
            elif context=='category_band':mask=(hist.category.astype(str).eq(str(row.category))&(hist.hour.astype(int)//6==int(row.hour)//6)).to_numpy()
            else:
                local=pd.concat([hist,frame.iloc[[i]]],ignore_index=True);keys=_context_keys(local,'transition');mask=np.array([k==keys[-1] for k in keys[:-1]],dtype=bool)
            contexts[context]=mask & (hist.unix_time.to_numpy()>=row.unix_time-30*86400)
        return contexts[context]
    result={}
    for spec in catalog(batch):
        s=spec['signal'];v=hist[s].to_numpy(dtype='float64');cur=float(row[s]);z=(cur-stat(v,10))/(stat(v,10,'std')+1e-6);op=spec['op']
        if batch in [2,4,5,6]:
            zz=historical_z(s);stress=np.where(np.isfinite(zz),(zz>1.5).astype(float),np.nan)
        if op=='median_delta10':r=cur-stat(v,10,'median')
        elif op in ['z10','z30']:r=(cur-stat(v,int(op[1:])))/(stat(v,int(op[1:]),'std')+1e-6)
        elif op=='regime5_vs20':r=(stat(v,5,minimum=3)-stat(v,20,offset=5))/(stat(v,20,'std',offset=5)+1e-6)
        elif op=='stress_rate5':r=stat(stress,5,minimum=3)
        elif op=='stress_shift5_vs20':r=stat(stress,5,minimum=3)-stat(stress,20,offset=5)
        elif op=='stress_current_x5':r=np.tanh(z/3)*stat(stress,5,minimum=3)
        elif op=='innovation_product':
            w=hist[spec['other']].to_numpy(dtype='float64');other=(float(row[spec['other']])-stat(w,10))/(stat(w,10,'std')+1e-6);r=np.tanh(z/3)*np.tanh(other/3)
        elif batch==3:
            mask=context_history(spec['context'])
            a=finite(hist.loc[mask,s]);pm=stat(v,max(len(v),1),minimum=1);ps=stat(v,max(len(v),1),'std',minimum=1);mu=(a.sum()+20*pm)/(len(a)+20);sd=np.sqrt(max(0,((a*a).sum()+20*(ps*ps+pm*pm))/(len(a)+20)-mu*mu));r=cur-mu if op=='context_delta' else (cur-mu)/(sd+1e-6)
        elif op=='stress_age':
            events=hist.unix_time.to_numpy()[np.isfinite(zz)&(zz>1.5)];r=np.log1p(row.unix_time-events[-1]) if len(events) else np.nan
        elif op=='signed_persistence20':r=stat(np.sign(zz),20)
        elif op=='tail_asymmetry30':r=(stat(v,30,'q0.9',10)+stat(v,30,'q0.1',10)-2*stat(v,30,'median',10))/(stat(v,30,'q0.9',10)-stat(v,30,'q0.1',10)+1e-6)
        elif op=='innovation_acceleration':r=z-stat(zz,5,minimum=3)
        elif op in ['corr20','corr_shift5_vs20']:
            w=hist[spec['other']].to_numpy(dtype='float64')
            def corr(k,minimum,offset=0):
                end=max(0,len(v)-offset);a=v[max(0,end-k):end];b=w[max(0,end-k):end];ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
                return float(np.corrcoef(a,b)[0,1]) if len(a)>=minimum and np.ptp(a)>0 and np.ptp(b)>0 else np.nan
            r=corr(20,5) if op=='corr20' else corr(5,3)-corr(20,5,5)
        elif op=='current_x_other_history':
            old=s;s=spec['other'];otherz=historical_z(s);s=old;r=np.tanh(z/3)*stat(np.tanh(otherz/3),10)
        elif op.startswith('stress_share'):
            secs=3600 if op.endswith('1h') else 86400;a=finite(stress[hist.unix_time.to_numpy()>=row.unix_time-secs]);r=float(a.mean()) if len(a)>=3 else np.nan
        elif op=='weighted_innovation20':
            a=zz[-20:];weights=np.arange(1,len(a)+1,dtype=float);ok=np.isfinite(a);r=float(np.average(a[ok],weights=weights[ok])) if ok.sum()>=5 else np.nan
        elif op=='sign_reversal20':
            prev=np.r_[np.nan,zz[:-1]];turn=np.where(np.isfinite(zz)&np.isfinite(prev),(zz*prev<0).astype(float),np.nan);r=stat(turn,20)
        else:raise ValueError(op)
        result[spec['name']]=r if np.isfinite(r) else np.nan
    return result

