"""Three predeclared strictly-past XGB30 candidate features; training rows only."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent; R=P.parents[1]
FIELDS=['ID','unix_time','customer_key','amt','category']
NAMES=['tc_near_amount_category_24h_count','tc_near_amount_category_recency','tc_recent_amount_dispersion_change']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def build(d):
 assert set(d.columns)==set(FIELDS) and d.ID.is_unique and d.unix_time.is_monotonic_increasing
 out=np.full((len(d),3),np.nan,dtype='float32'); t=d.unix_time.to_numpy(); a=d.amt.to_numpy(dtype='float64');c=d.category.to_numpy()
 for idx in d.groupby('customer_key',sort=False).indices.values():
  tt=t[idx];aa=a[idx];cc=c[idx];hi=np.searchsorted(tt,tt,side='left');lo=np.searchsorted(tt,tt-86400,side='left')
  for i in range(len(idx)):
   h=hi[i];j=np.arange(lo[i],h); near=j[(cc[j]==cc[i])&(np.abs(np.log1p(aa[j])-np.log1p(aa[i]))<=np.log(1.1))]
   out[idx[i],0]=len(near)
   if len(near):out[idx[i],1]=np.log1p(tt[i]-tt[near[-1]])
   if h>=25:
    new=np.log1p(aa[h-5:h]);old=np.log1p(aa[h-25:h-5]);out[idx[i],2]=np.std(new)-np.std(old)
 return out
def literal(d,i):
 row=d.iloc[i];past=d.loc[(d.customer_key==row.customer_key)&(d.unix_time<row.unix_time)];window=past.loc[(past.unix_time>=row.unix_time-86400)&(past.category==row.category)];near=window.loc[np.abs(np.log1p(window.amt)-np.log1p(row.amt))<=np.log(1.1)];old=past.amt.to_numpy();return np.array([len(near),np.log1p(row.unix_time-near.unix_time.iloc[-1]) if len(near) else np.nan,np.std(np.log1p(old[-5:]))-np.std(np.log1p(old[-25:-5])) if len(old)>=25 else np.nan],dtype='float32')
def main():
 P.mkdir(exist_ok=True,parents=True); (P/'columns').mkdir(exist_ok=True); (P/'qa').mkdir(exist_ok=True)
 cols=['ID','unix_time','amt','category','first','last','street','zip','dob'];types={k:'string' for k in ['category','first','last','street','zip','dob']};d=pd.read_csv(R/'fraudTrain.csv',usecols=cols,dtype=types);d['zip']=d.zip.str.zfill(5);identity=d[['first','last','street','zip']].astype('string');identity['dob_iso']=pd.to_datetime(d.dob,format='mixed').dt.strftime('%Y-%m-%d');d['customer_key']=pd.util.hash_pandas_object(identity,index=False).to_numpy(dtype='uint64');d=d[FIELDS].sort_values(['unix_time','ID'],kind='stable').reset_index(drop=True);np.testing.assert_array_equal(d.ID,np.load(R/'models/block_feature_selection/columns/ID.npy'))
 toy=pd.DataFrame(dict(ID=np.arange(8),unix_time=[0,0,1,2,3600,86400,86400,86401],customer_key=[1]*8,amt=[10.,11.,10.,12.,10.,10.,10.,12.],category=['a','a','a','b','a','a','a','a'])); v=build(toy)
 for i in range(len(toy)):np.testing.assert_allclose(v[i],literal(toy,i),equal_nan=True,atol=1e-6,rtol=0)
 try:build(toy.assign(is_fraud=1));raise RuntimeError('label column accepted')
 except AssertionError:pass
 print('building',len(d),flush=True);values=build(d); rng=np.random.default_rng(7406);ties=np.flatnonzero(d.duplicated(['customer_key','unix_time'],keep=False));spots=np.unique(np.r_[0,len(d)-1,ties[:8],rng.choice(len(d),20,replace=False)])
 for i in spots:np.testing.assert_allclose(values[i],literal(d,int(i)),equal_nan=True,atol=1e-6,rtol=0)
 cut=int(np.searchsorted(d.unix_time.to_numpy(),int(pd.Timestamp('2025-04-01').timestamp()),side='left'));prefix=build(d.iloc[:cut]);np.testing.assert_array_equal(prefix,values[:cut]);del prefix
 paths={}
 for j,n in enumerate(NAMES):
  path=P/'columns'/f'{n}.npy';np.save(path,values[:,j],allow_pickle=False);paths[n]={'path':str(path),'categorical':False,'dtype':'float32','sha256':digest(path)}
 result={'all_passed':True,'rows':len(d),'columns':paths,'source_sha256':digest(R/'fraudTrain.csv'),'builder_sha256':digest(Path(__file__)),'toy_literal':True,'real_points':len(spots),'same_second_excluded':True,'prefix_rows':cut,'label_rejected':True,'no_test':True}
 (P/'qa/features.json').write_text(json.dumps(result,indent=2)+'\n');print('FEATURE_QA_PASSED',flush=True)
if __name__=='__main__':main()

