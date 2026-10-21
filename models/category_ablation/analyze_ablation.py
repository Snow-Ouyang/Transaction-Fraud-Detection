"""Saved-prediction diagnostics only; never fit or optimize future thresholds."""
from pathlib import Path
import base64
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'capacity_comparison'
sys.path.insert(0,str(PRIOR))
from run_comparison import read
from run_core import evaluate,write_json,sha256

LABELS={'CB5':'CB10 对照','CB5_no_category':'CB9 去类别',
        'X5__combined':'XGB13 对照','X5_no_category':'XGB12 去类别',
        'X5_no_category_merchant':'XGB11 去类别/商户'}
PARENTS={'CB5_no_category':'CB5','X5_no_category':'X5__combined','X5_no_category_merchant':'X5__combined'}
ORDER=list(LABELS)
WINDOWS=['D1','D2','D3','June']
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'figure.dpi':140,'font.size':10})


def folder(rid):
    return (PRIOR if rid in ['CB5','X5__combined'] else HERE)/'runs'/rid


def score_summary(frame):
    y=frame.is_fraud.to_numpy();d=frame.decision.to_numpy(bool)
    tp=int(((y==1)&d).sum());fp=int(((y==0)&d).sum())
    fn=int(((y==1)&~d).sum());tn=int(((y==0)&~d).sum())
    return dict(n=len(frame),TP=tp,FP=fp,FN=fn,TN=tn,normal=fp+tn,fraud=tp+fn,
        fpr=fp/(fp+tn) if fp+tn else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,
        utility=50*tp-100*fn-5*fp,score_p10=frame.score.quantile(.1),
        score_median=frame.score.median(),score_p90=frame.score.quantile(.9))


def savefig(name):
    plt.savefig(HERE/'figures'/f'{name}.png',bbox_inches='tight')
    plt.close()


def main():
    assert (HERE/'completion.json').exists()
    p=read(HERE/'protocol.json')
    data=pd.read_pickle(HERE.parent/'catboost_progression/feature_cache.pkl').set_index('ID',drop=False)
    diagnostics=[];category=[];scores=[];by_support=[];pairs=[];error_features=[];coverage=[]
    for spec in p['folds']:
        fold=spec['name'];fit=data.loc[data.timestamp.lt(spec['fit_end'])]
        cal=data.loc[data.timestamp.ge(spec['fit_end'])&data.timestamp.lt(spec['threshold_end'])]
        fit_n=fit.loc[fit.is_fraud.eq(0)].groupby('category').size()
        fit_f=fit.loc[fit.is_fraud.eq(1)].groupby('category').size()
        cal_n=cal.loc[cal.is_fraud.eq(0)].groupby('category').size()
        fit_m=fit.loc[fit.is_fraud.eq(0)].groupby('merchant').size()
        roles=[('threshold',fold,f'{fold}_threshold.csv.gz'),('development',fold,f'{fold}_development.csv.gz')]
        if fold=='D3':roles.append(('diagnostic','June','June_predictions.csv.gz'))
        for role,window,filename in roles:
            predictions={}
            for rid in ORDER:
                f=folder(rid);r=read(f/'result.json')
                rec=next(v for v in r['folds'] if v['fold']==fold)
                threshold=rec['threshold_decision']['threshold']
                pred=pd.read_csv(f/filename,float_precision='round_trip')
                assert np.array_equal(pred.is_fraud,data.loc[pred.ID,'is_fraud'])
                frame=pred.join(data[['timestamp','category','merchant','amt','hour','amount_to_past_mean','count_24h','gap_seconds']],on='ID')
                frame['category_fit_normal_n']=frame.category.map(fit_n).fillna(0).astype(int)
                frame['category_cal_normal_n']=frame.category.map(cal_n).fillna(0).astype(int)
                frame['merchant_fit_normal_n']=frame.merchant.map(fit_m).fillna(0).astype(int)
                frame['fit_normal_support']=np.where(frame.category_fit_normal_n.eq(0),'zero','positive')
                metrics=evaluate(pred.is_fraud,pred.score,threshold)
                diagnostics.append(dict(run_id=rid,label=LABELS[rid],fold=fold,window=window,role=role,**metrics))
                for cat,sub in frame.groupby('category',observed=True):
                    category.append(dict(run_id=rid,window=window,role=role,category=cat,
                        fit_normal_n=int(fit_n.get(cat,0)),fit_fraud_n=int(fit_f.get(cat,0)),
                        threshold_normal_n=int(cal_n.get(cat,0)),**score_summary(sub)))
                for label,sub in frame.groupby('is_fraud',observed=True):
                    scores.append(dict(run_id=rid,window=window,role=role,is_fraud=int(label),
                        threshold=threshold,**score_summary(sub)))
                for support,sub in frame.groupby('fit_normal_support',observed=True):
                    by_support.append(dict(run_id=rid,window=window,role=role,support=support,**score_summary(sub)))
                normals=frame.loc[frame.is_fraud.eq(0)]
                coverage.append(dict(run_id=rid,window=window,role=role,normal=len(normals),
                    category_no_fit_normal=int(normals.category_fit_normal_n.eq(0).sum()),
                    category_no_threshold_normal=int(normals.category_cal_normal_n.eq(0).sum()),
                    merchant_no_fit_normal=int(normals.merchant_fit_normal_n.eq(0).sum())))
                for truth in ['FP','FN']:
                    mask=frame.is_fraud.eq(0)&frame.decision.eq(1) if truth=='FP' else frame.is_fraud.eq(1)&frame.decision.eq(0)
                    subset=frame.loc[mask]
                    row=dict(run_id=rid,window=window,role=role,error=truth,n=len(subset))
                    for col in ['amt','amount_to_past_mean','count_24h','gap_seconds']:
                        for q in [.1,.5,.9]:row[f'{col}_p{int(q*100)}']=subset[col].quantile(q)
                    row['night_22_to_3_share']=subset.hour.astype(int).isin([22,23,0,1,2,3]).mean()
                    row['category_no_fit_normal_share']=subset.category_fit_normal_n.eq(0).mean()
                    error_features.append(row)
                predictions[rid]=pred
            for rid,parent in PARENTS.items():
                a,b=predictions[rid],predictions[parent]
                assert np.array_equal(a.ID,b.ID)
                y=a.is_fraud.to_numpy();ad=a.decision.to_numpy(bool);bd=b.decision.to_numpy(bool)
                pairs.append(dict(run_id=rid,parent=parent,window=window,role=role,
                    FP_fixed=int(((y==0)&bd&~ad).sum()),FP_added=int(((y==0)&~bd&ad).sum()),
                    FN_fixed=int(((y==1)&~bd&ad).sum()),FN_added=int(((y==1)&bd&~ad).sum())))
    frames={name:pd.DataFrame(rows) for name,rows in [
        ('metrics',diagnostics),('category_support_errors',category),('score_distributions',scores),
        ('normal_support_groups',by_support),('paired_changes',pairs),('error_feature_profiles',error_features),('support_coverage',coverage)]}
    m=frames['metrics'];comparisons=[]
    for rid,parent in PARENTS.items():
        for window in WINDOWS:
            a=m.loc[m.run_id.eq(rid)&m.window.eq(window)&m.role.ne('threshold')].iloc[0]
            b=m.loc[m.run_id.eq(parent)&m.window.eq(window)&m.role.ne('threshold')].iloc[0]
            comparisons.append(dict(run_id=rid,parent=parent,window=window,
                delta_utility=int(a.utility-b.utility),delta_utility_per_10000=a.utility_per_10000-b.utility_per_10000,
                delta_FP=int(a.FP-b.FP),delta_FN=int(a.FN-b.FN),fpr=a.false_positive_rate,
                parent_fpr=b.false_positive_rate,recall=a.recall,parent_recall=b.recall,
                delta_vs_always_normal=int(a.delta_vs_always_normal)))
    frames['comparisons']=pd.DataFrame(comparisons)
    for name,frame in frames.items():frame.to_csv(HERE/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    # All plots show fixed prior-window thresholds, not retrospectively optimized ones.
    x=np.arange(4);width=.15
    fig,ax=plt.subplots(figsize=(11,4.8))
    for i,rid in enumerate(ORDER):
        sub=m.loc[m.run_id.eq(rid)&m.role.ne('threshold')].set_index('window').loc[WINDOWS]
        ax.bar(x+(i-2)*width,sub.false_positive_rate*100,width,label=LABELS[rid])
    ax.set(xticks=x,xticklabels=['D1：11–12月','D2：2–3月','D3：5月','6月：复用诊断'],ylabel='正常交易误报率（%）',ylim=(0,110),title='去掉类别及商户后，误报是否跨期下降？')
    ax.legend(ncol=3,fontsize=9);ax.grid(axis='y',alpha=.2);savefig('01_false_positive_rates')
    fig,ax=plt.subplots(figsize=(11,4.8))
    for i,rid in enumerate(PARENTS):
        sub=frames['comparisons'].loc[lambda t:t.run_id.eq(rid)].set_index('window').loc[WINDOWS]
        ax.bar(x+(i-1)*.24,sub.delta_utility_per_10000,.24,label=LABELS[rid])
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=x,xticklabels=['D1','D2','D3','6月：复用诊断'],ylabel='相对各自原模型的效用变化（美元/万笔）',title='误报下降须与漏报、成本一起判断')
    ax.legend();ax.grid(axis='y',alpha=.2);savefig('02_paired_utility')
    fig,axes=plt.subplots(2,3,figsize=(13,7.5));axes=axes.ravel()
    for ax,rid in zip(axes,ORDER):
        s=frames['score_distributions'].loc[lambda t:t.run_id.eq(rid)&t.is_fraud.eq(0)&((t.window.eq('D3'))|(t.window.eq('June')))].copy()
        s=s.set_index(['window','role']).loc[[('D3','threshold'),('D3','development'),('June','diagnostic')]]
        xx=np.arange(3)
        ax.errorbar(xx,s.score_median,yerr=[s.score_median-s.score_p10,s.score_p90-s.score_median],fmt='o-',capsize=4,label='正常分数中位数与10–90分位')
        ax.axhline(float(s.threshold.iloc[0]),color='red',ls='--',label='4月锁定阈值')
        ax.set(title=LABELS[rid],xticks=xx,xticklabels=['4月阈值期','5月开发期','6月诊断'],ylim=(-.05,1.05),ylabel='预测欺诈分数');ax.grid(alpha=.2)
    axes[-1].axis('off');handles,labels=axes[0].get_legend_handles_labels();axes[-1].legend(handles,labels,loc='center',fontsize=10)
    fig.suptitle('同一模型：正常交易分数随时间如何迁移？');fig.tight_layout();savefig('03_normal_score_shift')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    support=frames['normal_support_groups']
    for ax,window in zip(axes,['D3','June']):
        for i,group in enumerate(['positive','zero']):
            vals=[]
            for rid in ORDER:
                z=support.loc[support.run_id.eq(rid)&support.window.eq(window)&support.role.ne('threshold')&support.support.eq(group)]
                vals.append(z.fpr.iloc[0]*100 if len(z) and z.normal.iloc[0]>0 else np.nan)
            ax.bar(np.arange(5)+(i-.5)*.34,vals,.34,label='训练期该类别有正常样本' if group=='positive' else '训练期该类别无正常样本')
        ax.set(title='5月开发期' if window=='D3' else '6月复用诊断',xticks=np.arange(5),xticklabels=[LABELS[r] for r in ORDER],ylim=(0,110),ylabel='误报率（%）')
        ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('按类别在拟合期的正常样本覆盖拆分；缺失柱表示无该组样本');fig.tight_layout();savefig('04_fit_support_errors')
    write_json(HERE/'diagnostics.json',dict(generated_from_saved_predictions=True,no_future_threshold_optimization=True,
        tables={k:len(v) for k,v in frames.items()},figures=4,
        completion_sha256=sha256(HERE/'completion.json')))
    print(frames['comparisons'].to_string(index=False),flush=True)


if __name__=='__main__':main()
