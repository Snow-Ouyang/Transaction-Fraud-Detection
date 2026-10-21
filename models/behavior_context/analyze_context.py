"""Diagnostics from frozen saved results and predictions; no fitting or threshold search."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import SymLogNorm

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'capacity_comparison'))
from run_comparison import read
from run_core import write_json,sha256
from features import NEW

WINDOWS=['D1','D2','D3','June']
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':10,'figure.dpi':145})


def registry():
    p=read(HERE/'protocol.json');res=read(HERE/'results.json');items={}
    for family,b in p['baselines'].items():
        rid=b['source_run'];label=('CatBoost' if family=='catboost' else 'XGBoost')
        items[rid]=dict(family=family,folder=HERE.parent/'category_ablation/runs'/rid,label=label+' 基线',group='baseline',name='无类别基线')
        for g in b['groups']:
            rid=b['tag']+'__'+g['id']
            items[rid]=dict(family=family,folder=HERE/'runs'/rid,label=label+' '+g['name'],group=g['id'],name=g['name'])
        rid=b['tag']+'__combined'
        if rid in res['run_ids']:items[rid]=dict(family=family,folder=HERE/'runs'/rid,label=label+' 合并组',group='combined',name='通过组的合并检验')
    return p,res,items


def savefig(name):
    plt.savefig(HERE/'figures'/f'{name}.png',bbox_inches='tight');plt.close()


def plot_group_gains(changes,groups,p):
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    maxval=max(1,float(np.abs(changes.loc[changes.window.ne('June'),'delta_per_10000']).max()))
    for ax,family in zip(axes,p['baselines']):
        g=groups.loc[groups.family.eq(family)];matrix=[]
        for rid in g.run_id:
            z=changes.loc[changes.run_id.eq(rid)&changes.window.ne('June')].set_index('window').loc[WINDOWS[:3],'delta_per_10000']
            matrix.append([*z.to_list(),float(z.mean())])
        matrix=np.array(matrix)
        im=ax.imshow(matrix,cmap='RdYlGn',norm=SymLogNorm(linthresh=100,vmin=-maxval,vmax=maxval),aspect='auto')
        ax.set(xticks=range(4),xticklabels=['D1','D2','D3','等权均值'],yticks=range(len(g)),yticklabels=[n+('（通过）' if ok else '（未通过）') for n,ok in zip(g.name,g.accepted)],title=family+'：各组独立相对本算法基线')
        for i in range(len(g)):
            for j in range(4):
                tone=float(im.norm(matrix[i,j]))
                ax.text(j,i,f'{matrix[i,j]:+,.0f}',ha='center',va='center',fontsize=9,color='white' if tone<.15 or tone>.85 else 'black')
    fig.colorbar(im,ax=axes,label='Δscore（美元/万笔）；颜色为对称对数，数字为原值',shrink=.7);savefig('01_group_development_gains')


def main():
    # Keep the fresh-directory reproduction self-contained, including the user's
    # official-score clarification, without changing the frozen fit protocol.
    score_audit=HERE/'qa/score_definition_check.json'
    if not score_audit.exists():
        from pypdf import PdfReader
        source=HERE.parents[1]/'ISYE 7406 Project Description.pdf'
        doc=PdfReader(source);text=' '.join(doc.pages[i].extract_text() for i in [1,2])
        assert all(token in text for token in ['$50','$100','$5','30%','70%','Total Utility'])
        write_json(score_audit,dict(source=str(source),source_sha256=sha256(source),
            official_score='Total Utility = 50TP - 100FN - 5FP',higher_is_better=True,
            public_test_fraction_approx=.30,private_test_fraction_approx=.70,source_pages=[2,3],
            selection_policy_unchanged=True,auxiliary=['false_positive_rate','recall','precision','AP','ROC_AUC']))
    p,res,items=registry()
    data=pd.read_pickle(HERE/'feature_cache.pkl').set_index('ID',drop=False)
    rows=[];scores=[];errors=[];category=[];means=[];paired=[];fits=[]
    for rid,meta in items.items():
        run=read(meta['folder']/'result.json')
        means.append(dict(run_id=rid,family=meta['family'],group=meta['group'],name=meta['name'],features=len(run['features']),
            mean_delta_per_10000=run['mean_delta_per_10000'],worst_delta_per_10000=run['worst_delta_per_10000'],
            mean_score_per_10000=float(np.mean([f['development_metrics']['utility_per_10000'] for f in run['folds']])),
            pooled_score=sum(f['development_metrics']['utility'] for f in run['folds']),
            pooled_n=sum(f['development_metrics']['n'] for f in run['folds']),
            windows_above_always_normal=run['windows_above_always_normal'],selected=res['selections'][meta['family']]['final_run']==rid))
        for fold,fspec in zip(run['folds'],p['folds']):
            f=fold['fold'];t=fold['threshold_decision']['threshold']
            fits.append(dict(run_id=rid,family=meta['family'],fold=f,fit_seconds=fold['fit_seconds'],feature_count=len(run['features']),model_bytes=fold['model_bytes']))
            roles=[('threshold',f,f'{f}_threshold.csv.gz',fold['threshold_metrics']),
                   ('development',f,f'{f}_development.csv.gz',fold['development_metrics'])]
            if f=='D3':roles.append(('diagnostic','June','June_predictions.csv.gz',read(meta['folder']/'June_result.json')['selected']))
            fit_normal=data.loc[data.timestamp.lt(fspec['fit_end'])&data.is_fraud.eq(0)].groupby('category',observed=True).size()
            for role,window,filename,metric in roles:
                rows.append(dict(run_id=rid,family=meta['family'],group=meta['group'],features=len(run['features']),fold=f,window=window,role=role,**metric))
                pred=pd.read_csv(meta['folder']/filename,float_precision='round_trip')
                assert np.array_equal(pred.is_fraud,data.loc[pred.ID,'is_fraud'])
                joined=pred.join(data[['category','amt','hour']],on='ID')
                for label,z in pred.groupby('is_fraud'):
                    scores.append(dict(run_id=rid,window=window,role=role,is_fraud=int(label),threshold=t,n=len(z),
                        p10=z.score.quantile(.1),median=z.score.median(),p90=z.score.quantile(.9)))
                for cat,z in joined.groupby('category',observed=True):
                    normal=z.is_fraud.eq(0);fraud=~normal;flag=z.decision.eq(1)
                    fp=int((normal&flag).sum());tp=int((fraud&flag).sum());fn=int((fraud&~flag).sum())
                    category.append(dict(run_id=rid,window=window,role=role,category=cat,n=len(z),normal=int(normal.sum()),
                        fraud=int(fraud.sum()),fit_normal_n=int(fit_normal.get(cat,0)),FP=fp,TP=tp,FN=fn,
                        false_positive_rate=fp/normal.sum() if normal.sum() else np.nan,
                        utility=50*tp-100*fn-5*fp))
                for kind,mask in [('FP',joined.is_fraud.eq(0)&joined.decision.eq(1)),('FN',joined.is_fraud.eq(1)&joined.decision.eq(0))]:
                    subset=data.loc[joined.loc[mask,'ID']]
                    row=dict(run_id=rid,window=window,role=role,error=kind,n=len(subset))
                    for k in ['amt','amount_to_past_mean']+NEW:
                        row[k+'_median']=subset[k].median()
                    errors.append(row)
    metrics=pd.DataFrame(rows);mean=pd.DataFrame(means)
    decisions=[]
    for family,b in p['baselines'].items():
        selection=res['selections'][family]
        for g in selection['groups']:
            decisions.append(dict(family=family,run_id=g['run_id'],group=g['group']['id'],name=g['group']['name'],
                features=len(read(HERE/'runs'/g['run_id']/'result.json')['features']),
                mean_paired_change_per_10000=g['mean_paired_change_per_10000'],improved_windows=g['improved_windows'],accepted=g['accepted']))
        if selection['combined'] is not None:
            g=selection['combined'];rid=b['tag']+'__combined'
            decisions.append(dict(family=family,run_id=rid,group='combined',name='通过组的合并检验',features=len(read(HERE/'runs'/rid/'result.json')['features']),
                mean_paired_change_per_10000=g['mean_paired_change_per_10000'],improved_windows=g['improved_windows'],accepted=g['accepted']))
        base=metrics.loc[metrics.run_id.eq(b['source_run'])&metrics.role.ne('threshold')].set_index('window')
        for rid,meta in items.items():
            if meta['family']!=family or meta['group']=='baseline':continue
            trial=metrics.loc[metrics.run_id.eq(rid)&metrics.role.ne('threshold')].set_index('window')
            for w in WINDOWS:
                a,z=trial.loc[w],base.loc[w]
                paired.append(dict(family=family,run_id=rid,baseline=b['source_run'],group=meta['group'],window=w,
                    delta_utility=int(a.utility-z.utility),delta_per_10000=a.utility_per_10000-z.utility_per_10000,
                    delta_FP=int(a.FP-z.FP),delta_FN=int(a.FN-z.FN),delta_AP=a.average_precision-z.average_precision,
                    delta_ROC_AUC=a.roc_auc-z.roc_auc))
    distributions=[]
    for feature in NEW:
        for label in [0,1]:
            z=data.loc[data.is_fraud.eq(label),feature]
            distributions.append(dict(feature=feature,is_fraud=label,n=len(z),missing=int(z.isna().sum()),
                p01=z.quantile(.01),p10=z.quantile(.1),median=z.median(),p90=z.quantile(.9),p99=z.quantile(.99)))
    frames={'metrics':metrics,'run_summary':mean,'feature_groups':pd.DataFrame(decisions),'paired_changes':pd.DataFrame(paired),
        'score_distributions':pd.DataFrame(scores),'category_errors':pd.DataFrame(category),'error_feature_profiles':pd.DataFrame(errors),
        'new_feature_distributions':pd.DataFrame(distributions),'fits':pd.DataFrame(fits)}
    ranks=mean[['run_id','family','name','features','mean_score_per_10000','pooled_score','pooled_n','selected']].copy()
    ranks['equal_window_score_rank']=ranks.groupby('family').mean_score_per_10000.rank(method='min',ascending=False).astype(int)
    ranks['pooled_total_score_rank']=ranks.groupby('family').pooled_score.rank(method='min',ascending=False).astype(int)
    frames['score_rankings']=ranks.sort_values(['family','equal_window_score_rank'])
    for name,frame in frames.items():frame.to_csv(HERE/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    changes=frames['paired_changes'];groups=frames['feature_groups']
    plot_group_gains(changes,groups,p)
    chosen=[]
    for family,b in p['baselines'].items():chosen.extend([b['source_run'],res['selections'][family]['final_run']])
    chosen=list(dict.fromkeys(chosen))
    legends={rid:('CatBoost' if items[rid]['family']=='catboost' else 'XGBoost')+(' 基线' if items[rid]['group']=='baseline' else f" 入选 {int(mean.loc[mean.run_id.eq(rid),'features'].iloc[0])}项") for rid in chosen}
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for rid in chosen:
        sub=metrics.loc[metrics.run_id.eq(rid)&metrics.role.ne('threshold')].set_index('window').loc[WINDOWS]
        for ax,column in zip(axes,['false_positive_rate','recall']):ax.plot(range(4),sub[column]*100,'o-',label=legends[rid])
    for ax,title in zip(axes,['正常交易误报率','欺诈召回率']):
        ax.set(xticks=range(4),xticklabels=['D1','D2','D3','6月诊断'],ylabel='%',title=title);ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[0].set_ylim(bottom=0)
    fig.tight_layout();savefig('02_selected_error_tradeoff')
    fig,ax=plt.subplots(figsize=(11,4.8));width=.18
    for i,rid in enumerate(chosen):
        sub=metrics.loc[metrics.run_id.eq(rid)&metrics.role.ne('threshold')].set_index('window').loc[WINDOWS]
        ax.bar(np.arange(4)+(i-(len(chosen)-1)/2)*width,sub.delta_per_10000,width,label=legends[rid])
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(4),xticklabels=['D1','D2','D3','6月诊断'],ylabel='相对全部放行的效用（美元/万笔）',title='最终选择取决于开发期成本表现；六月仅复用诊断');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);savefig('03_selected_utility')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    cats=frames['category_errors']
    for ax,(family,b) in zip(axes,p['baselines'].items()):
        ids=list(dict.fromkeys([b['source_run'],b['tag']+'__category_context',res['selections'][family]['final_run']]))
        for i,rid in enumerate(ids):
            z=cats.loc[cats.run_id.eq(rid)&cats.window.eq('June')&cats.role.eq('diagnostic')&cats.normal.gt(0)].set_index('category')
            ax.bar(np.arange(len(z))+(i-(len(ids)-1)/2)*.25,z.false_positive_rate*100,.25,label='基线' if rid==b['source_run'] else ('类别交叉组' if rid.endswith('category_context') else '最终入选'))
        ax.set(xticks=range(len(z)),xticklabels=z.index.tolist(),ylabel='正常交易误报率（%）',title=family+'：类别交叉是否带回误报？');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();savefig('04_category_context_diagnostic')
    fig,axes=plt.subplots(2,2,figsize=(12,7));axes=axes.ravel()
    sd=frames['score_distributions']
    for ax,rid in zip(axes,chosen):
        z=sd.loc[sd.run_id.eq(rid)&sd.is_fraud.eq(0)&(sd.window.eq('D3')|sd.window.eq('June'))].set_index(['window','role']).loc[[('D3','threshold'),('D3','development'),('June','diagnostic')]]
        ax.errorbar(range(3),z['median'],yerr=[z['median']-z.p10,z.p90-z['median']],fmt='o-',capsize=4)
        ax.axhline(z.threshold.iloc[0],color='red',ls='--',label='4月锁定阈值');ax.set(title=legends[rid],xticks=range(3),xticklabels=['4月阈值期','5月开发期','6月诊断'],ylabel='正常分数中位数与10–90分位');ax.legend(fontsize=9);ax.grid(alpha=.2)
    for ax in axes[len(chosen):]:ax.axis('off')
    fig.tight_layout();savefig('05_normal_score_shift')
    write_json(HERE/'diagnostics.json',dict(tables={k:len(v) for k,v in frames.items()},figures=5,no_fits=True,no_future_threshold_optimization=True))
    print(groups.to_string(index=False),flush=True)


if __name__=='__main__':main()
