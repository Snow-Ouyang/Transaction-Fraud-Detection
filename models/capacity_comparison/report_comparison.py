"""Render saved model comparisons. No model fitting or selection here."""
from pathlib import Path
import base64
import json
import numpy as np
import pandas as pd


def render(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import markdown
    folder=Path(folder)
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    result=read(folder/'results.json');protocol=read(folder/'protocol.json')
    runs={rid:read(folder/'runs'/rid/'result.json') for rid in result['run_ids']}
    selections=result['selections']
    cb=selections['catboost'];xb=selections['xgboost']
    cb_best=cb['parameter_decision']['search_winner'];complex_id=cb['parameter_decision']['best_complex_candidate']
    x_best=xb['parameter_decision']['search_winner']
    final_ids={f:s['final_run'] for f,s in selections.items()}
    from capacity_diagnostics import summarize
    capacity_ids=[c['id'] for family in selections for c in protocol[family+'_grid']]+list(final_ids.values())
    summarize(folder,runs,capacity_ids).to_csv(folder/'tables/realized_capacity_D3.csv',index=False,encoding='utf-8-sig')
    june=pd.read_csv(folder/'tables/june_diagnostic.csv').set_index('run_id')
    rows=[]
    for r in runs.values():
        for f in r['folds']:
            rows.append({'run_id':r['run_id'],'family':r['family'],'fold':f['fold'],**f['development_metrics']})
    development=pd.DataFrame(rows)
    group=pd.read_csv(folder/'tables/feature_groups.csv')
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        fm.fontManager.addfont(str(font));plt.rcParams['font.family']=[fm.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'savefig.dpi':165,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    colors={'catboost':'#32729c','xgboost':'#378574'};red='#bf5a4c';muted='#81909c'
    family_name={'catboost':'CatBoost','xgboost':'XGBoost'}
    figs=[]
    def finish(fig,name,title,caption):
        fig.suptitle(title,x=.025,ha='left',fontsize=15)
        fig.tight_layout(rect=[0,0,1,.89]);fig.savefig(folder/'figures'/f'{name}.png',bbox_inches='tight',facecolor='white')
        plt.close(fig);figs.append({'file':f'figures/{name}.png','title':title,'caption':caption})
    def config_label(r):
        c=r['config'];return f'{r["run_id"]}\n深{c.get("depth",c.get("max_depth"))} / {c.get("iterations",c.get("n_estimators"))}树\nL2={c.get("l2_leaf_reg",c.get("reg_lambda"))}'
    for family,number in [('catboost',1),('xgboost',2)]:
        ids=[c['id'] for c in protocol[family+'_grid']]
        fig,ax=plt.subplots(figsize=(12,5.5));pos=np.arange(len(ids))
        for i,(fold,color) in enumerate(zip(['D1','D2','D3'],[colors[family],muted,red])):
            v=development.loc[development.fold.eq(fold)].set_index('run_id').reindex(ids).delta_per_10000
            ax.bar(pos+(i-1)*.23,v,width=.23,label=fold,color=color)
        ax.plot(pos,[runs[i]['mean_delta_per_10000'] for i in ids],'kD',label='等窗口平均')
        ax.axhline(0,color='#222',lw=.8);ax.set_xticks(pos,[config_label(runs[i]) for i in ids])
        ax.set_ylabel('相对全部放行的效用增量 / 万笔 ($)');ax.legend()
        finish(fig,f'0{number}_{family}_search',family_name[family]+'：固定十项输入的参数搜索',
               '全部候选、窗口和阈值规则预先固定。开发窗口用于选择，不能当作独立测试。L2 数值在两算法间不代表相同约束强度。')

    base_ids=list(dict.fromkeys(['CB0',cb_best,complex_id,x_best]))
    fig,ax=plt.subplots(figsize=(11,5));pos=np.arange(3)
    width=.75/len(base_ids)
    for i,rid in enumerate(base_ids):
        rr=development.loc[development.run_id.eq(rid)].set_index('fold').reindex(['D1','D2','D3'])
        ax.bar(pos+(i-(len(base_ids)-1)/2)*width,rr.delta_per_10000,width=width,label=rid)
    ax.axhline(0,color='#222',lw=.8);ax.set_xticks(pos,['D1：2024/11–12','D2：2025/02–03','D3：2025/05'])
    ax.set_ylabel('相对全部放行的效用增量 / 万笔 ($)');ax.legend()
    finish(fig,'03_fixed10_comparison','相同十项信息下的时间窗口比较',
           'CB0 是上一版参数。同时展示搜索总体最优、最佳复杂 CatBoost 以及最佳 XGBoost，重复身份只画一次。')

    branches=[b['base'] for s in selections.values() for b in s['branches']]
    group_ids=[g['id'] for g in protocol['groups']]
    fig,ax=plt.subplots(figsize=(12,6));pos=np.arange(3);width=.75/len(branches)
    for i,branch in enumerate(branches):
        gg=group.loc[group.branch.eq(branch)&group.group.isin(group_ids)].set_index('group').reindex(group_ids)
        x=pos+(i-(len(branches)-1)/2)*width
        ax.bar(x,gg.mean_paired_change_per_10000,width=width,label=branch)
        for xx,(_,r) in zip(x,gg.iterrows()):
            ax.annotate(f'{int(r.improved_windows)}/3'+(' ✓' if r.accepted else ''),(xx,r.mean_paired_change_per_10000),
                        xytext=(0,5 if r.mean_paired_change_per_10000>=0 else -16),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(0,color='#222',lw=.8);ax.set_xticks(pos,[g['name'] for g in protocol['groups']]);ax.margins(y=.25)
    ax.set_ylabel('相对各自固定十项基线的平均配对变化 / 万笔 ($)');ax.legend()
    finish(fig,'04_group_increments','原参数、复杂 CatBoost、XGBoost 对新增变量的吸收',
           '每组独立加到十项基线上。平均增量为正且至少两个窗口改善才通过，勾号表示通过。不同父版间不能直接比较候选总效用。')

    interactions=[]
    for gid in group_ids:
        old=group.loc[group.branch.eq('CB0')&group.group.eq(gid)].iloc[0]
        new=group.loc[group.branch.eq(complex_id)&group.group.eq(gid)].iloc[0]
        interactions.append({'group':gid,'group_name':old.group_name,'old_increment':old.mean_paired_change_per_10000,
                             'complex_increment':new.mean_paired_change_per_10000,
                             'increment_difference':new.mean_paired_change_per_10000-old.mean_paired_change_per_10000,
                             'old_accepted':bool(old.accepted),'complex_accepted':bool(new.accepted)})
    interaction=pd.DataFrame(interactions);interaction.to_csv(folder/'tables/capacity_interactions.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(range(3),interaction.increment_difference,color=colors['catboost'])
    ax.axhline(0,color='#222',lw=.8);ax.set_xticks(range(3),interaction.group_name)
    ax.set_ylabel('复杂参数新增组增量 − 原参数新增组增量 / 万笔 ($)')
    finish(fig,'05_parameter_group_interaction','所选参数方案是否改变变量增量',
           '两侧均以同样十项输入为父版；年龄/星期的旧参数对照重新训练。参数方案可能同时改变树数、深度和正则化，不能把差分单独归因于容量。')

    final_diag_ids=list(dict.fromkeys(['CB0',cb_best,x_best,*final_ids.values()]))
    fig,ax=plt.subplots(figsize=(12,5));jj=june.loc[final_diag_ids]
    ax.bar(range(len(jj)),jj.utility,color=[colors[runs[rid]['family']] for rid in final_diag_ids])
    ax.axhline(jj.always_normal_utility.iloc[0],color=red,ls='--',label='全部放行')
    ax.set_xticks(range(len(jj)),[rid+('（最终）' if rid in final_ids.values() else '') for rid in final_diag_ids],rotation=18,ha='right')
    ax.set_ylabel('六月复用诊断效用 ($)');ax.legend()
    finish(fig,'06_june_utility','锁定选择后的六月诊断',
           '六月已被此前版本观察和评估，不是盲测。不会按这些结果重选模型、阈值或变量。')

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,(family,rid) in zip(axes,final_ids.items()):
        r=june.loc[rid];count=np.array([[r.TN,r.FP],[r.FN,r.TP]],int);v=np.log1p(count)
        ax.imshow(v,cmap='Blues');ax.grid(False)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{count[i,j]:,}',ha='center',va='center',fontsize=16,color='white' if v[i,j]>v.max()*.65 else '#222')
        ax.set_title(f'{family_name[family]} / {rid}',fontsize=11);ax.set_xticks([0,1],['放行','拦截']);ax.set_yticks([0,1],['正常','欺诈'])
    finish(fig,'07_june_confusion','两个最终研究候选的六月混淆计数','文字为原始笔数，颜色为 log1p；阈值由四月选择，没有六月重训。')

    fig,axes=plt.subplots(1,2,figsize=(13,7))
    for ax,(family,rid) in zip(axes,final_ids.items()):
        imp=pd.read_csv(folder/'runs'/rid/'D3_importance.csv').sort_values('importance')
        ax.barh(imp.feature,imp.importance,color=colors[family]);ax.set_title(f'{family_name[family]} / {rid}',fontsize=11)
        ax.set_xlabel(imp.method.iloc[0])
    finish(fig,'08_importance','最终模型内部特征重要性','左右重要性定义不同、不可直接比数值；相关特征会分担重要性，不能据此判断独立增量或因果。')

    fig,ax=plt.subplots(figsize=(11,5.5))
    for family in selections:
        ids=[c['id'] for c in protocol[family+'_grid']]
        for rid in ids:
            r=runs[rid];seconds=sum(f['fit_seconds'] for f in r['folds'])
            if rid=='CB0':continue
            ax.scatter(seconds,r['mean_delta_per_10000'],color=colors[family],marker='o' if family=='catboost' else '^',s=55)
            ax.annotate(rid,(seconds,r['mean_delta_per_10000']),xytext=(5,3),textcoords='offset points')
        ax.scatter([],[],color=colors[family],marker='o' if family=='catboost' else '^',label=family_name[family])
    ax.set_xscale('log');ax.set_xlabel('三折拟合耗时之和（秒，对数轴）');ax.set_ylabel('开发平均效用增量 / 万笔 ($)');ax.legend()
    finish(fig,'09_compute_utility','更大计算量是否带来更高开发效用',
           '各拟合四线程，两算法独立进程并行；实测耗时受共享机器负载影响。CB0 直接复用，因此不以零秒纳入横轴。')

    _narrative(folder,result,protocol,runs,development,group,june,interaction,figs,markdown)
    (folder/'figure_index.json').write_text(json.dumps(figs,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT COMPLETE: 9 figures, report_zh.md/html',flush=True)


def _narrative(folder,result,protocol,runs,development,group,june,interaction,figs,markdown):
    from narrative_comparison import write
    write(folder,result,protocol,runs,development,group,june,interaction,figs,markdown)
