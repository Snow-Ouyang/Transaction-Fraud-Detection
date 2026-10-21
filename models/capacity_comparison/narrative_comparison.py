"""Chinese findings from frozen, saved experiment results only."""
from pathlib import Path
import base64
import json
import pandas as pd


def write(folder,result,protocol,runs,development,group,june,interaction,figs,markdown):
    selections=result['selections'];cb=selections['catboost'];xb=selections['xgboost']
    cb_best=cb['parameter_decision']['search_winner'];cb_complex=cb['parameter_decision']['best_complex_candidate']
    x_best=xb['parameter_decision']['search_winner'];old=runs['CB0']
    final_ids={f:s['final_run'] for f,s in selections.items()}
    def n(x,d=0):return '未定义' if pd.isna(x) else f'{x:,.{d}f}'
    def signed(x):return f'{x:+,.2f}'
    def pct(x):return '未定义' if pd.isna(x) else f'{100*x:.2f}%'
    def table(df,cols):
        out=['| '+' | '.join(label for _,label,_ in cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
        for _,r in df.iterrows():out.append('| '+' | '.join(str(func(r[key]) if func else r[key]).replace('|','/') for key,_,func in cols)+' |')
        return '\n'.join(out)
    def image(i):
        f=figs[i-1];return f'![{f["title"]}]({f["file"]})\n\n{f["caption"]}\n'
    def config(rid):
        c=runs[rid]['config']
        text=f'深度 {c.get("depth",c.get("max_depth"))}、{c.get("iterations",c.get("n_estimators"))} 棵树、L2={c.get("l2_leaf_reg",c.get("reg_lambda"))}'
        if 'min_child_weight' in c:text+=f'、min_child_weight={c["min_child_weight"]}'
        return text
    final_rows=[]
    for family,rid in final_ids.items():
        r=runs[rid];j=june.loc[rid]
        final_rows.append({'family':family,'run_id':rid,'features':len(r['features']),'config':config(rid),
                           'mean_delta':r['mean_delta_per_10000'],'positive_windows':r['windows_above_always_normal'],
                           'june_utility':j.utility,'june_FP':j.FP,'june_recall':j.recall,'june_precision':j.precision,
                           'june_fpr':j.false_positive_rate})
    finals=pd.DataFrame(final_rows)
    finals.to_csv(folder/'tables/final_comparison.csv',index=False,encoding='utf-8-sig')
    comparison_ids=list(dict.fromkeys(['CB0',cb_best,cb_complex,x_best,*final_ids.values()]))
    dev_comparison=development.loc[development.run_id.isin(comparison_ids)]
    dev_comparison.to_csv(folder/'tables/head_to_head_development.csv',index=False,encoding='utf-8-sig')
    capacity_gain=runs[cb_complex]['mean_delta_per_10000']-old['mean_delta_per_10000']
    best_gain=runs[cb_best]['mean_delta_per_10000']-old['mean_delta_per_10000']
    x_gain=runs[x_best]['mean_delta_per_10000']-old['mean_delta_per_10000']
    group_summaries=[]
    for family,s in selections.items():
        for b in s['branches']:
            subset=group.loc[group.branch.eq(b['base'])&group.group.ne('combined')]
            yes='、'.join(subset.loc[subset.accepted,'group_name']) or '无'
            no='、'.join(subset.loc[~subset.accepted,'group_name']) or '无'
            group_summaries.append(f'- **{family} / {b["base"]}**：单组通过 **{yes}**；未通过 **{no}**。')
    failed_all=all(runs[r]['windows_above_always_normal']==0 for r in final_ids.values())
    verdict=('两种最终候选的开发平均效用仍低于全部放行；尽管个别窗口改善，本轮尚未得到跨窗口稳定的效用优势。'
             if all(runs[r]['mean_delta_per_10000']<0 for r in final_ids.values())
             else '两个最终候选在三个开发窗口均未超过全部放行；本轮尚未通过效用基线。' if failed_all
             else '部分开发窗口超过全部放行，但这些窗口已参与选择；仍需全新数据确认泛化。')
    parts=[f'''# 固定十项输入：CatBoost 容量搜索、XGBoost 与变量重试 v3

实验日期：2026-10-21。完整保留 v1/v2 模型与结果，本轮未生成官方测试预测或提交。

## 1. 核心结论

**{verdict}**

- 固定十项输入后，CatBoost 搜索总体最优为 **`{cb_best}`**（{config(cb_best)}），相对原参数 CB0 的开发平均效用变化 **${best_gain:+,.2f}/万笔**。
- 最佳复杂 CatBoost 为 **`{cb_complex}`**（{config(cb_complex)}），相对 CB0 变化 **${capacity_gain:+,.2f}/万笔**。复杂分支实际进行了变量重试；“最佳复杂”只指预定五组复杂参数中的最优，不自动优于原模型。
- XGBoost 六组搜索最优为 **`{x_best}`**（{config(x_best)}），相同十项信息下相对 CB0 变化 **${x_gain:+,.2f}/万笔**。相对搜索总体最优 CatBoost 的变化为 **${runs[x_best]['mean_delta_per_10000']-runs[cb_best]['mean_delta_per_10000']:+,.2f}/万笔**。
- 最终候选含有各自参数与变量选择，列于下表。所有“通过”只表示在开发窗口满足研究保留规则，不表示独立测试通过。

单组检验的取舍：

'''+ '\n'.join(group_summaries)+'\n\n'+table(finals,[('family','算法',None),('run_id','最终候选',None),('features','输入数',n),('config','参数',None),
                 ('mean_delta','开发平均相对放行 / 万笔 ($)',lambda x:n(x,2)),('positive_windows','优于放行窗口数',n),
                 ('june_utility','六月复用效用 ($)',n),('june_FP','六月误报',n)])+'''

开发窗口已经在前两轮被使用，本轮再次用于参数、变量和组合选择，存在选择乐观偏差。六月也早已观察与评估，只在本轮两种算法选择全部锁定之后复用诊断。**本轮没有新的独立测试结果。**

## 2. 本轮回答什么问题

本轮将两个问题分开：①同样十项信息下，提高模型容量或更换算法是否有收益；②固定所选参数后，原先未保留变量是否出现增量。不能因为变量变多，就预设复杂模型一定更好。

固定输入为：`amt, category, hour, amount_to_past_mean, has_customer_history, past_count, past_mean_amt, gap_seconds, count_24h, new_merchant`。

| 折 | 拟合截至（不含） | 阈值选择窗口 | 开发选择窗口 |
| --- | --- | --- | --- |
| D1 | 2024-10-01 | 2024/10 | 2024/11–12 |
| D2 | 2025-01-01 | 2025/01 | 2025/02–03 |
| D3 | 2025-04-01 | 2025/04 | 2025/05 |

项目效用 `U=50TP−100FN−5FP`，相对全部放行的增量 `ΔU=150TP−5FP`。每个窗口先换算为每万笔，再对三个窗口等权平均。阈值只在此前窗口扫描全部不同分数；并列分数整体处理、效用并列优先较少拦截，包含全部放行与全部拦截。

参数按平均效用增量排名，再比较最差窗口、较浅深度、较少树数、较强 L2。没有早停、额外调参、类别权重或重采样。所有拟合使用完整自然比例数据，学习率统一0.05、种子7406、每次四线程。

协议、代码和输入在新拟合前写入 [manifest.json](manifest.json)。参数、变量或诊断结果不会覆盖旧版。
''']
    for family,index,label in [('catboost',1,'CatBoost'),('xgboost',2,'XGBoost')]:
        rr=[]
        for ranked in selections[family]['parameter_decision']['ranked']:
            rid=ranked['run_id'];r=runs[rid]
            rr.append({'run_id':rid,'config':config(rid),'mean':r['mean_delta_per_10000'],'worst':r['worst_delta_per_10000'],
                       'wins':r['windows_above_always_normal'],'fit_seconds':sum(f['fit_seconds'] for f in r['folds'])})
        if family=='catboost':
            intro='CatBoost 采用 Plain boosting、按时间输入与 has_time=True，分类计数 SkipTest。CB0 复用上一版十项输入模型，其余五组实际重训；候选增加树数、深度，并包含更强正则化。'
        else:
            intro='XGBoost 使用 CPU hist 原生分类。每折分类词表仅从拟合期建立，未来新类别映射为缺失，已知字符串词表保持一致。数值缺失保留，不做外部目标编码，模型保存为 UBJSON。X4/X5 同时增加 min_child_weight；六组是预定有限设计，不是全网格。'
        parts.append(f'''## {3 if family=='catboost' else 4}. {label} 固定十项输入的搜索

{intro}

'''+table(pd.DataFrame(rr),[('run_id','候选',None),('config','参数',None),('mean','平均相对放行 / 万笔 ($)',lambda x:n(x,2)),
                           ('worst','最差窗口 / 万笔 ($)',lambda x:n(x,2)),('wins','优于放行窗口数',n),('fit_seconds','三折拟合秒数',lambda x:n(x,1))])+'\n\n'+image(index))
    parts.append('''分类处理参考 [XGBoost 原生分类文档](https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/categorical.html)，复杂度与约束含义参考 [XGBoost 参数文档](https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html) 和 [CatBoost 调参文档](https://catboost.ai/docs/en/concepts/parameter-tuning)。本地实际版本以运行环境记录为准。

相同输入、时间切分和效用口径使模型管线可比；相同树数、深度和 L2 数字并不意味着两算法拥有完全相同的有效容量。分类表示与树结构差异也属于此次模型管线比较。

'''+image(3)+image(9))

    gg=group.copy();gg['decision']=gg.accepted.map({True:'通过',False:'未通过'})
    for fold in ['D1','D2','D3']:gg[fold]=pd.to_numeric(gg[fold])
    merchant_gain=group.loc[group.branch.eq(x_best)&group.group.eq('merchant')].iloc[0]
    parts.append('''## 5. 待定变量重试与容量交互

CatBoost 在 CB0 和最佳复杂参数两个分支上重试；XGBoost 在搜索最优参数上重试。每组都独立加入相同的十项基线，参数保持不变。平均配对效用增量为正且至少2/3开发窗口改善才通过。

| 组 | 新增字段 |
| --- | --- |
| 年龄与星期 | age, weekday |
| 商户名称 | merchant |
| 职业、城市、邮编 | job, city, zip |

'''+table(gg,[('family','算法',None),('branch','父版',None),('group_name','新增组',None),('feature_count','输入数',n),
              ('mean_paired_change_per_10000','平均配对变化 / 万笔 ($)',signed),('improved_windows','改善窗口数',n),('decision','决定',None)])+f'''

保留规则以零增量为门槛，不是显著性检验，也未设实用收益下限。例如 XGBoost 的商户单组平均变化为 **${merchant_gain.mean_paired_change_per_10000:+,.2f}/万笔**，应结合幅度和窗口差异解读，不能仅凭“通过”宣称稳定有效。

'''+image(4)+'''
逐窗口变化：

'''+table(gg,[('run_id','候选',None),('D1','D1变化 / 万笔 ($)',signed),('D2','D2变化 / 万笔 ($)',signed),('D3','D3变化 / 万笔 ($)',signed)])+'''

上轮年龄/星期试验以五项输入为父版，本轮以十项为父版；为消除这个差异，本轮重新训练原参数下的“十项＋年龄/星期”。原参数商户和身份代理试验与本轮口径完全一致，因此直接复用，并重算阈值与指标确认一致。

下面比较“复杂参数下新增组增量”减去“原参数下新增组增量”。正值表示所选复杂参数方案更能利用该组，但不保证该组在复杂模型下已为正，更不保证整个模型超过效用基线。

'''+table(interaction,[('group_name','组',None),('old_increment','原参数增量 / 万笔 ($)',signed),
                       ('complex_increment','复杂参数增量 / 万笔 ($)',signed),('increment_difference','两者差分 / 万笔 ($)',signed),
                       ('old_accepted','原参数通过',lambda x:'是' if x else '否'),('complex_accepted','复杂参数通过',lambda x:'是' if x else '否')])+'\n\n'+image(5)+'''
这些参数方案可能同时改变树数、深度与正则化，不能将差分单独解释成容量的因果作用。开发窗口反复参与选择，也不能据此宣称统计显著或独立泛化成立。
''')
    branches=[]
    for family,s in selections.items():
        for b in s['branches']:
            branches.append({'family':family,'base':b['base'],'accepted':', '.join(b['accepted_individual_groups']) or '无',
                             'eligible':', '.join(b['eligible_runs']),'final':b['final_run']})
    parts.append('''## 6. 合并规则与最终研究候选

同分支至少两个单组通过，才训练一次它们的合并版，并按相同规则相对十项父版验证。最终从基线、通过的单组和通过的合并版中按开发效用排名选择；合并版不会自动胜出。CatBoost 还将原参数和复杂参数的分支结果一起比较。

'''+table(pd.DataFrame(branches),[('family','算法',None),('base','十项父版',None),('accepted','单组通过',None),('eligible','合格候选',None),('final','分支最终',None)])+'''

最终保留输入：

'''+ '\n'.join(f'- **{family} / {rid}**：`{", ".join(runs[rid]["features"])}`。' for family,rid in final_ids.items())+'''

主要候选的开发窗口表现：

'''+table(dev_comparison,[('run_id','候选',None),('fold','折',None),('utility','效用 ($)',n),('delta_vs_always_normal','相对放行 ($)',n),
                         ('precision','Precision',pct),('recall','Recall',pct),('FP','误报',n),('FN','漏报',n),('average_precision','AP',lambda x:n(x,4)),('roc_auc','ROC-AUC',lambda x:n(x,4))])+'''

所有候选与阈值选择期指标见 [完整指标](tables/metrics.csv)。上述开发结果均参与选择，不是独立测试成绩。相同客户与相邻事件会聚集，三折拟合样本也重叠，没有把它们当独立样本计算显著性。
''')

    parts.append('''## 7. 六月复用诊断

两种算法的参数、单组/合并决定和最终候选都写入选择文件后，才写入 [总选择锁](selection_lock.json)，随后生成本轮六月分数。各诊断文件绑定总锁散列；没有根据六月表现重新挑选候选。六月沿用截至三月拟合的D3模型与各自四月阈值。

'''+table(june.reset_index(),[('family','算法',None),('run_id','候选',None),('utility','效用 ($)',n),('delta_vs_always_normal','相对放行 ($)',n),
                             ('precision','Precision',pct),('recall','Recall',pct),('false_positive_rate','误报率',pct),('FP','误报',n),('FN','漏报',n)])+'\n\n'+image(6)+image(7)+'''
六月87,304笔正常交易全部属于 personal_care 或 travel，这两类在D3截至三月的拟合期没有正常样本。这种类别和时间支持变化在前两版已明确；本轮没有通过把五月加入拟合来改变既定比较口径。更多参数或变量能否缓解这一现象，只由保存结果判断。

当前结果不代表按全量训练数据拟合后的真实Kaggle表现。官方测试没有标签，本轮没有预测或提交；已被反复观察的六月也不能重新包装为独立测试。
''')
    capacity=pd.read_csv(folder/'tables/realized_capacity_D3.csv')
    parts.append('''## 8. 实际容量、特征口径与重要性

下面直接统计保存的D3模型实际树形；同样深度/树数上限不能视为相同实际容量。CatBoost的实际深度由每棵对称树叶数推得，XGBoost从保存节点的父子关系重建深度。所有候选D3都拟合到三月，旧模型拟合耗时记零表示复用。

'''+table(capacity,[('run_id','候选',None),('feature_count','输入数',n),('tree_count','实际树数',n),('total_leaves','存储叶子总数',n),
                    ('nonempty_leaves','非空叶子数',n),('nonempty_fraction','非空比例',pct),
                    ('mean_leaves_per_tree','每树平均叶数',lambda x:n(x,1)),('maximum_realized_depth','最大实际深度',n),
                    ('mean_tree_depth','平均树深',lambda x:n(x,2)),('model_megabytes','模型MiB',lambda x:n(x,2)),('fit_seconds','D3拟合秒数',lambda x:n(x,1))])+'''

CatBoost按保存的训练叶权重是否大于零判断非空，XGBoost按叶节点Cover是否大于零判断非空；两者权重定义不同，只比较是否有训练支持。存储叶子可能包含空叶，叶数也不等于有效自由度。这些统计和模型体积仅用于解释结构与成本，不参与重新选择，也不能代替时间窗口效用。

历史特征直接复用不可变的v2缓存：按静态客户键分组，严格使用当前时间之前的无标签交易，排除全部同秒事件；窗口内仅使用此前已发生的交易更新历史。原始卡号、姓名、街道、生日、客户键、绝对日期和目标标签均不进入模型。年龄由交易时点计算，邮编为五位字符串。

原历史核验包含36个直接逐笔重算样本及682,991行截断前缀一致性。缓存与历史源码散列本轮再次冻结，未来追加不会改变过去特征。XGBoost分类词表另有未知类别、顺序一致性、数值缺失和序列化冒烟检查，每折未知类别数量保存在折结果中。

'''+image(8)+'''
CatBoost重要性为PredictionValuesChange，XGBoost为归一化gain；两者数值定义不同，不横向比较大小。高度相关变量可能相互替代或分担重要性。职业、城市、邮编还可能代理模拟客户身份，其重要性不等于业务因果关系。
''')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    parts.append(f'''## 9. 保存与核验

共保存 **{result['total_checkpoints']}** 个候选/折模型，其中 **{result['new_model_fits']}** 次新拟合、**{result['reused_model_fits']}** 个旧模型复用。主运行 **{len(result['checks'])}** 项一致性检查通过。独立核验程序另检查原始标签对齐、全部保存决策、并列分数阈值最优性、仅拟合期词表、序列化评分、参数排名、变量与合并规则、选择锁与六月先后顺序；记录见 [独立核验结果](qa/independent_validation.json)。

运行环境：Python {result['runtime']['python']}、CatBoost {result['runtime']['catboost']}、XGBoost {result['runtime']['xgboost']}、pandas {result['runtime']['pandas']}、numpy {result['runtime']['numpy']}。使用已有Anaconda及项目内CatBoost依赖，没有修改全局环境。

主要文件：[复现说明](README.md)、[冻结协议](protocol.json)、[输入与代码散列](manifest.json)、[完整指标](tables/metrics.csv)、[变量检验](tables/feature_groups.csv)、[容量交互](tables/capacity_interactions.csv)、[最终对比](tables/final_comparison.csv)、[保存模型与预测](runs/)、[核验程序](verify_comparison.py)。

本轮没有在看到结果后增加候选、改权重、改时间切分或搜索已拒绝组的内部组合。结论只覆盖预定容量范围和变量组；核验通过表示计算一致，不表示模型效用或泛化通过。
''')
    text='\n\n'.join(parts);(folder/'report_zh.md').write_text(text,encoding='utf-8')
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    for f in figs:
        encoded=base64.b64encode((folder/f['file']).read_bytes()).decode('ascii')
        body=body.replace(f'src="{f["file"]}"',f'src="data:image/png;base64,{encoded}" loading="lazy"')
    css='''body{margin:0;background:#f1f5f7;color:#263847;font:16px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1250px;margin:24px auto;padding:35px 45px;background:white;border:1px solid #dbe4ea;border-radius:8px}h1{font-size:29px}h2{margin-top:40px;font-size:22px;color:#245775;border-bottom:1px solid #dce5ec;padding-bottom:8px}table{display:block;overflow-x:auto;border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}td,th{padding:8px 10px;border-bottom:1px solid #e3e9ed;text-align:left;vertical-align:top}th{white-space:nowrap;background:#e9f0f4}tr:nth-child(even){background:#f8fafb}img{max-width:100%;height:auto;display:block;margin:24px auto 10px}code{background:#edf2f5;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}a{color:#245f86}li{margin:8px 0}strong{color:#204b68}@media(max-width:760px){main{margin:0;padding:18px 14px}body{font-size:15px}}'''
    (folder/'report_zh.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定十项输入：模型容量与变量重试 v3</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>',encoding='utf-8')
