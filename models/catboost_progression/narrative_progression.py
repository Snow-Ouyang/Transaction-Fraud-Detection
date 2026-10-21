"""Chinese narrative built only from completed experiment records."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import pandas as pd


def write_narrative(folder, result, protocol, grid, progression, development, june, figures, markdown):
    folder = Path(folder)
    selected_id, final_id = result['selected_parameter_run'], result['final_retained_run']
    original_grid = grid.loc[grid.run_id == 'core_default'].iloc[0]
    selected_grid = grid.loc[grid.run_id == selected_id].iloc[0]
    final = june.loc[june.run_id == final_id].iloc[0]
    original = june.loc[june.run_id == 'core_default'].iloc[0]
    tuned = june.loc[june.run_id == selected_id].iloc[0]
    audit = json.loads((folder/'qa/feature_validation.json').read_text(encoding='utf-8'))
    features_table = pd.read_csv(folder/'tables/run_features.csv')
    params = result['selected_parameters']

    def number(x, places=0):
        return '未定义' if pd.isna(x) else f'{x:,.{places}f}'

    def signed(x):
        return '未定义' if pd.isna(x) else f'{x:+,.2f}'

    def percent(x):
        return '未定义' if pd.isna(x) else f'{100*x:.2f}%'

    def mdtable(frame, columns):
        out = ['| '+' | '.join(v[1] for v in columns)+' |', '| '+' | '.join('---' for _ in columns)+' |']
        for _, row in frame.iterrows():
            out.append('| '+' | '.join((f(row[k]) if f else str(row[k])).replace('|', '/') for k,_,f in columns)+' |')
        return '\n'.join(out)

    def image(index):
        f = figures[index-1]
        return f'![{f["title"]}]({f["file"]})\n\n{f["caption"]}\n'

    retained = progression.loc[progression.retained]
    rejected = progression.loc[~progression.retained]
    retained_names = '、'.join(retained.group_name) if len(retained) else '无新增组'
    rejected_names = '、'.join(rejected.group_name) if len(rejected) else '无'
    delta_parameters = selected_grid.mean_delta_per_10000-original_grid.mean_delta_per_10000
    final_mean = result['final_development_mean_delta_per_10000']
    delta_features = final_mean-selected_grid.mean_delta_per_10000
    description = ('本轮调参和扩展带来小幅相对改善，但最终版本在三个开发窗口和六月复用诊断中均劣于全部放行；按当前时间验证，本轮模型未通过效用基线。'
                   if result['final_development_positive_windows'] == 0 and final.delta_vs_always_normal <= 0
                   else '开发期三个窗口均优于全部放行，六月复用诊断也优于该参照；可以保留为后续研究候选，但尚无全新独立测试确认。'
                   if result['final_development_positive_windows'] == 3 and final.delta_vs_always_normal > 0
                   else '当前保留的是按预先固定规则选出的研究版本；跨窗口稳定性或六月复用诊断仍存在限制，不能直接视为验证通过。')
    parts = []
    parts.append(f'''# CatBoost v2：小范围调参与逐组扩展

实验日期：2026-10-21。原始 EDA、核心模型 v1 与输入 CSV 保持不变。本轮只研究 CatBoost，没有重训 Logistic，没有运行 XGBoost，也没有产生官方测试预测或提交。

## 1. 结果与适用边界

{description}

- 小网格选中 **深度 {params['depth']}、L2={params['l2_leaf_reg']:g}**；500 棵树、学习率 0.05 不变。相对原始参数，三个开发窗口等权平均效用变化为 **${delta_parameters:+,.2f}/万笔**。
- 按序保留的扩展组：**{retained_names}**。本轮未保留：**{rejected_names}**。未保留只代表当前父版、固定参数与三个开发窗口下未满足规则，不证明该组在所有模型中无用。
- 最终研究版本：`{final_id}`，共 **{len(result['final_features'])}** 项输入。相对调优后核心版，开发期平均效用再变化 **${delta_features:+,.2f}/万笔**。
- 最终开发版本在 **{result['final_development_positive_windows']}/3** 个窗口优于全部放行。开发结果参与选择，存在乐观偏差。
- 六月复用诊断效用 **${final.utility:,.0f}**，全部放行参照为 **${final.always_normal_utility:,.0f}**；Recall **{percent(final.recall)}**，Precision **{percent(final.precision)}**，正常交易误报率 **{percent(final.false_positive_rate)}**，误报 **{int(final.FP):,}** 笔。原始核心效用 **${original.utility:,.0f}**，本轮选定核心参数效用 **${tuned.utility:,.0f}**。

**六月在 v1 中已被观察和评估，本轮没有全新的独立测试集。** 本轮先锁定所有参数、特征保留决定，再生成六月诊断；没有依据六月结果重新挑选候选。真实 Kaggle 测试表现未知。

## 2. 冻结的实验设计

协议见 [protocol.json](protocol.json)，初始文件散列为 `{result['signature']['protocol_sha256']}`。参数候选、扩展组、顺序、阈值与保留规则均在新拟合前固定。

| 折 | 拟合期末（不含） | 独立阈值窗口 | 开发选择窗口 |
| --- | --- | --- | --- |
| D1 | 2024-10-01 | 2024/10 | 2024/11–12 |
| D2 | 2025-01-01 | 2025/01 | 2025/02–03 |
| D3 | 2025-04-01 | 2025/04 | 2025/05 |

六月诊断沿用 D3 截至三月的模型与各自四月阈值，不为六月重训。每个候选都使用相同的三个时间窗口、全量记录、自然类别比例、种子 7406；无类别权重、无重采样。没有为扩展组继续调参。

评分为 `U = 50×TP − 100×FN − 5×FP`。相对全部放行的增量为 `ΔU = 150×TP − 5×FP`。对每个窗口先除以交易数再乘 10,000，三个窗口等权平均；这样 D1 较长的窗口不会仅凭行数支配选择。

每个候选、每折都只在此前阈值窗口扫描全部不同分数，整体处理并列分数，包含全部放行和全部拦截，同效用优先更少拦截。阈值决定保存后才生成该折开发预测。未另行做概率校准；分数不保证是真实概率，理论 3.226% 阈值仅作预先指定参考。

选择后的开发表现用于描述研究结果，不能当作独立泛化估计。三折拟合样本重叠、同客户事件聚集，也不把它们当成独立实验计算显著性。
''')

    parts.append('''## 3. 小范围参数比较

只比较 `depth ∈ {4,6}` 与 `l2_leaf_reg ∈ {3,10}`。原始深6/L2=3模型直接复用，其他三组重新拟合。全部保持 500 棵树、学习率 0.05、CPU 四线程、Logloss、Plain boosting、按时间输入与 `has_time=True`，分类计数使用 `SkipTest`，无验证早停。

深度与 L2 的选择参考 [CatBoost 调参文档](https://catboost.ai/docs/en/concepts/parameter-tuning) 与 [参数定义](https://catboost.ai/docs/en/references/training-parameters/common)。本轮是有限网格比较，不能声称全局最优。

'''+mdtable(grid, [('run_id','候选',None),('depth','深度',number),('l2_leaf_reg','L2',number),
                   ('mean_delta_per_10000','相对放行的平均增量 / 万笔 ($)',lambda x:number(x,2)),
                   ('worst_delta_per_10000','最差窗口增量 / 万笔 ($)',lambda x:number(x,2)),
                   ('windows_above_always_normal','优于放行窗口数',number)])+'\n\n'+image(1)+f'''
选择先最大化平均增量；完全并列时依次比较最差窗口、较浅深度、较强 L2。选定 `{selected_id}` 后，参数文件 [parameter_decision.json](parameter_decision.json) 在所有扩展试验前写入。

调参比较的是同一组三项核心信息，不能把后续变量带来的改善计入参数收益。即使选中参数的开发平均效用最高，也不意味着它已经超过全部放行，完整绝对值和增量都保留。
''')

    view = progression.copy()
    view['retained_zh'] = view.retained.map({True:'保留',False:'不保留'})
    parts.append('''## 4. 按序增加变量与保留路径

每次将一组加入**当前保留父版**。平均配对效用变化为正、且至少 2/3 开发窗口改善，才让该组进入后续版本。没有通过的组被退回；后面的试验不会自动携带它。

'''+mdtable(view, [('group_name','新增组',None),('parent_run','父版',None),('candidate_feature_count','候选输入数',number),
                   ('mean_paired_delta_per_10000','平均配对变化 / 万笔 ($)',signed),('improved_windows','改善窗口数',number),
                   ('retained_zh','决定',None),('retained_run_after_step','决定后版本',None)])+'\n\n'+image(2)+'''
逐窗口变化：

'''+mdtable(view, [('group_name','新增组',None),('D1_change_per_10000','D1 变化 / 万笔 ($)',signed),
                   ('D2_change_per_10000','D2 变化 / 万笔 ($)',signed),('D3_change_per_10000','D3 变化 / 万笔 ($)',signed)])+f'''

最终保留输入：`{', '.join(result['final_features'])}`。

保留只是选定下一步研究分支。若父版仍低于参照，新增组即使改善父版，也不自动构成可用模型。没有额外搜索组内组合、改变组顺序或在看完六月后重试已拒绝的变量。
''')

    definitions = pd.DataFrame([
        ['amt','当前交易金额，美元','核心','保留非单调关系，不按金额改变评分成本'],
        ['category','当前商户类别','核心','存在明显类别×时间分块'],
        ['hour','由 Unix 时间解码的记录时钟小时','核心','不推断客户本地时区'],
        ['amount_to_past_mean','当前金额 / 严格此前个人金额均值','个人金额偏离','无历史保持缺失'],
        ['has_customer_history','严格此前是否已有交易','个人金额偏离','冷启动标记；模拟数据纯欺诈小客户可能影响关系'],
        ['age','交易日与出生日期差 / 365.2425 天','年龄与星期','连续近似年龄，不使用分析日年龄'],
        ['weekday','记录日期的星期','年龄与星期','作为分类变量'],
        ['past_count','严格此前累计交易笔数','其他历史','不包含当前及所有同秒交易'],
        ['past_mean_amt','严格此前累计金额均值','其他历史','不读取历史标签'],
        ['gap_seconds','距上一个不同时间戳的个人交易秒数','其他历史','第一笔缺失'],
        ['count_24h','此前 [t−24h,t) 的个人交易数','其他历史','开发窗口可读取此前无标签交易'],
        ['new_merchant','此前是否从未在当前商户交易','其他历史','同秒多笔首次事件均视为首次'],
        ['merchant','当前商户名称','商户','与 category 信息重叠且有时间结构'],
        ['job','职业类别','身份代理','可能记忆客户/模拟数据结构'],
        ['city','城市名','身份代理','城市名不唯一确定州'],
        ['zip','补齐至五位的字符串邮编','身份代理','字符串处理；不作为连续数字']
    ], columns=['feature','definition','group','caution'])
    definitions['in_final'] = definitions.feature.isin(result['final_features']).map({True:'是',False:'否'})
    definitions.to_csv(folder/'tables/feature_definitions.csv',index=False,encoding='utf-8-sig')
    parts.append('''## 5. 特征口径与可用性

'''+mdtable(definitions,[('feature','字段',None),('definition','定义',None),('group','组',None),('in_final','最终保留',None),('caution','边界',None)])+f'''

静态客户键由标准化姓名、生日、街道和邮编构造，仅用于内部历史分组；原始姓名、街道、生日、卡号和客户键本身不作为预测输入。没有加入性别、州、人口、经纬度或绝对日期。

历史函数只接收 ID、时间、金额、商户、客户键五个字段，物理上不接收标签。先按客户和时间聚合，再从累计量中减去整个当前时间戳的贡献，确保同秒互不读取。所有模型在每个拟合截止点之后，只使用当时已经发生的无标签事件更新历史。

独立检查：直接筛选此前事件逐笔重算 **{audit['spot_rows']}** 个样本；将数据截断到 2024-10-01 重算 **{audit['future_append_prefix_n']:,}** 行，其全部历史特征与完整数据前缀一致。未来追加不会改变过去的值。第一次事件历史均值缺失共 **{audit['first_history_missing_rows']:,}** 行。

与先前 EDA 口径一致，正常与欺诈的个人金额偏离中位数分别为 **{audit['normal_relative_amount_median']:.6f}** 与 **{audit['fraud_relative_amount_median']:.6f}**。这些分布用于口径核对，不作为六月选择依据。
''')

    parts.append('''## 6. 开发窗口的全部候选结果

'''+mdtable(development,[('run_id','候选',None),('fold','折',None),('utility','效用 ($)',number),
                         ('delta_vs_always_normal','相对放行 ($)',number),('precision','Precision',percent),
                         ('recall','Recall',percent),('FP','误报',number),('FN','漏报',number)])+'\n\n'+image(3)+'''
同一数据窗口内的候选差异可作配对比较。不同窗口的总效用受交易数和基准率影响，选参数及特征时统一使用每万笔效用变化。各候选阈值各自从此前窗口选择，模型与阈值作为同一决策系统比较。

这些开发窗口用于多个候选的选择，结果不是确认性证据。完整 AP、ROC-AUC、Logloss、Brier、混淆计数和阈值选择期指标见 `tables/metrics.csv`。
''')

    parts.append(f'''## 7. 选择锁定后才进行的六月复用诊断

选择锁定文件 [selection_lock.json](selection_lock.json) 的 SHA-256 为 `{result['selection_lock_sha256']}`。参数与五组保留决定全部固定之后，程序才首次生成本轮六月分数；每个六月诊断文件绑定该锁定散列。v1 对六月的已有观察无法撤销，因此本轮仍明确标为复用诊断。

'''+mdtable(june,[('run_id','候选',None),('utility','效用 ($)',number),('delta_vs_always_normal','相对放行 ($)',number),
                 ('precision','Precision',percent),('recall','Recall',percent),('FP','误报',number),('FN','漏报',number),
                 ('roc_auc','ROC-AUC',lambda x:number(x,4))])+'\n\n'+image(4)+image(5)+'''
未选中的参数组合不在六月重新竞争。新增组即使在六月表现较好，也不会推翻此前“不保留”的决定。不能把本图中的最好一根柱子改成最终版本。

'''+image(6)+image(8)+image(9)+'''
六月正常交易的类别支持变化已在 v1 中明确：87,304 笔正常交易全部属于 `personal_care` 与 `travel`，它们在 D3 截至三月的拟合数据中均无正常样本。本轮扩展仍未消除这些误报。这与已知类别和时间支持变化一致，切片本身不能单独证明因果。

本轮仍保持 D3 拟合截至三月、阈值截至四月的协议，以便比较增量；没有为了让六月分数更好而将五月重训进去。下一轮宜优先检查 `category` 消融与时间支持敏感性。更贴近正式测试训练覆盖的验证设计，应另立版本，而不能覆盖本轮结果。
''')

    attribution = [{'component':'参数调优','development_change_per_10000':delta_parameters}]
    attribution.extend({'component':row.group_name,'development_change_per_10000':row.mean_paired_delta_per_10000 if row.retained else 0.}
                       for _,row in progression.iterrows())
    attribution = pd.DataFrame(attribution)
    attribution.to_csv(folder/'tables/retained_path_attribution.csv',index=False,encoding='utf-8-sig')
    assert abs(attribution.development_change_per_10000.sum()-(final_mean-original_grid.mean_delta_per_10000)) < 1e-6
    parts.append('''## 8. 增量归属与模型内部重要性

下表将保留路径的平均效用变化逐步相加。未保留组贡献记为零；各步之和与最终版相对原始核心版的开发平均差一致。

'''+mdtable(attribution,[('component','步骤',None),('development_change_per_10000','保留路径变化 / 万笔 ($)',signed)])+'\n\n'+image(7)+'''
特征重要性采用拟合模型的 PredictionValuesChange，不能替代前面的分组增量检验。金额与个人金额偏离、客户均值等相关字段可能相互替代；职业、城市与邮编还可能代理客户身份。样本内高重要性不构成因果影响或真实业务的稳定风险。
''')

    parts.append(f'''## 9. 核验、保存文件与后续边界

完成 **{result['new_model_fits']}** 次新拟合，复用 **{result['reused_model_fits']}** 个原始检查点，共 **{result['total_model_checkpoints']}** 个按候选/折保存的模型。训练、预测与选择总耗时 **{result['elapsed_seconds']:.1f}** 秒（不含报告渲染）。

主运行 **{len(result['checks'])}/{len(result['checks'])}** 项核验通过：冻结协议与输入散列、原版复现、模型重载、特征时序检查、选择锁定在六月前及原文件未变。另提供 [verify_progression.py](verify_progression.py) 对保存预测、原始标签对齐、成本计数、参数排名、特征保留路径和最终模型评分做独立重算；结果见 [独立核验记录](qa/independent_validation.json)。核验通过表示计算和记录一致，不表示模型效用达到基线。

环境：Python {result['runtime']['python']}、CatBoost {result['runtime']['catboost']}、pandas {result['runtime']['pandas']}、numpy {result['runtime']['numpy']}。复用本项目 v1 的 CatBoost 依赖目录，未修改全局环境。

主要文件：

- [README.md](README.md)：执行、恢复和复现方法。
- [parameter_decision.json](parameter_decision.json)：小网格排名与参数选择。
- [feature_decisions.json](feature_decisions.json)：逐组父版、配对变化和保留决定。
- [selection_lock.json](selection_lock.json)：六月复用诊断前的最终选择。
- [tables/](tables/)：完整指标、增量表和特征定义。
- [runs/](runs/)：各折模型、独立阈值决定、保存预测、特征重要性及散列。
- [qa/feature_validation.json](qa/feature_validation.json)：无前视特征核验。

本轮只完成有限网格与预定五组变量，没有搜索组内组合、重新排列组顺序、增加新的特征类别或调类别权重。若继续研究，应保留开发期已被用于选择、六月已被多次观察的历史；不能把后续改进的六月结果重新包装为独立测试通过。
''')

    text = '\n\n'.join(parts)
    (folder/'report_zh.md').write_text(text,encoding='utf-8')
    body = markdown.markdown(text,extensions=['tables','fenced_code'])
    for figure in figures:
        b64=base64.b64encode((folder/figure['file']).read_bytes()).decode('ascii')
        body=body.replace(f'src="{figure["file"]}"',f'src="data:image/png;base64,{b64}" loading="lazy"')
    style='''body{margin:0;background:#f2f5f7;color:#243443;font:16px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}
    main{max-width:1220px;margin:24px auto;padding:38px 48px;background:white;border:1px solid #dce3e9;border-radius:8px}
    h1{font-size:29px;line-height:1.5}h2{margin-top:42px;font-size:22px;color:#244f73;border-bottom:1px solid #dce3e9;padding-bottom:8px}
    table{display:block;overflow-x:auto;border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #e3e8ed;vertical-align:top}
    th{background:#eaf0f5;white-space:nowrap}tr:nth-child(even){background:#f8fafb}img{max-width:100%;height:auto;display:block;margin:25px auto 10px}
    code{background:#eef2f5;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}a{color:#245e8b}li{margin:8px 0}strong{color:#183f61}
    @media(max-width:760px){main{margin:0;padding:20px 15px;border:0}body{font-size:15px}}
    '''
    out='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CatBoost 调参与逐组扩展 v2</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>'
    (folder/'report_zh.html').write_text(out,encoding='utf-8')
