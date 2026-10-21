"""Render the frozen experiment's computed results; does not fit or select models."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd


def render_report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import markdown
    from sklearn.metrics import precision_recall_curve

    folder = Path(folder)
    read = lambda name: pd.read_csv(folder/'tables'/f'{name}.csv')
    results = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    metrics, thresholds, slices = read('metrics'), read('thresholds'), read('slices')
    selected = metrics.loc[(metrics.policy == 'selected') & (metrics.role != 'threshold_selection_in_sample')].copy()
    final = selected.loc[selected.window == 'D3_June'].copy()
    final_predictions = pd.read_csv(folder/'predictions/D3_June.csv.gz')
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [fm.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 165,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15})
    colors = {'Logistic': '#316a9a', 'CatBoost': '#c55345'}
    figures = []

    def finish(fig, name, title, caption):
        fig.suptitle(title, x=.025, ha='left', fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .89])
        fig.savefig(folder/'figures'/f'{name}.png', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append({'name': name, 'file': f'figures/{name}.png', 'title': title, 'caption': caption})

    windows = ['D1_NovDec', 'D2_FebMar', 'D3_May', 'D3_June']
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(windows))
    for i, model in enumerate(colors):
        v = selected.loc[selected.model == model].set_index('window').reindex(windows)
        ax.bar(x+(i-.5)*.25, v.utility_per_10000, width=.25, color=colors[model], label=model)
    base = selected.drop_duplicates('window').set_index('window').reindex(windows)
    ax.plot(x, base.always_normal_utility/base.n*10000, 'o--', color='#697886', label='全部放行')
    ax.axhline(0, color='#222', lw=.8)
    ax.set_xticks(x, ['2024/11–12', '2025/02–03', '2025/05', '2025/06 验收'])
    ax.set_ylabel('每万笔交易效用 ($)'); ax.legend(frameon=False)
    finish(fig, '01_temporal_utility', '固定阈值向后评估：核心特征的跨时间表现', '每个模型、每个折的阈值仅由此前独立阈值窗口选择。负效用如实保留，参照线为全部放行。')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, model in zip(axes, colors):
        row = final.loc[final.model == model].iloc[0]
        counts = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
        im = ax.imshow(np.log1p(counts), cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{counts[i,j]:,}', ha='center', va='center', fontsize=16,
                        color='white' if np.log1p(counts[i,j]) > np.log1p(counts).max()*.65 else '#222')
        ax.set_xticks([0, 1], ['放行', '拦截']); ax.set_yticks([0, 1], ['实际正常', '实际欺诈'])
        ax.set_title(model); ax.grid(False)
    finish(fig, '02_june_confusion', '六月验收混淆矩阵', '颜色采用 log1p(计数) 以显示稀有正例，文字为原始计数。预测均使用四月锁定阈值。')

    curves = read('threshold_curves').query("fold == 'D3'")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, model in zip(axes, colors):
        curve = curves.loc[(curves.model == model) & (curves.threshold > 0) & (curves.threshold <= 1)].sort_values('threshold')
        choice = thresholds.loc[(thresholds.fold == 'D3') & (thresholds.model == model)].iloc[0]
        ax.plot(curve.threshold, curve.utility, color=colors[model])
        ax.axvline(choice.threshold, color=colors[model], ls='--', label=f'锁定阈值 {choice.threshold:.5g}')
        ax.axvline(5/155, color='#697886', ls=':', label='理论参考 3.226%')
        ax.axhline(curve.always_normal_utility.iloc[0], color='#697886', lw=.8, label='全部放行')
        ax.set_xscale('log'); ax.set_xlabel('分数阈值'); ax.set_ylabel('四月阈值窗口效用 ($)'); ax.set_title(model); ax.legend(fontsize=8)
    finish(fig, '03_april_thresholds', '阈值只在四月选择，不使用五月或六月挑选', '曲线为展示用网格；实际选择扫描全部不同分数并整体处理并列分数。四月最高效用属于选择样本内结果。')

    categories = slices.loc[(slices.window == 'D3_June') & (slices.slice_feature == 'category')]
    order = categories.groupby('slice_value').FP.sum().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    support = read('fit_category_support').query("fold == 'D3'").set_index('category')
    for ax, model in zip(axes, colors):
        v = categories.loc[categories.model == model].set_index('slice_value').reindex(order)
        ax.barh(np.arange(len(v)), v.FP, color=colors[model])
        labels = [c+(' *' if support.loc[c, 'fit_normal'] == 0 else '') for c in order]
        ax.set_yticks(range(len(v)), labels); ax.invert_yaxis(); ax.set_xlabel('六月误报数'); ax.set_title(model)
    finish(fig, '04_category_errors', '六月误报按商户类别分解', '* 表示该类别在本轮拟合期没有正常样本；此标记仅用于诊断，不进入模型。')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in colors:
        y, p = final_predictions.is_fraud, final_predictions['p_'+model]
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(recall, precision, color=colors[model], label=model)
        row = final.loc[final.model == model].iloc[0]
        axes[0].scatter([row.recall], [row.precision], color=colors[model], s=40)
        xx = np.arange(3)
        components = np.array([row.true_positive_gain, row.false_negative_cost, row.false_positive_cost])
        axes[1].bar(xx+(-.18 if model == 'Logistic' else .18), components, width=.36, color=colors[model], label=model)
    axes[0].axhline(final_predictions.is_fraud.mean(), color='#697886', ls=':')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision'); axes[0].set_ylim(0, 1.03); axes[0].legend()
    axes[1].set_xticks(range(3), ['正确拦截收益', '漏报损失', '误报损失']); axes[1].set_ylabel('效用分项 ($)'); axes[1].legend()
    finish(fig, '05_ranking_and_utility', '排序表现与最终效用需要分别判断', 'PR 曲线是六月诊断；圆点为此前锁定阈值，未用该曲线重新选阈值。')

    response = read('amount_response')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = {'gas_transport': '#427b9c', 'grocery_pos': '#b48132', 'shopping_net': '#9a4a5c'}
    for ax, model in zip(axes, colors):
        dat = response.loc[(response.model == model) & (response.hour == 23)]
        for category, group in dat.groupby('category'):
            ax.plot(group.amt, group.score, color=palette[category], label=category)
        ax.set_xscale('log'); ax.set_xlabel('金额 ($)，小时固定为 23'); ax.set_ylabel('模型欺诈分数'); ax.set_title(model); ax.legend(fontsize=8)
    finish(fig, '06_amount_response', '相同核心信息下的金额响应形状', '预先指定的三类条件场景，仅说明模型形状；不代表因果关系或该场景在每个时期均有充分正常样本。')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cal = read('calibration_diagnostic')
    for ax, role, title in zip(axes, ['threshold_April', 'D3_June'], ['四月阈值窗口', '六月验收']):
        for model in colors:
            v = cal.loc[(cal.role == role) & (cal.model == model)].sort_values('mean_score')
            ax.plot(v.mean_score, v.fraud_rate, 'o-', color=colors[model], label=model)
        ax.plot([0, 1], [0, 1], '--', color='#697886'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('分箱平均模型分数'); ax.set_ylabel('分箱实际欺诈率'); ax.set_title(title); ax.legend(fontsize=8)
    finish(fig, '07_probability_diagnostic', '分数与真实概率的对应关系并未被保证', '固定分箱完整计数见 calibration_diagnostic.csv。本轮没有校准器，未根据六月曲线修改概率或阈值。')

    loss = read('catboost_training_loss')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for fold, data in loss.groupby('fold'):
        ax.plot(data.iteration, data.train_logloss, label=fold)
    ax.set_yscale('log'); ax.set_xlabel('树数量'); ax.set_ylabel('拟合期 Logloss（对数）'); ax.legend()
    finish(fig, '08_training_loss', 'CatBoost 拟合过程', '500 棵树在拟合前固定，未使用任何阈值或评估窗口进行早停；训练损失下降不能证明向后泛化。')

    def number(x, digits=0):
        return '未定义' if pd.isna(x) else f'{x:,.{digits}f}'

    def percent(x):
        return '未定义' if pd.isna(x) else f'{100*x:.2f}%'

    def mdtable(frame, columns):
        out = ['| '+' | '.join(v[1] for v in columns)+' |', '| '+' | '.join('---' for _ in columns)+' |']
        for _, row in frame.iterrows():
            out.append('| '+' | '.join((f(row[k]) if f else str(row[k])).replace('|', '/') for k, _, f in columns)+' |')
        return '\n'.join(out)

    def image(index):
        item = figures[index-1]
        return f'![{item["title"]}]({item["file"]})\n\n{item["caption"]}\n'

    baseline = -100*int(final_predictions.is_fraud.sum())
    summaries = []
    for model in colors:
        f = final.loc[final.model == model].iloc[0]
        dev = selected.loc[(selected.model == model) & (selected.window != 'D3_June')]
        summaries.append(f'- **{model}**：六月效用 **${f.utility:,.0f}**，相对全部放行变化 **${f.delta_vs_always_normal:+,.0f}**；Recall **{percent(f.recall)}**、Precision **{percent(f.precision)}**、误报 **{f.FP:,.0f}** 笔。此前三个评估窗口中，{int(dev.delta_vs_always_normal.gt(0).sum())}/3 个优于全部放行。')
    no_support = slices.loc[(slices.window == 'D3_June') & (slices.slice_feature == 'support_group')]
    support_table = read('fit_category_support').query("fold == 'D3'")
    threshold_final = thresholds.query("fold == 'D3'")
    counts = read('splits')
    default_final = metrics.loc[metrics.window == 'D3_June']
    negative_windows = int(selected.delta_vs_always_normal.le(0).sum())
    verdict = ('本轮已建立可复现的核心特征参照，但存在向后窗口低于全部放行的情况，尚无稳定跨时间晋级证据。'
               if negative_windows else '本轮两个核心模型在预先指定窗口均优于全部放行，但这仍不是官方测试表现或扩展特征有效性的证据。')

    text = f'''# 核心特征建模 v1：Logistic Regression 与 CatBoost

实验日期：2026-10-21。仅使用 **金额 amt、商户类别 category、小时 hour**。本轮未加入客户身份、年龄、地理或客户历史，也未运行 XGBoost。官方测试文件没有进入训练、阈值选择或预测。

## 1. 结果与判断

{verdict}

六月验收共 **{len(final_predictions):,}** 笔，其中欺诈 **{int(final_predictions.is_fraud.sum()):,}** 笔。全部放行效用为 **${baseline:,}**。

{chr(10).join(summaries)}

模型输出是未经额外校准的分数。0.5 和理论 3.226% 阈值只作预先指定的参照；主结果使用此前独立窗口选择的阈值。所有负结果如实保存，看到五月/六月结果后没有更改模型、超参数、特征或阈值。

六月正常交易全部来自 `personal_care` 与 `travel`，二者在 D3 截至三月的拟合期都没有正常样本。这个评估暴露出核心模型在当前支持变化下的不稳定性，不能据此断言三项特征在完整训练数据拟合后必然无效，或某种模型算法普遍不适合本项目。

{image(1)}

## 2. 实验前固定的协议

协议文件为 [protocol.json](protocol.json)，SHA-256 为 `{results['protocol_sha256']}`。它在首次拟合前写入，运行结束确认未改变。

- 评分：`Utility = 50×TP − 100×FN − 5×FP`。全部放行损失为 `−100×欺诈数`；相对全部放行的增量为 `150×TP − 5×FP`。
- 全量记录，自然类别比例，无过采样、欠采样或类别权重；随机种子 7406。
- 特征仅为三列。`unix_time` 用于排序、划分与派生小时；绝对时间和 ID 不进入模型。目标列仅用于拟合及其所属评估阶段。
- Logistic：金额做 `log1p → 8 节点三次样条 → 标准化`，类别和小时做 one-hot，L2、C=1、lbfgs、最大 1200 次迭代。全部预处理只在拟合窗口估计；没有人工交互。它仍是对固定基函数的加性 Logistic 模型。
- CatBoost：CPU、500 棵树、深度 6、学习率 0.05、L2=3，类别与小时作为分类变量。输入按时间排序，`has_time=True`，`boosting_type=Plain`，分类计数使用 `SkipTest`。没有验证早停或超参数搜索。
- 两模型接收相同原始信息，允许各自合适的表示。CatBoost 可以学习交互，Logistic 保持加性，不能把二者差异单独归因为某一个机制。

模型接口依据 [LogisticRegression 文档](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html)、[SplineTransformer 文档](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.SplineTransformer.html)、[CatBoost 参数文档](https://catboost.ai/docs/en/references/training-parameters/common)。

## 3. 时间划分与阈值选择

{mdtable(counts, [('fold','折',None),('role','用途',None),('window','窗口',None),('time_min','起始',None),('time_max','结束',None),('n','交易数',number),('fraud','欺诈数',number)])}

D1：拟合到 2024/09，10 月选阈值，11–12 月评估。D2：拟合到 2024/12，2025/01 选阈值，02–03 月评估。D3：拟合到 2025/03，04 月选阈值，05 月作开发回测，06 月作保留验收。

**D3 在四月锁定后不再重训。** 五月和六月均使用该检查点及四月阈值。没有把四、五月重新加入拟合后继续沿用另一个模型的验证结论。未来若进行全量重训，应作为单独版本重新记录。

每个模型在各自阈值窗口扫描全部不同分数，按并列分数整体决定拦截，包含全部放行和全部拦截；同效用时优先更少拦截。决定文件在生成后续窗口预测前写入。阈值窗口最高效用是选择样本内结果，不能作为泛化表现。

{mdtable(thresholds, [('fold','折',None),('model','模型',None),('threshold','锁定阈值',lambda x:number(x,8)),('threshold_window_utility','阈值窗口效用 ($)',number),('flags','拦截数',number)])}

{image(3)}

## 4. 固定阈值的向后评估

{mdtable(selected, [('window','窗口',None),('model','模型',None),('utility','效用 ($)',number),('delta_vs_always_normal','相对全部放行 ($)',number),('precision','Precision',percent),('recall','Recall',percent),('FP','误报数',number),('FN','漏报数',number),('average_precision','AP',lambda x:number(x,4))])}

AP 指 Average Precision，不与梯形积分 PR-AUC 混用。ROC-AUC、Logloss、Brier、每万笔正常交易误报数和所有混淆计数均在 `tables/metrics.csv`。月、周、类别、小时、金额区间及拟合支持量分组保存在 `tables/slices.csv`。不同窗口交易数与基准率不同，图中使用每万笔效用辅助比较。

{image(2)}
{image(5)}

六月三个预先指定阈值策略：

{mdtable(default_final, [('model','模型',None),('policy','阈值策略',None),('threshold','阈值',lambda x:number(x,8)),('utility','效用 ($)',number),('precision','Precision',percent),('recall','Recall',percent),('FP','误报数',number)])}

这张表用于理解阈值迁移，不能从六月结果中选一个新阈值替换主结果。模型排序能力、概率校准与决策效用分别报告。

## 5. 类别时间分块如何影响误报

EDA 已发现类别在时间上集中。拟合期没有正常样本的类别，可能在后续窗口出现大量正常交易。我们只使用拟合期标签检查支持量，没有把未来正常类别活跃区间输入模型。

D3 拟合期类别支持：

{mdtable(support_table, [('category','类别',None),('fit_n','拟合交易数',number),('fit_normal','拟合正常数',number),('fit_fraud','拟合欺诈数',number)])}

六月按拟合期是否存在正常样本划分：

{mdtable(no_support, [('model','模型',None),('slice_value','拟合支持组',None),('n','六月交易数',number),('utility','效用 ($)',number),('TP','TP',number),('FP','FP',number),('FN','FN',number)])}

{image(4)}

误报与零正常支持类别集中出现，提供了与时间分块一致的诊断证据；本轮没有删除类别后的重训，因此尚不能把全部性能下降因果归于 category。下一步需要使用相同时间协议做特征组对照，不能在看到验收结果后直接利用未来类别标签生成规则。

六月 Logistic 的 ROC-AUC 为 {float(final.loc[final.model == 'Logistic', 'roc_auc'].iloc[0]):.4f}，CatBoost 为 {float(final.loc[final.model == 'CatBoost', 'roc_auc'].iloc[0]):.4f}；这说明当前窗口的排序表现也很弱，不能把全部问题归结为默认 0.5 阈值。参考阈值结果只是诊断，不用于修改主决定。

## 6. 模型响应、训练与概率诊断

{image(6)}
{image(7)}
{image(8)}

样条金额变换允许 Logistic 表达非单调金额效应；CatBoost 使用相同输入学习更灵活的关系。当前可视化说明拟合模型的形状，不证明这些关系跨时间稳定或具有因果含义。

## 7. 验证边界与复现

- 本轮比较核心三特征的组合表现，未完成逐特征删除实验，不能据此确认每个特征的独立增益。
- 全部训练月份在先前 EDA 中已进行描述性查看。因此六月是未用于本轮拟合/阈值选择的验收窗口，不能称为从未观察的盲测。它在本轮已完成评估，后续改动应记录该使用历史。
- 三组回测是扩展拟合窗口，后期模型可以使用当时已经可获知的较早标签。各折超参数保持不变；评估记录本身互不重叠，但不同折的拟合样本重叠，不把窗口结果当独立重复实验。
- 同客户欺诈可能跨天聚集。本轮给出按周切片，不把逐笔记录当独立样本计算显著性或声称置信保证。
- 正式测试标签、测试效用和比赛排名未知。本轮没有官方测试预测、提交或 AI-Assistant 比较。
- 输入及原 EDA 文件 SHA-256 未变；各窗口时间严格分离；样条、缩放、编码只在各自拟合期学习；所有预测概率有限且在 [0,1]；保存模型重载后概率一致；保存预测重算效用一致。

本轮 **{len(results['checks'])}/{len(results['checks'])}** 项核验通过。保存了 6 个模型文件、3 组阈值决定、7 份分窗口预测以及完整指标。决定文件包含模型与协议散列；详细复现见 [README.md](README.md)。

另提供 [verify_core.py](verify_core.py)：从原始 ID 对齐全部已保存记录，独立重算混淆计数与成本，并用保存模型对六月全部原始核心输入重新评分。结果保存在 [qa_validation.json](qa_validation.json)，不重新拟合。

运行环境：Python {results['runtime']['python']}、scikit-learn {results['runtime']['sklearn']}、CatBoost {results['runtime']['catboost']}、pandas {results['runtime']['pandas']}、numpy {results['runtime']['numpy']}。拟合和评估耗时 {results['elapsed_seconds']:.1f} 秒。CatBoost 包安装在本项目 `.deps`，其余依赖使用现有环境，全局环境未修改。
'''
    (folder/'report_zh.md').write_text(text, encoding='utf-8')
    body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    for fig in figures:
        encoded = base64.b64encode((folder/fig['file']).read_bytes()).decode('ascii')
        body = body.replace(f'src="{fig["file"]}"', f'src="data:image/png;base64,{encoded}" loading="lazy"')
    style = '''body{margin:0;background:#f2f5f7;color:#243443;font:16px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}
    main{max-width:1180px;margin:24px auto;padding:38px 48px;background:white;border:1px solid #dce3e9;border-radius:8px}
    h1{font-size:29px;line-height:1.5}h2{margin-top:42px;font-size:22px;color:#244f73;border-bottom:1px solid #dce3e9;padding-bottom:8px}
    table{display:block;overflow-x:auto;border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #e3e8ed;vertical-align:top}
    th{background:#eaf0f5;white-space:nowrap}tr:nth-child(even){background:#f8fafb}img{max-width:100%;height:auto;display:block;margin:25px auto 10px}
    code{background:#eef2f5;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}a{color:#245e8b}li{margin:8px 0}strong{color:#183f61}
    @media(max-width:760px){main{margin:0;padding:20px 15px;border:0}body{font-size:15px}}
    '''
    output = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>核心特征建模 v1</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>'
    (folder/'report_zh.html').write_text(output, encoding='utf-8')
    (folder/'figure_index.json').write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT COMPLETE: 8 figures, report_zh.md and self-contained report_zh.html', flush=True)

