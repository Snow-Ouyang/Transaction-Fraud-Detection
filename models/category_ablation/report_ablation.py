"""Build editable Chinese report and a self-contained offline HTML."""
from pathlib import Path
import base64
import json
import sys
import pandas as pd
import markdown

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_ablation import LABELS,ORDER,WINDOWS,PARENTS


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def table(rows,cols):return pd.DataFrame(rows,columns=cols).to_markdown(index=False)
def money(n):return f'{n:,.0f}'
def pct(n):return f'{n:.2%}'


def main():
    p=read(HERE/'protocol.json');qa=read(HERE/'qa/independent_validation.json')
    assert qa['all_passed']
    m=pd.read_csv(HERE/'tables/metrics.csv');c=pd.read_csv(HERE/'tables/comparisons.csv')
    s=pd.read_csv(HERE/'tables/score_distributions.csv');cat=pd.read_csv(HERE/'tables/category_support_errors.csv')
    pairs=pd.read_csv(HERE/'tables/paired_changes.csv')
    main_rows=[];june_rows=[];threshold_rows=[];dev_rows=[];trade_rows=[];ranking_rows=[]
    for rid in ORDER:
        sub=m.loc[m.run_id.eq(rid)&m.role.ne('threshold')].set_index('window')
        main_rows.append([LABELS[rid]]+[pct(sub.loc[w,'false_positive_rate']) for w in WINDOWS])
        j=sub.loc['June'];dev=sub.loc[['D1','D2','D3']]
        june_rows.append([LABELS[rid],int(j.TP),int(j.FP),int(j.FN),pct(j.recall),pct(j.precision),money(j.utility),money(j.delta_vs_always_normal)])
        ranking_rows.append([LABELS[rid],f'{j.roc_auc:.4f}',f'{j.average_precision:.4f}'])
        dev_rows.append([LABELS[rid],money(dev.delta_per_10000.mean()),int(dev.delta_vs_always_normal.gt(0).sum())])
        score=s.loc[s.run_id.eq(rid)&s.is_fraud.eq(0)&((s.window.eq('D3'))|s.window.eq('June'))].set_index(['window','role'])
        cal=m.loc[m.run_id.eq(rid)&m.window.eq('D3')&m.role.eq('threshold')].iloc[0]
        threshold_rows.append([LABELS[rid],f'{cal.threshold:.6f}',pct(cal.false_positive_rate),
            f"{score.loc[('D3','threshold'),'score_median']:.4f}",f"{score.loc[('D3','development'),'score_median']:.4f}",f"{score.loc[('June','diagnostic'),'score_median']:.4f}"])
    for rid,parent in PARENTS.items():
        z=c.loc[c.run_id.eq(rid)&c.window.eq('June')].iloc[0]
        trade_rows.append([LABELS[rid],LABELS[parent],int(z.delta_FP),int(z.delta_FN),money(z.delta_utility),
                          money(-5*z.delta_FP),money(-150*z.delta_FN)])
    group_rows=[]
    for rid in ORDER:
        for row in cat.loc[cat.run_id.eq(rid)&cat.window.eq('June')&cat.role.eq('diagnostic')&cat.normal.gt(0)].itertuples():
            group_rows.append([LABELS[rid],row.category,int(row.normal),int(row.fit_normal_n),int(row.fit_fraud_n),int(row.threshold_normal_n),int(row.FP),pct(row.fpr)])
    wins=[]
    for rid in PARENTS:
        z=c.loc[c.run_id.eq(rid)&c.window.ne('June')]
        wins.append(f"{LABELS[rid]}相对其原对照：开发期平均每万笔效用变化 **{z.delta_utility_per_10000.mean():+,.2f} 美元**，{int(z.delta_utility.gt(0).sum())}/3 个窗口改善。")
    text='''# 类别及商户代理消融：固定参数、时间划分与成本口径

**主要结论：删除 category 后，两算法在三个开发窗口的效用均较各自原模型改善，六月误报率从接近 100% 降至不足 0.5%。这支持先修正类别依赖，而不是优先增加低证据变量。** CatBoost 9 项和 XGBoost 12 项应作为后续研究基线；继续删除 merchant 没有显示一致收益。

本轮只检验三个预先指定的特征删除分支。原 CatBoost 10 项、XGBoost 13 项模型全部保留；新增 9 次拟合，没有追加参数搜索、重采样或变量工程。结果属于反复使用的开发窗口及六月复用诊断，不是新的独立测试。

## 主要结果：正常交易误报率

'''+table(main_rows,['模型','D1：11–12月','D2：2–3月','D3：5月','6月：复用诊断'])+'''

![误报率](figures/01_false_positive_rates.png)

误报下降必须结合漏报和项目成本判断。项目效用为 U = 50TP − 100FN − 5FP；相对原模型每增加一笔漏报，不仅增加 100 美元漏报成本，还损失 50 美元正确识别收益，因此净损失为 150 美元。

## 开发窗口的配对比较

'''+ '\n\n'.join(wins)+'\n\n'+table(dev_rows,['模型','开发期平均相对全部放行的效用（美元/万笔）','优于全部放行的窗口数 / 3'])+'''

![配对效用](figures/02_paired_utility.png)

上表按三个时间窗口等权平均，避免交易量大的窗口完全主导选择。所有分支均报告，没有根据六月结果选择参数或补充候选。CatBoost 9 项在三个开发窗口的绝对效用均为正；XGBoost 12 项的 D1 效用仍为负。因此，即使六月 XGBoost 12 项更好，也不能只按六月确定全局优胜者。

## 六月：误报下降的收益与漏报代价

'''+table(june_rows,['模型','TP','FP','FN','召回率','精确率','效用 U（美元）','相对全部放行效用变化'])+'''

六月共有 87,778 笔交易，其中欺诈 474 笔、正常 87,304 笔；全部放行效用为 −47,400 美元。六月已在先前研究中观察过，本轮不能把它当作全新的盲测。

'''+table(trade_rows,['新分支','对应原模型','FP变化','FN变化','效用变化','误报减少净收益','漏报变化净收益'])+'''

六月排序能力也同时改善：

'''+table(ranking_rows,['模型','六月 ROC AUC','六月 Average Precision'])+'''

因此，改善不只是改变决策阈值；预测分数区分欺诈与正常交易的能力也增强。继续去掉 merchant 的 XGBoost 11 项相对 12 项仅在 1/3 个开发窗口效用改善，六月效用也略低，暂不支持继续删除 merchant。

## 阈值与正常交易分数迁移

'''+table(threshold_rows,['模型','4月选定阈值','4月误报率','4月正常分数中位数','5月正常分数中位数','6月正常分数中位数'])+'''

![正常分数迁移](figures/03_normal_score_shift.png)

同一分支的五月与六月均使用该分支在四月锁定的阈值；删除特征后重新在四月选择阈值，没有硬套旧模型阈值。分数变化与阈值迁移一起检查，不把输出分数直接解释为已经校准的概率。本轮未在五月或六月寻找最优阈值。

## 类别的正常样本覆盖与误报

'''+table(group_rows,['模型','六月正常类别','六月正常笔数','拟合期正常笔数','拟合期欺诈笔数','4月正常笔数','六月误报笔数','误报率'])+'''

![正常支持分组](figures/04_fit_support_errors.png)

上表的类别和标签只用于事后分组诊断，不输入已经删除类别的模型。D3 拟合期截止 2025-04-01；六月的两类正常交易在该拟合期均没有正常样本，但存在欺诈样本。这比“出现了全新类别”更具体：类别已见过，类别与标签的组合覆盖发生变化。

## 解释边界与后续方向

类别删除后跨期误报、效用均明显改善，支持原模型对类别相关样本结构的依赖是重要失效因素。消融同时会改变分裂、交互和阈值，不能证明唯一原因，也不能推出类别对所有未来任务都无用。商户仍可能代理类别，因此同时报告去类别、去类别及商户两条 XGBoost 分支。

完整训练集延伸至六月，比历史 D3 拟合期覆盖更多类别的正常交易，因此历史模型的六月结果不能直接等同于使用全部训练集后对官方测试集的表现。官方测试集没有标签，本轮未读取为特征、未预测、未提交。

后续优先以本轮证据改善的简化分支作为研究候选，再检验严格过去的短时行为特征；不通过新增大量低证据变量来掩盖原有跨期误报。当前结果不足以宣布模型可部署，或确认未来成本表现。

## 可复现方案与审计

- CatBoost：CB5 固定参数，500 树、深度 10、L2 30，删除 category 后 9 项输入。
- XGBoost：X5 固定参数，500 树、深度 10、L2 30、min_child_weight 5；以原 13 项输入为父版，分别删为 12 项和 11 项。
- 两算法学习率 0.05、随机种子 7406，每次拟合 4 线程；类别词表仅由各折拟合期建立；其余设置沿用原比较实验。
- D1：拟合截止 2024-10-01，10月选阈值，11–12月开发；D2：拟合截止 2025-01-01，1月选阈值，2–3月开发；D3：拟合截止 2025-04-01，4月选阈值，5月开发，6月复用诊断。
- 每折阈值在更早的独立阈值窗口上精确最大化项目效用，分数并列整体处理，同效用选择更少拦截；开发期和六月标签不参与阈值选择。
- 原始数据、历史特征缓存、底层训练代码、协议和两个父模型的文件散列均冻结；所有新分支拟合完成后才生成六月诊断。

'''+f"独立复核 **{qa['passed']} / {qa['total_checks']} 项通过**，核对 **{qa['prediction_rows_checked']:,} 行保存预测**，包括原始标签/时间对齐、完整窗口、阈值最优性、误报漏报与成本、模型重载评分和分组统计。XGBoost 的训练参数通过冻结构造代码及拟合时参数记录核对；UBJSON 重载后的默认训练配置不能用来反推原训练参数。\n\n"+'''主要文件：

- [冻结方案](protocol.json)、[输入及代码散列](manifest.json)、[拟合锁](fit_lock.json)。
- [完整逐窗口指标](tables/metrics.csv)、[配对变化](tables/comparisons.csv)、[逐笔误报/漏报转移](tables/paired_changes.csv)。
- [类别覆盖及错误](tables/category_support_errors.csv)、[分数分布](tables/score_distributions.csv)、[错误样本特征概况](tables/error_feature_profiles.csv)。
- [独立复核记录](qa/independent_validation.json)、[训练代码与复现说明](README.md)。
'''
    (HERE/'report_zh.md').write_text(text,encoding='utf-8')
    html=markdown.markdown(text,extensions=['tables','fenced_code'])
    for f in (HERE/'figures').glob('*.png'):
        html=html.replace('figures/'+f.name,'data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode())
    style='body{font-family:"Microsoft YaHei",sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#183047;line-height:1.8;background:#f9fbfd}h1,h2{color:#12385a}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:20px 0}th,td{border:1px solid #dbe3eb;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf1f8}img{max-width:100%;background:white}a{color:#1669ad}strong{color:#12385a}'
    (HERE/'report_zh.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>类别及商户代理消融诊断</title><style>'+style+'</style></head><body>'+html+'</body></html>',encoding='utf-8')
    print('Chinese Markdown and self-contained HTML reports written.',flush=True)


if __name__=='__main__':main()
