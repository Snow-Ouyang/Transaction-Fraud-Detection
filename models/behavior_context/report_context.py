"""Chinese research report built only after both choices and diagnostics are fixed."""
from pathlib import Path
import base64
import json
import sys
import pandas as pd
import markdown

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_context import registry,WINDOWS


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def tab(rows,columns):return pd.DataFrame(rows,columns=columns).to_markdown(index=False)
def pct(x):return f'{x:.2%}'
def cash(x):return f'{x:,.0f}'


def main():
    p,res,items=registry();qa=read(HERE/'qa/independent_validation.json');audit=read(HERE/'qa/feature_validation.json')
    assert qa['all_passed']
    metrics=pd.read_csv(HERE/'tables/metrics.csv');groups=pd.read_csv(HERE/'tables/feature_groups.csv')
    changes=pd.read_csv(HERE/'tables/paired_changes.csv');summary=pd.read_csv(HERE/'tables/run_summary.csv')
    scores=pd.read_csv(HERE/'tables/score_distributions.csv');distributions=pd.read_csv(HERE/'tables/new_feature_distributions.csv')
    chosen=[];head=[];selection_details=[];june=[];cross=[];score_rows=[];absolute_rows=[]
    for family,b in p['baselines'].items():
        s=res['selections'][family];final=s['final_run'];chosen.extend([b['source_run'],final])
        base=summary.loc[summary.run_id.eq(b['source_run'])].iloc[0];win=summary.loc[summary.run_id.eq(final)].iloc[0]
        bm=metrics.loc[metrics.run_id.eq(b['source_run'])&metrics.role.eq('development')]
        wm=metrics.loc[metrics.run_id.eq(final)&metrics.role.eq('development')]
        head.append([family,len(b['features']),len(s['final_features']),cash(bm.utility_per_10000.mean()),cash(wm.utility_per_10000.mean()),f'{win.mean_delta_per_10000-base.mean_delta_per_10000:+,.2f}',', '.join(s['accepted_groups']) or '无'])
        addition=[k for k in s['final_features'] if k not in b['features']]
        selection_details.append(f"**{family}**：最终研究候选 `{final}`，{len(s['final_features'])} 项输入。新增输入："+(', '.join('`'+k+'`' for k in addition) if addition else '无，保留基线')+'。')
        for rid in [b['source_run'],final]:
            z=metrics.loc[metrics.run_id.eq(rid)&metrics.window.eq('June')].iloc[0]
            june.append([family,'基线' if rid==b['source_run'] else '开发期入选',int(z.features),int(z.TP),int(z.FP),int(z.FN),pct(z.false_positive_rate),pct(z.recall),cash(z.utility),cash(z.delta_vs_always_normal),f'{z.average_precision:.4f}'])
        rid=b['tag']+'__category_context';g=groups.loc[groups.run_id.eq(rid)].iloc[0]
        z=changes.loc[changes.run_id.eq(rid)].set_index('window')
        cross.append([family]+[f'{z.loc[w,"delta_per_10000"]:+,.2f}' for w in WINDOWS[:3]]+
            [f'{g.mean_paired_change_per_10000:+,.2f}',f'{int(g.improved_windows)}/3','通过' if g.accepted else '未通过',cash(z.loc['June','delta_utility'])])
    for rid in dict.fromkeys(chosen):
        ab=metrics.loc[metrics.run_id.eq(rid)&metrics.role.eq('development')].set_index('window').loc[WINDOWS[:3]]
        absolute_rows.append([rid]+[cash(ab.loc[w,'utility']) for w in WINDOWS[:3]]+[cash(ab.utility.sum()),cash(ab.utility_per_10000.mean())])
        sub=scores.loc[scores.run_id.eq(rid)&scores.is_fraud.eq(0)&(scores.window.eq('D3')|scores.window.eq('June'))].set_index(['window','role'])
        score_rows.append([rid,f"{sub.threshold.iloc[0]:.6f}",f"{sub.loc[('D3','threshold'),'median']:.5f}",f"{sub.loc[('D3','development'),'median']:.5f}",f"{sub.loc[('June','diagnostic'),'median']:.5f}"])
    text='''# 去类别新基线：行为背景工程与第二类变量重检

本轮接受去掉原始 `category` 的 CatBoost 9 项、XGBoost 12 项作为新研究基线。两个算法固定各自已选最优超参数，独立检验四组行为背景和此前未入选的第二类变量；没有追加参数搜索。

**主指标为官方 score：Total Utility = 50TP − 100FN − 5FP，越高越好。误报率、召回率、AP、AUC 只辅助解释，不组成额外综合分。** 已核对[项目说明 PDF](<../../ISYE 7406 Project Description.pdf>)：公开 leaderboard 使用约30%的测试数据，最终表现按其余隐藏70%计算。因此本轮不使用公开榜反馈调参；依靠固定时间窗口的score筛选。

学习器的训练目标仍沿用Logloss，保持本轮超参数设置不变；官方score用于独立阈值选择和开发期变量组选择，不以训练损失或AUC决定最终入选。

## 开发期选择结果

'''+tab(head,['算法','基线输入数','入选输入数','基线平均score/万笔','入选平均score/万笔','相对基线Δscore/万笔','单独筛选通过组'])+'''

跨窗口主排序使用三个开发窗口等权的每万笔score。在相同窗口内，每万笔score、总score及相对全部放行的score增量产生相同排名；等权窗口平均与把所有窗口拼接求总score不是同一种加权方式。本轮保持训练前冻结的等权规则，同时报告总score。入选版本取基线、筛选通过的单组、筛选通过的合并组中开发score最优者；单组通过不保证一定出现在最终版本中。两个算法原基线的输入不同，因此算法间差异也包含输入差异。

'''+ '\n\n'.join(selection_details)+'''

### 绝对score与汇总

'''+tab(absolute_rows,['模型','D1 score','D2 score','D3 score','三个窗口总score','等权平均score/万笔'])+'''

“至少两个窗口改善”的稳定性条件同样比较score，不要求误报率、召回率或AUC分别改善。所有组的逐窗口绝对score均保存在完整指标表。

## 新行为背景：七项特征的确切定义

| 组 | 新输入 | 提供的行为背景 |
|---|---|---|
| 一小时交易密度 | `count_1h`, `distinct_merchants_1h` | [t−1小时,t) 内该客户的交易次数、不同商户数，补充既有24小时交易次数 |
| 七日消费变化 | `amount_to_mean_7d`, `mean_7d_to_past_mean` | 本笔金额/该客户过去7日均额，以及7日均额/累计历史均额，区分近期变化与长期个人水平 |
| 个人时段习惯 | `hour_unusualness` | −log[(该客户过去在当前小时的交易数+1)/(客户过去交易总数+24)]；固定平滑，不用标签 |
| 类别与个人历史交叉 | `category_past_share`, `amount_to_shrunk_category_mean` | 当前类别在客户过去交易中的占比，以及本笔金额相对该客户该类别收缩历史均额的偏离 |

收缩类别均额 = (客户该类别历史金额总和 + 5×客户累计历史均额)/(客户该类别历史笔数+5)。收缩强度5在训练前固定，没有调优；类别历史不足时向个人长期消费水平回退。无客户历史时相关比值缺失，无7日历史时7日比值缺失，由既有缺失处理管线处理。

七日组的两个比值共享同一个新历史统计量，且与既有个人金额偏离存在代数联系；本轮把整组作为一次检验，不把两个输入视为两份独立信息。新增背景来自过去7日的消费水平，而非只对当前金额做变换。

所有历史统计仅使用严格早于交易时刻的记录；同一客户同一秒的交易互相排除。窗口左边界包含、右边界不包含。小时由 unix_time 按既有口径推导。阈值期、开发期和六月允许更新先前无标签交易历史，使用时不读取任何过去欺诈标签。

原始类别不会重新作为模型输入；类别交叉组也不是目标编码。它仍可能间接携带类别结构，所以必须接受相同的跨期成本检验。

## 第二类变量如何重新进入候选

- CatBoost 分别重试：年龄/星期、商户名称、职业/城市/邮编，共三组。
- XGBoost 的基线已经包含年龄、星期、商户，因此只新增重试职业/城市/邮编；不重复加入已有字段。
- 每组独立加到本算法新基线，避免通过顺序试加混淆各组贡献。基线已有的金额相对个人均值和累计交易历史继续保留。

## 每组的三个开发窗口表现

'''
    for family in p['baselines']:
        rows=[]
        for g in groups.loc[groups.family.eq(family)].itertuples():
            z=changes.loc[changes.run_id.eq(g.run_id)].set_index('window')
            rows.append([g.name,g.features]+[f'{z.loc[w,"delta_per_10000"]:+,.2f}' for w in WINDOWS[:3]]+
                [f'{g.mean_paired_change_per_10000:+,.2f}',f'{int(g.improved_windows)}/3','通过' if g.accepted else '未通过'])
        text+='\n### '+family+'\n\n'+tab(rows,['组','总输入数','D1变化/万笔','D2变化/万笔','D3变化/万笔','平均变化/万笔','改善窗口','筛选'])+'\n'
    text+='''
![各组开发期效用变化](figures/01_group_development_gains.png)

筛选规则在拟合前冻结：相对本算法基线平均效用提高，且至少2/3个开发窗口改善。至少两组单独通过时，只拟合一次通过组的并集，重新按同一规则检验。没有因为看见失败结果而临时改变特征、时间窗、平滑强度或超参数。这里的“通过”是研究筛选，不等于统计显著。

## 类别交叉组的专门判断

'''+tab(cross,['算法','D1变化/万笔','D2变化/万笔','D3变化/万笔','开发均值变化/万笔','改善窗口','筛选','六月效用变化（总美元）'])+'''

上述比较始终针对没有原始类别输入的各自基线。原始类别曾造成严重跨期失效，不意味着所有类别交叉都无用；同样，交叉组有开发期收益也不意味着已经消除了类别支持漂移风险。六月分组诊断单独展示，不用于回头选择交叉形式。

这里衡量的是类别交叉组相对新基线的单组增益；尚未单独验证该组在最终合并版中的边际贡献，合并收益也不能拆成单组收益的简单相加。

![类别交叉诊断](figures/04_category_context_diagnostic.png)

## 入选版本的误报、漏报与六月复用诊断

![误报与召回](figures/02_selected_error_tradeoff.png)

![项目效用](figures/03_selected_utility.png)

'''+tab(june,['算法','版本','输入数','TP','FP','FN','误报率','召回率','六月效用U','相对全部放行效用变化','六月AP'])+'''

六月有87,778笔交易、474笔欺诈、87,304笔正常，全部放行效用为−47,400美元。项目成本为 U=50TP−100FN−5FP；相对同一父模型多一笔漏报的净代价为150美元，因此误报减少不自动意味着效用提高。

所有组、合并组和最终版本都保留记录。六月已在多个前序实验中使用，只有复用诊断意义；两算法开发选择均在本轮六月评分前锁定，不因六月结果改选版本或阈值。如果开发期提升未延续到六月，应明确视为稳定性问题，不能删除不利结果。

## 正常交易分数随时间迁移

'''+tab(score_rows,['模型','4月选定阈值','4月正常分数中位数','5月正常分数中位数','6月正常分数中位数'])+'''

![正常分数迁移](figures/05_normal_score_shift.png)

每个分支使用自己的前期阈值窗口选择阈值；五月和六月沿用四月阈值，不在未来窗口重新优化。输出分数未被假定为已校准概率。

## 复现、审计与结论边界

- CatBoost 固定500树、深度10、学习率0.05、L2=30；XGBoost固定500树、深度10、学习率0.05、L2=30、min_child_weight=5；其他设置完整继承已选版本，每次拟合4线程、种子7406。
- 三个时间折完全沿用原方案：D1拟合至2024-10-01，10月选阈值，11–12月开发；D2拟合至2025-01-01，1月选阈值，2–3月开发；D3拟合至2025-04-01，4月选阈值，5月开发，6月诊断。截止日期均为不含当天。
- 所有输入行自然比例进入拟合；没有类别权重、重采样、提前停止或额外参数网格。分类词表仅由拟合期建立，未知类别沿用原处理。
- 相同开发窗口已反复参与研究筛选，结果存在选择乐观偏差；三个窗口不足以给出强统计显著性结论。没有新的独立有标签测试，不能把某组通过解释为未来收益保证。
- 全量训练到六月拥有比历史D3拟合期更完整的类别正常样本覆盖，但这一点不改变本轮沿用时间验证的规则。官方无标签测试集不读为特征、不预测、不提交。

'''+f"本轮共 **{res['new_fits']} 次新拟合**；原模型保持不变。新历史特征通过 **{audit['independent_spots']} 个独立逐笔检查**、同秒/窗口边界检查及 **{audit['prefix_rows']:,} 行前缀不变性检查**。保存预测独立复核 **{qa['passed']}/{qa['total_checks']} 项通过**，核对 **{qa['prediction_rows_checked']:,} 行预测**，包括原始标签对齐、完整时间窗口、阈值最优性、成本、模型重载评分及筛选重算。\n\n"+'''XGBoost UBJSON重载不保留全部原训练配置，因此训练设置通过冻结构造代码和拟合时参数记录核对，实际保存树用于重载评分核对。特征分布表包含全训练期的描述统计，仅供解释，不据此修改候选。

主要文件：

- [冻结方案](protocol.json)、[代码/数据散列](manifest.json)、[单组选择锁](group_lock.json)、[最终选择锁](selection_lock.json)。
- [逐组筛选表](tables/feature_groups.csv)、[全部逐窗口指标](tables/metrics.csv)、[相对基线变化](tables/paired_changes.csv)。
- [全部候选的score排名](tables/score_rankings.csv)：同时保留预先冻结的等权窗口score排名与拼接窗口总score排名；两者权重不同，不混为一谈。
- [新特征描述统计](tables/new_feature_distributions.csv)、[类别分组误报](tables/category_errors.csv)、[错误样本特征概况](tables/error_feature_profiles.csv)。
- [历史特征审计](qa/feature_validation.json)、[保存预测独立复核](qa/independent_validation.json)、[复现说明](README.md)。
'''
    (HERE/'report_zh.md').write_text(text,encoding='utf-8')
    html=markdown.markdown(text,extensions=['tables','fenced_code'])
    for f in (HERE/'figures').glob('*.png'):html=html.replace('figures/'+f.name,'data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode())
    style='body{font-family:"Microsoft YaHei",sans-serif;max-width:1400px;margin:28px auto;padding:0 24px;color:#183047;line-height:1.8;background:#f8fafc}h1,h2,h3,strong{color:#153b60}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:20px 0}th,td{border:1px solid #d7e1eb;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8f0f8}img{max-width:100%;background:white}a{color:#1669ad}code{font-size:12px;overflow-wrap:anywhere}'
    (HERE/'report_zh.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>行为背景工程与第二类变量重检</title><style>'+style+'</style></head><body>'+html+'</body></html>',encoding='utf-8')
    print('Editable Chinese Markdown and embedded-image HTML reports written.',flush=True)


if __name__=='__main__':main()
