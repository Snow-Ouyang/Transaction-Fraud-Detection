"""Build the Chinese EDA narrative from computed aggregate results only."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_reports(folder, results, figures, markdown_module):
    folder = Path(folder)
    read = lambda name: pd.read_csv(folder/'tables'/f'{name}.csv')
    q, entities = results['quality'], results['entities']
    time_structure = results['temporal_structure']
    n, pos, prevalence = q['train']['rows'], q['train']['fraud'], q['train']['fraud_rate']
    numeric = read('04_numeric_summary')
    category = read('rates_category').sort_values('fraud_rate', ascending=False)
    hour = read('rates_hour')
    monthly = read('12_monthly')
    by_month = monthly.loc[monthly.dataset == 'train']
    shifts = read('30_numeric_train_test_shift')
    cat_shifts = read('31_categorical_train_test_shift')
    pearson = read('05_pearson').set_index('feature')
    effects = read('10_numeric_target_association').sort_values('ks_class_difference', ascending=False)
    target_cat = read('09_categorical_target_association').sort_values('target_cramers_v_bias_corrected', ascending=False)
    dependencies = read('08_functional_dependencies')
    sequence = read('25_sequence_summary')
    profiles = read('22_customer_profile_summary')
    active_categories = read('36_category_active_periods')
    manifest = json.loads((folder/'input_manifest.json').read_text(encoding='utf-8'))

    def fmt(value, spec=',.0f'):
        if pd.isna(value):
            return '未知 / 不适用'
        return format(value, spec)

    def pct(value, decimals=3):
        return format(value*100, f'.{decimals}f')+'%'

    def table(frame, columns):
        # Each entry is (source column, display heading, formatting function).
        rows = ['| '+' | '.join(c[1] for c in columns)+' |',
                '| '+' | '.join('---' for _ in columns)+' |']
        for _, record in frame.iterrows():
            cells = []
            for key, _, formatter in columns:
                value = record[key]
                text = formatter(value) if formatter else str(value)
                cells.append(text.replace('|', '/').replace('\n', ' '))
            rows.append('| '+' | '.join(cells)+' |')
        return '\n'.join(rows)

    def num(feature, label, key):
        return float(numeric.loc[(numeric.dataset == 'train') & (numeric.label == label) & (numeric.feature == feature), key].iloc[0])

    def plot(index):
        item = figures[index-1]
        return f'![{item["title"]}]({item["file"]})\n\n{item["caption"]}\n'

    def group_text(frame, allowed):
        selected = frame.loc[frame.level.isin(allowed)]
        return selected.fraud.sum()/selected.n.sum()

    late = group_text(hour, [22, 23])
    early = group_text(hour, [0, 1, 2, 3])
    daytime = group_text(hour, list(range(4, 22)))
    dictionary = read('33_feature_dictionary_and_actions')
    month_view = monthly[['month', 'dataset', 'n', 'fraud', 'fraud_rate']]
    amount = read('14_amount_bands')
    new_merchant = read('29_new_merchant')
    windows = read('24_new_customers_over_time')
    pure_fraction = entities['fraud_only_customer_transactions']/pos
    total_checks = len(results['checks'])

    sections = []
    sections.append(f'''# ISYE 7406 欺诈交易数据：建模前 EDA

分析日期：2026-10-21。全部计数、分组统计、相关矩阵和分布距离均使用本地全量数据计算，没有拟合预测模型、生成预测或提交 Kaggle。

本报告覆盖全部 23 个训练字段与 22 个测试字段。输入文件保持原样，客户姓名、卡号、详细地址及其原始组合不写入汇总表；客户明细使用匿名编号。

## 1. 需要带入下一阶段的发现

1. 训练集有 **{n:,}** 笔交易，欺诈 **{pos:,}** 笔，占 **{pct(prevalence)}**；样本不平衡且欺诈在客户与时间上聚集。
2. 金额存在明显但非单调的类别差异。正常与欺诈金额中位数分别为 **${num('amt', 0, 'q50'):,.2f}** 和 **${num('amt', 1, 'q50'):,.2f}**；正常金额最大值 **${num('amt', 0, 'q100'):,.2f}**，远高于欺诈最大值 **${num('amt', 1, 'q100'):,.2f}**。不能把极端大额一律视为欺诈。
3. 类别存在显著时间分块：正常交易只出现在 **{time_structure['normal_nonempty_category_month_cells']}/{time_structure['total_category_month_cells_train']}** 个“类别×月份”格中，欺诈覆盖全部 **{time_structure['fraud_nonempty_category_month_cells']}** 格；**{pct(time_structure['fraud_share_in_zero_normal_cells'])}** 的欺诈落在同格没有正常交易的月份/类别组合。该结构已从原始 CSV 独立核对，不能按普通季节波动轻描淡写，也不能直接推断其产生原因。
4. 按数据记录时钟，22–23 时欺诈率为 **{pct(late)}**，0–3 时为 **{pct(early)}**，4–21 时为 **{pct(daytime)}**。类别、金额和时段需要联合考察。
5. **{entities['fraud_only_customers']}** 个客户组合在训练样本中只出现欺诈交易，合计 **{entities['fraud_only_customer_transactions']:,}** 笔，占全部欺诈的 **{pct(pure_fraction)}**。身份字段可能记住样本结构；“仅欺诈”不代表未来行为必然相同。
6. 测试卡号保存格式/精度与训练不一致。六位有效数字归一化在训练中产生 **{entities['rounded_6_significant_digit_collision_buckets']}** 个碰撞组，不能作为唯一身份修复方案。静态客户组合可匹配测试中的 **{entities['shared_customers']:,}** 个既有客户；**{entities['new_test_customer_rows']:,}** 笔交易属于未见组合。
7. 客户/商户经纬度、时间/行号高度相关；客户键也与地址、职业等字段重叠。相关性影响解释，不能据相关阈值机械删列。
8. 测试标签缺失。测试期仅进行质量、身份覆盖和特征分布检查，不估计测试欺诈率、效用或模型表现。

这些是描述性证据。模型是否能利用这些关系、关系能否跨时间保持，仍需冻结验证方案后检验。

## 2. 数据构成与项目评分

| 文件 | 行数 | 原始列数 | 时间范围（由 unix_time 解码） | 标签 |
| --- | ---: | ---: | --- | --- |
| fraudTrain.csv | {q['train']['rows']:,} | {q['train']['columns']} | {q['train']['time_min']} 至 {q['train']['time_max']} | 0 / 1 |
| fraudTest.csv | {q['test']['rows']:,} | {q['test']['columns']} | {q['test']['time_min']} 至 {q['test']['time_max']} | 缺失，未知 |

项目说明中的目标是 `Utility = 50×TP − 100×FN − 5×FP`。金额作为特征，不改变每笔交易的固定评分权重。若概率已经校准，理论拦截阈值为 `5/155 ≈ 3.226%`。

在训练集上，全部预测正常的准确率为 **{pct(results['always_normal_accuracy_train'])}**，效用为 **${results['always_normal_utility_train']:,}**。这只是评分口径参照，没有使用模型。不同月份的参照值保存在 `34_utility_reference_by_month.csv`。

PDF 将客户/商户概括为 1,000 / 800，并写测试集 469,842 行。实际训练中有 {entities['train_cards']:,} 个不同卡号、{int(dictionary.loc[dictionary.feature == 'merchant', 'unique'].iloc[0])} 个商户；本地测试有 {q['test']['rows']:,} 行。以后续本地文件及正式提交模板为准，本轮没有访问或修改比赛账户。
''')

    quality_keys = [('missing_cells', '缺失单元格'), ('duplicate_full_rows', '完全重复行'),
                    ('duplicate_business_rows_excluding_ID', '去掉 ID 后重复行'),
                    ('duplicate_trans_num', '重复交易编号'), ('duplicate_ID', '重复 ID'),
                    ('out_of_order_unix_steps', '原始行序时间倒退次数'),
                    ('largest_backward_seconds', '最大倒退秒数（负值）'),
                    ('text_unix_second_mismatch', '文本时间与 Unix 秒不一致'),
                    ('text_unix_minute_mismatch', '文本时间与 Unix 分钟不一致'),
                    ('cc_num_non_digit_rows', '卡号非纯数字行'),
                    ('zip_less_than_5_char_rows', '邮编不足 5 字符行'),
                    ('nonfinite_numeric_cells', '非有限数值单元格'),
                    ('negative_or_zero_amount_rows', '非正金额行'),
                    ('age_outside_0_110_rows', '年龄超出 0–110 岁行'),
                    ('invalid_latitude_rows', '纬度越界行'), ('invalid_longitude_rows', '经度越界行')]
    quality_rows = pd.DataFrame([{'check': title, 'train': q['train'][key], 'test': q['test'][key]} for key, title in quality_keys])
    sections.append('''## 3. 数据质量、时间和标识符

'''+table(quality_rows, [('check', '检查', None), ('train', '训练', fmt), ('test', '测试', fmt)])+f'''

- 训练文件包含 {q['train']['out_of_order_unix_steps']} 次时间倒退，最大倒退 {abs(q['train']['largest_backward_seconds']):,} 秒。它们不是自动删除的异常交易；后续顺序计算应显式按时间排序。
- 测试文本时间只保留分钟。与 Unix 秒不一致的 {q['test']['text_unix_second_mismatch']:,} 行，在分钟层面全部一致，因此统一采用秒级 `unix_time`。日期出生字段也按文件各自的格式解析。
- 全局同秒时间戳重复不等同于重复交易；不同客户可以同秒交易。历史描述排除同一客户的所有同秒交易，避免假定未知先后顺序。
- 两个文件的 ID 均从 0 开始，跨文件重叠 {entities['ID_overlap']:,} 个，不能用它连接交易。跨文件交易编号重叠为 {entities['trans_num_overlap']}。
- 所有商户名都带有 `fraud_` 前缀（训练 {q['train']['merchant_fraud_prefix_rows']:,} 行），这是命名格式，不能解释为欺诈标签。
- 原始时间没有可靠的客户本地时区描述。本报告的“小时”指记录时钟，不推断客户当地时间。
- 年龄按交易与出生日期的时间差除以 365.2425 天计算，是连续近似年龄；没有用分析当天日期替代交易日期。邮编保留原字符串，不在原文件中补零或改写。

## 4. 全字段清单与处理边界

'''+table(dictionary, [('feature', '字段', None), ('meaning', '含义', None), ('unique', '训练唯一值', fmt),
                       ('candidate_action', '后续处理候选', None), ('caution', '注意事项', None)])+'\n')

    sections.append('''## 5. 时间分布、基准率与季节变化

'''+plot(1)+plot(3)+f'''
训练月度欺诈率从 **{pct(results['fraud_rate_min_month'])}** 到 **{pct(results['fraud_rate_max_month'])}**，交易量也随月份变化。比较模型时应同时报告样本数、正例数和效用，不能把不同长度窗口的总效用直接当作模型优劣。

'''+table(month_view, [('month', '月份', None), ('dataset', '数据集', None), ('n', '交易数', fmt),
                       ('fraud', '欺诈数', fmt), ('fraud_rate', '欺诈率', lambda x: '未知' if pd.isna(x) else pct(x))])+'''

月度、小时、星期汇总均有完整表。任何时间聚集都需要与同一客户重复交易联系起来解释，不能把逐笔记录当成完全独立的重复实验。
''')

    active_normal = active_categories.loc[active_categories.group == 'normal'].copy()
    active_normal['first_time'] = active_normal.first_time.str[:10]
    active_normal['last_time'] = active_normal.last_time.str[:10]
    sections.append('''### 类别时间分块需要单独解释

'''+plot(21)+plot(22)+f'''
正常交易仅覆盖 {time_structure['normal_nonempty_category_month_cells']} 个类别×月份格，欺诈覆盖 {time_structure['fraud_nonempty_category_month_cells']} 格。**{time_structure['zero_normal_positive_fraud_cells']}** 个格子没有正常交易但有欺诈，共 **{time_structure['fraud_rows_in_zero_normal_cells']:,}** 笔，占全部欺诈 **{pct(time_structure['fraud_share_in_zero_normal_cells'])}**。其中大多数支持量很小，前面的欺诈率热力图将它们标灰；组成图保留真实零计数以显露结构。

训练正常交易的“类别—月份”校正 Cramer’s V 为 **{time_structure['normal_category_month_cramers_v']:.4f}**，训练欺诈为 **{time_structure['fraud_category_month_cramers_v']:.4f}**，无标签测试全部交易为 **{time_structure['test_category_month_cramers_v']:.4f}**。标签条件下的时间依赖明显不同。

例如，正常 `gas_transport` 主要处在训练开头，正常 `travel` 只出现在 2025 年 6 月；测试中的这些类别又分别集中在 7 月与 12 月。训练覆盖 18 个月，测试覆盖 6 个月，集中区间不能通过绝对日期直接平移。

'''+table(active_normal, [('category', '类别（训练正常）', None), ('n', '交易数', fmt),
                         ('first_time', '首次日期', None), ('last_time', '末次日期', None), ('active_months', '出现月份数', fmt)])+'''

这项结构已通过重新读取原始 CSV、分别用字符串和分类类型计算得到相同结果。其生成或整理原因尚未核实，不能仅凭 EDA 宣称数据损坏。它会影响时间验证的难度：部分未来正常类别在较早训练期只有欺诈样本。因此必须记录每个窗口的类别支持与标签构成，使用多个时间窗口检查，不能把全期类别平均风险直接当作稳定规则。
''')

    amount_quantiles = pd.DataFrame([{'label': '正常' if label == 0 else '欺诈',
                                      **{key: num('amt', label, key) for key in ['mean', 'q25', 'q50', 'q75', 'q95', 'q99', 'q100']}}
                                     for label in [0, 1]])
    sections.append('''## 6. 金额与类别、时段的联合关系

'''+plot(2)+table(amount_quantiles, [('label', '标签', None), ('mean', '均值 ($)', lambda x: fmt(x, ',.2f')),
                                   ('q25', 'P25', lambda x: fmt(x, ',.2f')), ('q50', 'P50', lambda x: fmt(x, ',.2f')),
                                   ('q75', 'P75', lambda x: fmt(x, ',.2f')), ('q95', 'P95', lambda x: fmt(x, ',.2f')),
                                   ('q99', 'P99', lambda x: fmt(x, ',.2f')), ('q100', '最大值', lambda x: fmt(x, ',.2f'))])+'''

'''+table(amount, [('level', '金额区间 ($)', None), ('n', '交易数', fmt), ('fraud', '欺诈数', fmt), ('fraud_rate', '欺诈率', pct)])+'\n\n'+plot(4)+
        table(category, [('level', '商户类别', None), ('n', '交易数', fmt), ('fraud', '欺诈数', fmt), ('fraud_rate', '欺诈率', pct)])+'\n\n'+plot(5)+plot(6)+plot(7)+'''
金额区间、类别与时段之间存在联合结构，不能从一个类别的总体欺诈率推导其所有金额/时段都同样高风险。热力图将少于 200 笔的格子隐藏，原始计数全部保留，避免把不稳定的小格子当成规律。类别跨月变化图用于观察稳定性，没有据此搜索或筛选最佳规则。
''')

    sections.append('''## 7. 客户人口属性、职业、人口与地理位置

'''+plot(8)+plot(9)+plot(10)+plot(20)+f'''
球面距离中位数：正常 **{num('distance_km', 0, 'q50'):.2f} km**，欺诈 **{num('distance_km', 1, 'q50'):.2f} km**。两类的整体距离分布相近，目前不能仅凭领域直觉认定距离是强特征；也不能因此排除它与类别/时段的交互。

年龄范围、经纬度范围和城市人口长尾见 `04_numeric_summary.csv`。州、城市、职业、商户、性别均导出全水平支持量与欺诈率；图中只展示支持量较大的州和职业，不展示按最高风险挑选的小样本组。

这些统计以交易为权重。某客户交易多，会贡献更多观测；职业、性别、地区之间的差异不能直接解释为客户级别风险或独立影响。职业和地址还可能代理客户身份，需要后续按组检验。

## 8. 客户结构、身份一致性与历史描述

客户键定义为标准化后的 `first + last + dob + street + zip`，内部使用散列，仅用于分组和一致性检查。训练中该组合与原卡号一一对应；训练、测试各自唯一组合数与散列数相同，未发现键碰撞。输出客户表仅含匿名编号。

'''+table(profiles, [('profile', '样本内标签组成', None), ('customers', '客户组合数', fmt),
                    ('transactions', '交易数', fmt), ('fraud', '欺诈数', fmt)])+'\n\n'+plot(13)+f'''
仅欺诈组的每客户交易数中位数为 **{entities['fraud_only_customer_median_support']:g}**。完整客户支持量图与表保留这类组，不能把样本内 100% 欺诈率直接转成线上黑名单。

测试有 **{entities['test_customers']:,}** 个客户组合，其中 **{entities['shared_customers']:,}** 个在训练出现，**{entities['new_test_customers']}** 个未见组合贡献 **{entities['new_test_customer_rows']:,}** 笔交易；商户未见交易数为 **{entities['new_test_merchant_rows']}**。卡号原始 token 只共享 {entities['raw_card_token_shared']} 个，主要反映格式/精度损失。

训练内部按时间观察新客户：

'''+table(windows, [('start', '窗口起点', None), ('end_exclusive', '窗口终点（不含）', None), ('n', '窗口交易', fmt),
                   ('new_customers', '此前未见客户', fmt), ('new_customer_rows', '其交易数', fmt),
                   ('fraud_in_new_customer_rows', '其中训练欺诈数', fmt)])+f'''

训练期有 **{entities['first_customer_transaction_rows']:,}** 笔严格此前无客户历史的交易，其中 **{entities['first_customer_transaction_fraud']:,}** 笔为欺诈（{pct(entities['first_customer_transaction_fraud_rate'])}）。首次客户事件必须有明确的缺失/回退处理。

'''+plot(14)+plot(15)+f'''
历史描述只包含严格早于当前时间戳的信息：此前累计交易数/金额均值、此前 24 小时次数 `[t−24h, t)`、距上一个不同时间戳的间隔、当前金额与历史均值之比、客户是否首次在该商户交易。没有读取历史欺诈标签。

同秒交易整体排除，第一笔交易的历史均值/间隔保留缺失。测试期可以逐笔使用此前已经发生的无标签交易；没有使用当前或未来交易构造历史均值。上述描述仅用于 EDA，没有接入模型。

当前金额/历史均值的中位数，正常为 **{float(sequence.loc[(sequence.label == 0) & (sequence.feature == 'amount_to_past_mean'), 'q50'].iloc[0]):.3f}**，欺诈为 **{float(sequence.loc[(sequence.label == 1) & (sequence.feature == 'amount_to_past_mean'), 'q50'].iloc[0]):.3f}**，值得作为增量候选。相反，两类的此前 24 小时交易数中位数都为 **3**；当前描述不支持直接把“更高频”当作普遍欺诈规则。首次商户也与文件中的类别时间结构交织，需要跨窗口验证。

另外，按客户排序后，存在上一笔欺诈事件的训练正例中，**{pct(entities['repeat_fraud_within_1h_share'])}** 与上一笔相距不超过一小时，**{pct(entities['repeat_fraud_within_24h_share'])}** 不超过 24 小时。这项**标签诊断**揭示事件聚集，不能直接作为可部署特征，也不能假定上一笔欺诈已及时获知。
''')

    strong = dependencies.loc[dependencies.determinant.isin(['merchant', 'zip', 'city', 'street', 'customer_key'])]
    sections.append('''## 9. 相关性、类别依赖与标签边际关联

'''+plot(11)+f'''
几个明确的 Pearson 相关系数：`lat ↔ merch_lat = {pearson.loc['lat', 'merch_lat']:.6f}`，`long ↔ merch_long = {pearson.loc['long', 'merch_long']:.6f}`，`ID ↔ unix_time = {pearson.loc['ID', 'unix_time']:.6f}`。

Pearson 检查线性关系，Spearman 检查单调秩关系。分类变量采用偏差校正 Cramer’s V，而非把卡号/邮编当连续数字计算相关性。函数依赖额外检查“一个值能否唯一确定另一个字段”：

'''+table(strong, [('determinant', '确定字段', None), ('dependent', '被确定字段', None), ('n_levels', '前者水平数', fmt),
                  ('ambiguous_levels', '对应多个后者的水平数', fmt), ('affected_rows', '涉及交易数', fmt)])+'\n\n'+plot(12)+plot(19)+'''
数值标签边际差异（完整结果，不作为自动选特征规则）：

'''+table(effects, [('feature', '字段', None), ('ks_class_difference', '正常/欺诈 KS 距离', lambda x: fmt(x, '.4f')),
                   ('spearman_with_label', '与标签的 Spearman', lambda x: fmt(x, '.4f'))])+'''

分类字段的样本内关联与支持量：

'''+table(target_cat, [('feature', '字段', None), ('levels', '水平数', fmt),
                      ('target_cramers_v_bias_corrected', '与标签的校正 V', lambda x: fmt(x, '.4f')),
                      ('levels_below_100_rows', '不足 100 笔的水平', fmt)])+'''

L2 Logistic Regression 需要数值标准化、正则化和明确去重；树模型通常更能容忍相关字段，但分裂选择和重要性分配仍会变化。下一阶段宜对客户身份、地理、商户等整组字段做删除后重训对照。不能用一张相关矩阵证明字段无用，也不能把高基数身份变量的样本内关联当成泛化能力。
''')

    sections.append('''## 10. 训练/测试无标签分布差异

'''+plot(16)+plot(17)+plot(18)+table(shifts.sort_values('ks_distance', ascending=False),
        [('feature', '字段', None), ('ks_distance', '经验 KS 距离', lambda x: fmt(x, '.4f')),
         ('psi_train_quantile_bins', '分箱 PSI', lambda x: fmt(x, '.4f')),
         ('test_outside_train_range_share', '测试超出训练范围', pct)])+'''

分类字段的分布差异：

'''+table(cat_shifts.sort_values('tvd', ascending=False), [('feature', '字段', None), ('train_levels', '训练水平', fmt),
         ('test_levels', '测试水平', fmt), ('tvd', '总变差 TVD', lambda x: fmt(x, '.4f')),
         ('test_unseen_level_row_share', '测试未见水平占比', pct)])+'''

月份字段的差异包含明确的时间窗口和季节组成差异；卡号原始 token 的差异包含格式失真。年龄随时间增长也会自然变化。以上现象不能直接解释为模型失效或标签漂移。

金额、类别和时段的全期边际分布差异较小，但客户组合的交易权重 TVD 为 **{float(cat_shifts.loc[cat_shifts.feature == 'customer_key', 'tvd'].iloc[0]):.4f}**。多数身份重叠并不代表其交易频率不变。类别在全期的占比接近，也无法消除上文已经观察到的类别×时间分块差异。

经验 KS 使用全量有限观测值的经验分布；PSI 采用训练集分位点分箱（重复边界合并、两端延伸至无穷），每格计数加 0.5 后归一化；JS 用自然对数；TVD 范围 0–1。它们量化特征差异，不估计测试标签，不使用通用 PSI 阈值自动判断好坏。

## 11. 下一阶段的特征边界与验证安排

- 第一版候选以金额、商户/类别、小时/星期、年龄、人口、位置为主。对数金额优先服务于线性模型，树模型不必默认加入重复的单调变换。
- `ID` 与 `trans_num` 排除预测输入。姓名/详细地址首先用于一致性匹配；卡号保持字符串，修复前不直接连接，绝不当连续数值。
- 客户历史有描述性差异，适合作为单独增量实验；身份字段、历史描述、地理字段分别成组检验，避免无法归因的同时堆叠。
- 分类字段的欺诈率表只用于 EDA。后续若使用目标编码，必须限定在各训练折内部，不能把本报告全期分组欺诈率回填模型。
- 按时间划分，禁止随机打散后以相邻事件互相验证。校准/阈值选择与最后验收窗口分开；验证集保持自然类别比例。
- 对明显的类别时间分块，必须增加多个时间窗口与按类别切片的检查，并明确哪些正常类别在拟合窗口尚无支持。不能使用完整训练期或测试期的“类别何时出现”反向生成历史特征；当前或未来标签都不可用于识别正常类别活跃区间。
- 本次已查看全部训练月份的描述性标签统计，包括六月。因此未来六月可作为**不参与参数/阈值选择的验收窗口**，但不能称为从未观察过的盲测。建模前冻结划分和特征假设，并记录后续是否使用其标签作决策。
- 测试文件已进行无标签分布审计；后续模型、阈值及特征增删应依据训练期时间验证，不能把测试分布比较当作性能反馈。正式比赛提交格式、提交限制与榜上 baseline 分数仍需从授权可访问的比赛页面核对。

## 12. 方法限制、复现与核验

数据是模拟交易，描述性模式不构成对真实信用卡业务的性能保证。样本量很大，统计显著性容易出现，因此不使用 p 值筛选特征。分组 Wilson 区间只是逐笔 Bernoulli 参考，没有校正同客户、同欺诈事件的相关性；不能当作跨客户/跨时间稳定性的置信区间。大量分组探索也没有进行多重比较调整，不据此声称确认性发现。

全部统计采用全量记录，没有为图形随机抽样。金额截断图明确标注范围外观测仍在分母；低支持量热力图标灰不补零。测试标签、测试期效用均保持未知。历史描述验证了严格此前时间约束及聚合守恒。

'''+f'本次 **{total_checks}/{total_checks}** 项核验通过，包括输入 SHA-256 未变、月度/类别/客户计数与正例数守恒、交易编号跨集不重复、测试分钟时间一致和历史描述边界。\n\n'+'''
复现方法和实际运行环境见 [README.md](README.md)。完整统计位于 [tables/](tables/)，图形位于 [figures/](figures/)，计算方法位于 [run_eda.py](run_eda.py)。报告生成没有调用任何模型拟合接口。

输入来源及 SHA-256：

'''+table(pd.DataFrame([{'file': name, 'bytes': record['bytes'], 'sha256': record['sha256']} for name, record in manifest['sources'].items()]),
         [('file', '文件', None), ('bytes', '字节数', fmt), ('sha256', 'SHA-256', None)])+'\n')

    markdown_text = '\n\n'.join(sections)
    (folder/'report_zh.md').write_text(markdown_text, encoding='utf-8')
    body = markdown_module.markdown(markdown_text, extensions=['tables', 'fenced_code', 'toc'])
    for item in figures:
        data = base64.b64encode((folder/item['file']).read_bytes()).decode('ascii')
        body = body.replace(f'src="{item["file"]}"', f'src="data:image/png;base64,{data}" loading="lazy"')
    style = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#202d39;
    font:16px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1180px;margin:24px auto;background:white;
    padding:40px 52px;border:1px solid #dfe5eb;border-radius:8px}h1{font-size:29px;line-height:1.5;margin-top:0}
    h2{font-size:22px;margin-top:48px;border-bottom:1px solid #dce3e9;padding-bottom:9px;color:#244c6f}
    img{display:block;max-width:100%;height:auto;margin:24px auto 10px}table{width:100%;border-collapse:collapse;
    font-size:13px;margin:20px 0;display:block;overflow-x:auto}th,td{padding:9px 11px;border-bottom:1px solid #e3e8ed;
    text-align:left;vertical-align:top}th{background:#eaf0f5;white-space:nowrap}tr:nth-child(even){background:#f8fafb}
    code{background:#eef2f5;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}a{color:#235b88}
    li{margin:8px 0}strong{color:#1c3c57}p{margin:14px 0}td{overflow-wrap:anywhere}header{font-size:13px;color:#667887}
    @media(max-width:760px){main{margin:0;padding:22px 16px;border:0}body{font-size:15px}h1{font-size:25px}}
    @media print{body{background:white}main{max-width:none;margin:0;padding:0;border:0}h2{break-after:avoid}img{break-inside:avoid}}
    '''
    output = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    output += '<title>ISYE 7406 建模前 EDA</title><style>'+style+'</style></head><body><main>'
    output += '<header>ISYE 7406 · 全量描述性分析 · 2026-10-21</header>'+body+'</main></body></html>'
    (folder/'report_zh.html').write_text(output, encoding='utf-8')
