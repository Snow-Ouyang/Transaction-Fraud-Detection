# Research progress and decisions

## Completed sequence

| Stage | Code / evidence | Decision |
|---|---|---|
| EDA | `eda/run_eda.py`, `eda/results.json`, `eda/tables/` | 全量质量、关联、共线性、时间结构和无标签分布检查；按证据规划变量优先级，不直接以相关阈值删列。 |
| Core baseline | `models/core_baseline/` | `amt, category, hour`；L2 Logistic（金额 spline + 标准化、类别 one-hot）与 CatBoost。Logistic 用作早期对照，没有在后期继续调优。 |
| CatBoost progression | `models/catboost_progression/` | 小范围 depth/L2 搜索；加入个人金额偏离和无标签交易历史。早期年龄/星期、商户、职业/城市/邮编增量未达保留规则；最终 10 项。 |
| Capacity comparison | `models/capacity_comparison/` | 在固定 10 项上扩大深度和正则搜索，并同步引入 XGBoost，重试拓展组。选定两者深度 10、L2=30、500 树；CatBoost10、XGBoost13。 |
| Category ablation | `models/category_ablation/` | 固定参数删除原始 category：两模型三个窗口均明显改善，接受 CatBoost9 / XGBoost12。继续删除 merchant 未显示一致收益。 |
| Behavior context | `models/behavior_context/` | 七个新历史背景特征、分组消融与旧拓展组复测，共 42 次新拟合；接受 CatBoost16 / XGBoost18。 |
| Input robustness | `models/leave_one_out/` | 34 个单项删除 + 一个预设联合删除，三时间窗口共 105 次拟合。CatBoost16 无提分删除；XGBoost18 联合删除三项，得到当前 XGBoost15。 |

## Why raw category was removed

EDA 发现正常交易只覆盖 50/252 个类别×月份格，而欺诈覆盖全部格；这提示类别与时间共同形成的样本结构。后续时间外诊断中，包含原始 category 的 CatBoost10 六月误报率为 100%，XGBoost13 为 99.75%。固定参数删除 category 后，分别降到约 0.46% 和 0.38%，六月 score 分别从 -419420、-419390 升至 9230、11530。

这支持修正当前数据上的类别依赖，不说明 category 在所有任务中无用，也未证明数据生成过程。类别×个人过去交易背景后来被作为单独候选重新检验；不能把全期类别欺诈率直接编码回模型。

## Behavior engineering

四组共七项：一小时频次与商户数；七日金额水平与长期偏离；个人时段异常；当前类别在个人历史中的占比及收缩类别金额偏离。所有计算无标签、严格排除当前及同秒事件。

- CatBoost 接受七项全部加入，9 → 16；相对无类别基线平均 score +939.93/万笔，3/3 窗口提高；六月 9230 → 19335。
- XGBoost 接受六项（不含个人时段异常），12 → 18；平均 score +1386.45/万笔，3/3 窗口提高；六月 11530 → 19775。
- CatBoost 复测年龄/星期、商户和职业/城市/邮编均未入选；XGBoost 复测职业/城市/邮编未入选。
- 类别背景组的独立增量相对较弱，不能把整个 union 的提升归因于它；最新单项删除则支持在当前组合中保留两个类别背景输入。

## Current simplification result

| XGBoost deleted input(s) | Mean gain / 10000 | Pooled gain | Positive windows |
|---|---:|---:|---:|
| `age` | +114.28 | +7080 | 2/3 |
| `new_merchant` | +17.37 | +1195 | 2/3 |
| `distinct_merchants_1h` | +23.39 | +880 | 3/3 |
| All three together | **+121.28** | **+7820** | **2/3** |
| `has_customer_history` | 0 | 0 | 0/3 |

单项收益不能直接相加。联合方案已实测，开发窗口 score 为 `[41590, 46690, 26995]`；CatBoost16 为 `[42670, 45890, 26280]`。联合 XGBoost 在 D2/D3 高于 CatBoost，D1 仍低于 CatBoost。

`has_customer_history` 单独删除保持分数不变，但未另加到三项联合删除中，因此不能把尚未训练的 14 项组合当作已验证结果。

六月 XGBoost18 → XGBoost15：TP 455→460、FP 215→305、FN 19→14，score 19775→20075。按主 score 应保留净收益，不能只因误报增加否决；同时也不能忽略这一操作代价。

最新独立 QA：3046/3046 检查通过，26640185 行保存的预测被检查。行为工程轮 QA 为 2235/2235，通过记录不等于新 clone 已重做完整历史 QA。[逐项结果](../models/leave_one_out/RESULTS.md)

## Current direction

XGBoost15 做快速探索主线；CatBoost16 保持固定比较，只复核有希望的方案。尚未继续大范围参数搜索、全面测试第三层字段、训练最终官方测试模型或提交。下一步研究必须新建实验并记录阈值、变量、候选和选择规则，避免沿用结果倒推假设。
