# Feature tiers and current inputs

## EDA-based research priority

分层用于安排实验优先级，不等同于单变量相关强弱排名或永久删列规则。下面区分最初假设和后来实测结论。

| Tier / role | Candidate variables or groups | Evidence and current status |
|---|---|---|
| 1: initial core | `amt`, `hour`, original `category` | 金额与时段有明显标签差异；category 边际相关强但时间结构不稳定。先做核心基线，后来消融明确移除原始 category，保留 amt/hour。 |
| 2: contextual extensions | 个人金额偏离、客户交易次数/均值/间隔/24h 频次/新商户；`age`, `weekday`, `merchant`；职业/城市/邮编代理组 | 按组加入、时间验证。不同算法保留不同；历史行为普遍有价值。身份代理组复测仍未保留，年龄在最新 XGBoost 删除后提分。 |
| 2: engineered behavior context | 一小时密度、七日消费与长期偏离、个人时段习惯、类别×个人历史 | 无标签严格历史，作为独立增量组验证。CatBoost 接受七项、XGBoost 接受六项；再通过输入删除精简。 |
| 3: deferred | `gender`, `state`, `city_pop`, 客户/商户经纬度及距离等 | EDA 信号较弱、冗余或尚缺可靠增量证据；未系统进入当前模型，不宣称已证明无用。 |
| Excluded from prediction | `ID`, `trans_num`, raw `unix_time`/绝对日期、`cc_num`, 姓名、详细地址、客户键 | 对齐、时间切分或历史匹配专用；避免标识符记忆、时间代理及卡号精度问题。原始身份组合仅构造内部客户键。 |

`job/city/zip` 曾作为第二层的高基数代理组实际测试，最终暂缓；这与从未全面建模的第三层字段不同。

## Selected ordered inputs

精确有序列表和类型适配见 [current_models.json](../configs/current_models.json)。

| Feature | Meaning | CatBoost16 | XGBoost15 |
|---|---|:---:|:---:|
| `amt` | 当前交易金额 | yes | yes |
| `hour` | Unix 秒解码的记录小时，分类输入 | yes | yes |
| `amount_to_past_mean` | 当前金额 / 个人严格过去均值 | yes | yes |
| `has_customer_history` | 是否有严格过去交易 | yes | yes |
| `past_count` | 严格过去交易数 | yes | yes |
| `past_mean_amt` | 严格过去金额均值 | yes | yes |
| `gap_seconds` | 与上一不同时间戳交易的间隔 | yes | yes |
| `count_24h` | `[t−24h,t)` 交易数 | yes | yes |
| `new_merchant` | 此商户是否此前未出现 | yes | dropped |
| `weekday` | 记录星期，分类输入 | no | yes |
| `merchant` | 商户分类输入，fit-only 词表 | no | yes |
| `count_1h` | `[t−1h,t)` 交易数 | yes | yes |
| `distinct_merchants_1h` | 同窗口不同商户数 | yes | dropped |
| `amount_to_mean_7d` | 当前金额 / 个人过去七日均值 | yes | yes |
| `mean_7d_to_past_mean` | 七日均值 / 全历史均值 | yes | yes |
| `hour_unusualness` | 当前时段个人历史稀有程度，Laplace 平滑 | yes | no |
| `category_past_share` | 当前类别在个人过去交易中的占比 | yes | yes |
| `amount_to_shrunk_category_mean` | 当前金额 / 向个人全历史均值收缩的类别均值，固定权重 5 | yes | yes |

冷启动缺少均值/间隔时保留 NaN，不用未来样本补齐。所有同秒事件相互排除。`age` 原在 XGBoost18 中，最新联合删除后不在 XGBoost15 中。

## Correlation interpretation

树模型可以使用相关变量，但冗余会改变分裂和重要性分配；不能据重要性或相关阈值机械删除。Logistic 的早期对照采用标准化、L2 正则和 one-hot。最新 leave-one-out 是固定参数的输入表示检查；保留派生项意味着删除原列不等于删除其信息来源。
