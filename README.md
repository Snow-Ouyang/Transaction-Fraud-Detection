# Transaction Fraud Detection

ISYE 7406 欺诈交易竞赛。当前已完成 **EDA → 变量分层 → 第一、二层变量建模 → 行为特征工程与输入精简**。以官方 score `50×TP − 100×FN − 5×FP` 为主，其他指标用于诊断。

## 当前进度

| 阶段 | 已完成工作 | 主要结论与入口 |
|---|---|---|
| 1. EDA | 全量数据质量、标签分布、时间结构、变量关联、共线性及无标签训练/测试差异检查 | 严重不平衡；类别×时间存在强烈样本结构；卡号格式不宜直接跨集匹配。[EDA 要点](eda/README.md) |
| 2. 依据 EDA 对变量分层 | 核心信号、增量候选、暂缓字段分别处理；标识符仅作对齐或历史匹配 | 分层是实验优先级，不是永久有效性判断。[变量层级及演变](docs/FEATURE_TIERS.md) |
| 3. 测试第一、二层变量 | 核心 Logistic/CatBoost；CatBoost 小范围调优；扩大模型容量并与 XGBoost 同步比较；重试拓展组 | 最终两模型均使用深度 10、500 棵树、L2=30；原始 `category` 跨期误报严重，消融后删除。[实验进展](docs/PROGRESS.md) |
| 4. 特征工程与精简输入 | 严格历史金额、频次、近期行为及类别×个人历史背景；复测旧候选；逐项删除和联合删除 | 当前 CatBoost16 保留作固定对照；XGBoost18 删除三项后得到 XGBoost15，作为后续主线。[最新删变量结果](models/leave_one_out/RESULTS.md) |

## 当前模型结果

同一组 D1/D2/D3 时间回测；阈值只在各自之前的校准窗口选择，模型超参数在本轮删变量检查中不变。

| 方案 | 输入数 | 开发期等窗口平均 score／万笔 | 开发期总 score | 六月 score |
|---|---:|---:|---:|---:|
| CatBoost16 | 16 | 2926.45 | 114840 | 19335 |
| XGBoost18：精简前对照 | 18 | 2839.21 | 107455 | 19775 |
| **XGBoost15：当前主线** | **15** | **2960.49** | **115275** | **20075** |

XGBoost15 同时删除 `age`、`new_merchant`、`distinct_merchants_1h`；相对 XGBoost18，2/3 开发窗口提分，总分 +7820。CatBoost 的 16 个单项删除都未提升平均分。当前两模型差距仍小：XGBoost 平均分领先约 1.16%，开发期总分只高 435，因此保留 CatBoost 对照。

六月已被多轮观察，仅是复用诊断；开发窗口也反复参与选择。这些不是独立盲测结果，不代表 leaderboard 分数。公开榜使用约 30% 官方测试数据、最终排名使用隐藏约 70%；目前尚未生成官方测试预测或提交。[验证口径](docs/VALIDATION.md)

## 快速复现

两个原始 CSV **未经压缩、改名或内容修改**，通过 Git LFS 保存。安装 Git LFS 后：

```bash
git lfs install
git clone https://github.com/Snow-Ouyang/Transaction-Fraud-Detection.git
cd Transaction-Fraud-Detection
git lfs pull
python -m pip install -r requirements.txt
python scripts/verify_data.py
python scripts/run_current.py --output outputs/reproduce_current --workers 2
```

建议 Python 3.13.5。当前入口重建无标签历史特征、执行两模型的三个时间窗口，并在全部开发结果完成后生成六月诊断；只产生 Markdown、JSON 和必要的本地模型/预测文件。也可先用 `--prepare-only` 检查特征构建，或用 `--family xgboost` 只复现主线。[完整复现说明](docs/REPRODUCIBILITY.md)

整理发布时已通过新入口完成六次重训：重建特征与历史缓存完全一致，1,440,010 行预测得到复核，两个模型全部窗口分数均复现成功。[机器可读复现记录](docs/reproduction_results.json)

## 目录与 agent 阅读顺序

```text
README.md                       当前进度和最终方案
AGENTS.md                       后续研究规则
docs/                           进展、变量层级、验证与复现说明
configs/current_models.json     两个最终模型的有序输入和固定超参数
fraudTrain.csv / fraudTest.csv   原始数据，Git LFS
data/                           数据字节数、行数和 SHA-256 清单
scripts/                        新 clone 可运行的当前方案入口与数据校验
eda/                            原始 EDA 代码、机器可读统计和要点
models/                         历轮代码、协议、锁定决策、汇总指标和 QA
```

其他 agent 建议依次阅读 `AGENTS.md` → `docs/PROGRESS.md` → `docs/FEATURE_TIERS.md` → `docs/VALIDATION.md` → `configs/current_models.json`。历史实验目录保留失败分支，避免重复试验；它们不是都能在缺少本地模型的 clone 中直接续跑。

所有新报告均为 Markdown。既有人工检阅报告和图形已留存于本地 `archive/pre_github_2026-10-21/`，不上传；本地历史模型、逐笔预测、特征缓存、第三方 `.deps` 同样不上传。普通 Git 保留精简结果记录，原始 CSV 通过 LFS 上传。
