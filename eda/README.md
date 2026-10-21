# EDA findings and machine-readable evidence

全量分析完成，未拟合模型。完整旧展示报告仅在本地归档，GitHub 保留要点和统计。

- 训练 1,402,996 行、10,593 正例（0.755%）；测试 469,843 行，无标签。原文件均无缺失单元格和重复交易编号。
- 正常/欺诈金额中位数 44.21/441.84，但正常金额最大值 37,392，高于欺诈最大值 1,367.10；金额风险不是简单单调关系。
- 正常交易仅覆盖 50/252 个类别×月份格，欺诈覆盖全部格；80.110% 欺诈落在没有正常样本的同格。该结构后来促成原始 category 的跨期消融。
- 记录时钟 22–23 时欺诈率 3.525%，0–3 时 2.048%，4–21 时 0.152%。不推断客户本地时区。
- 训练有 159 次时间倒退，按 `unix_time, ID` 稳定排序。测试文本时间缺秒，统一以 Unix 秒为准。
- 测试卡号格式/精度失真，不能直接跨集匹配；姓名、出生日期、地址和规范邮编组成的内部静态键能匹配 1096 个测试客户组合，551 笔属于未见组合。键不直接入模。
- 地理、身份代理、时间/行号存在明显冗余；相关性只描述关系，不用固定阈值机械删列。目标分组统计仅作 EDA，禁止用全期欺诈率回填预测特征。
- 测试仅用于无标签质量/分布审计，没有性能或欺诈率结论。全部训练月份已被 EDA 观察，六月不能称为盲测。

## Evidence and reproduction

- `results.json`：质量、关键统计和检查结果；原 EDA 聚合检查 22/22 通过。
- `input_manifest.json`：原文件字节数与 SHA-256。
- `tables/01_feature_inventory.csv`、`02_data_quality.csv`：字段与质量。
- `tables/05_pearson.csv`、`06_spearman.csv`、`07_categorical_cramers_v.csv`、`08_functional_dependencies.csv`：依赖和冗余。
- `tables/12_monthly.csv`、`17_category_month.csv`、`35_category_month_composition.csv`、`38_time_structure_summary.csv`：时间结构。
- `tables/30_numeric_train_test_shift.csv`、`31_categorical_train_test_shift.csv`：无标签分布差异。
- 其他 `tables/` 文件保留全量机器可读摘要。

从仓库根目录运行 `python eda/run_eda.py --mode analyze`。不要使用旧 `render/all` 模式生成图形；后续记录均使用 Markdown。

变量层级与后续证据见 [FEATURE_TIERS](../docs/FEATURE_TIERS.md)，时间验证规则见 [VALIDATION](../docs/VALIDATION.md)。
