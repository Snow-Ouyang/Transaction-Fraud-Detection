# XGBoost fraud detection: results and exploration

用户报告公开最佳为 XGBoost30：150945，未经独立查询。模型参数：depth 5、800 棵树、learning rate 0.05、lambda 10、min child weight 1、seed 7406、未加权；全部标注数据拟合，阈值来自此前前推 OOF。历史开发 pooled score 260755（FP 239、FN 10）；不同数据集的公开分和本地分不可直接比较。隐藏榜分数未知。

## 研究路径

- [探索变量及构造目录](FEATURE_EXPLORATION_ARCHIVE.md)：历史候选名称和操作。
- [第一轮变量迭代](overnight_FINAL.md)：32 到 36 项，255625 → 260310，逐项记录条件增益。
- [行为变量迭代](behavior_FINAL.md)：29 到 30 项，类别转移背景 +370，260385 → 260755。
- [针对误判的变量](targeted_FINDINGS.md)：三个单加候选未超过 30 项基线。
- [深度 8 独立单加](depth8_single_FINDINGS.md)：44 项独立对照，最好 260525，低于深度 5 基线。
- [深度 8 联合加入](depth8_union_FINDINGS.md)：44 项联合的分数与误报漏报变化。
- [当前 30 个输入](current30_features.json)；[候选构造源码](overnight_features.py)、[行为特征源码](behavior_features.py)、[误判候选源码](targeted_features.py)。

验证采用严格前推：每个外层时间块的阈值由更早 prefix 的预测和标签选择。自有历史排除当前整秒；部分队友来源特征保留其同秒较早行历史语义。历史窗口和阈值被反复用于选择，因此不是独立盲测，不能承诺公开榜或隐藏榜提升。

这些源码是研究材料，并非独立可执行的完整生产包：原始交易数据、部分已清理缓存、完整运行依赖和部分管线模块未包含。不会上传原始数据、凭据、预测文件或模型权重。

