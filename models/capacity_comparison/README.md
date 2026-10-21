# Capacity search and CatBoost/XGBoost comparison

在10项基础输入上扩大容量并引入XGBoost，固定所选深度10、L2=30、500树。拓展后CatBoost10、XGBoost13；包含原始category的历史版本已被消融轮替代。model_io.py仍是当前便携入口使用的精确适配器。

## Read first

- [Research progress](../../docs/PROGRESS.md)
- [Feature tiers](../../docs/FEATURE_TIERS.md)
- [Validation contract](../../docs/VALIDATION.md)
- [Clean-clone reproduction](../../docs/REPRODUCIBILITY.md)

## Evidence in this directory

原始Python、protocol、manifest、decision/selection locks、汇总results、QA和机器可读tables保留用于审计。`runs/*/result.json`和`June_result.json`保留各候选指标；大型模型、缓存、逐笔预测和旧报告不上传。

历史runner/verify可能依赖未上传的原始本地模型、预测和展示资产，不能在精简clone中直接续跑。当前两模型请从仓库根目录运行 `python scripts/run_current.py --output outputs/new_run --workers 2`。旧可视化生成代码只作历史来源保留，不属于当前Markdown流程。
