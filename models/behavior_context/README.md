# Past-only behavior context and legacy retests

固定无category父版本和超参数，测试四组七项新背景特征及旧拓展组。42次拟合后选CatBoost16和XGBoost18；XGBoost18随后经leave-one-out精简至15项。features.py是当前便携入口使用的原始严格历史实现。

## Read first

- [Research progress](../../docs/PROGRESS.md)
- [Feature tiers](../../docs/FEATURE_TIERS.md)
- [Validation contract](../../docs/VALIDATION.md)
- [Clean-clone reproduction](../../docs/REPRODUCIBILITY.md)

## Evidence in this directory

原始Python、protocol、manifest、decision/selection locks、汇总results、QA和机器可读tables保留用于审计。`runs/*/result.json`和`June_result.json`保留各候选指标；大型模型、缓存、逐笔预测和旧报告不上传。

历史runner/verify可能依赖未上传的原始本地模型、预测和展示资产，不能在精简clone中直接续跑。当前两模型请从仓库根目录运行 `python scripts/run_current.py --output outputs/new_run --workers 2`。旧可视化生成代码只作历史来源保留，不属于当前Markdown流程。
