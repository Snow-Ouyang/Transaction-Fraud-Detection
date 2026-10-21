# Leave-one-out robustness check

## 本轮结论

- 固定模型超参数、随机种子及时间划分；34 个单项删除和 1 个预设联合删除，共 105 次训练。阈值仍只在各窗口之前的校准期选择。
- CatBoost：16 个单项删除均未提高开发期等权平均 score，保留现有 16 项输入。
- XGBoost：同时删除 `new_merchant`、`age`、`distinct_merchants_1h` 的 15 项方案，本轮平均分和总分均最高；相对原 18 项方案，平均分 +121.28/万笔、总分 +7,820，2/3 开发窗口提高。六月 score 19,775 → 20,075；FP 215 → 305，FN 19 → 14，按 score 仍有净收益。
- 两模型最终方案差距较小：XGBoost 平均分 2,960.49 vs CatBoost 2,926.45（+1.16%）；开发期总分 115,275 vs 114,840（+435）；六月 20,075 vs 19,335（+740）。六月已多次观察，仅作诊断，不能证明独立测试优势。
- 建议 XGBoost15 作为下一轮探索主线，CatBoost16 冻结为对照、只复核有希望的方案；当前分数差距不足以彻底淘汰 CatBoost。此建议未覆盖或修改原 baseline 模型。
- 独立核验 3,046/3,046 项通过，检查 26,640,185 行保存的预测。官方无标签测试集未加载或预测。

Status: complete; complete candidate runs: 35/35.

Fixed CatBoost16 / XGBoost18 baselines and model hyperparameters. Same D1/D2/D3 chronological folds and prior-window threshold selection. Primary = equal-window mean official score/10000. No figures or HTML.

Input columns are removed; other derived features and historical inputs remain unchanged. A dropped raw amount/history column may still be represented through ratios. This is not proof that its underlying information is unnecessary.

| Family | Baseline inputs | Mean score/10000 | Pooled development score |
|---|---:|---:|---:|
| catboost | 16 | 2,926.45 | 114,840 |
| xgboost | 18 | 2,839.21 | 107,455 |

## All completed candidates

Positive gain is reported even if concentrated in one window. Robust+ means mean gain>0 and >=2/3 windows improve. Delta units below are score/10000; pooled delta is total dollars.

| Family | Removed inputs | Kind | D1 delta | D2 delta | D3 delta | Mean delta | Pooled delta | Wins | Robust+ |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| catboost | `amount_to_past_mean` | single | -27.66 | -34.45 | +19.99 | -14.04 | -880 | 1/3 | no |
| catboost | `distinct_merchants_1h` | single | +6.07 | -105.76 | +23.02 | -25.56 | -995 | 2/3 | no |
| catboost | `amount_to_mean_7d` | single | +3.37 | -44.07 | -79.95 | -40.22 | -1,135 | 1/3 | no |
| catboost | `count_24h` | single | -55.32 | -160.64 | +61.18 | -51.59 | -2,730 | 1/3 | no |
| catboost | `count_1h` | single | -20.01 | -109.36 | -36.34 | -55.24 | -2,110 | 0/3 | no |
| catboost | `gap_seconds` | single | -54.19 | -145.02 | +33.31 | -55.30 | -2,740 | 1/3 | no |
| catboost | `new_merchant` | single | -11.69 | -55.68 | -106.00 | -57.79 | -1,830 | 0/3 | no |
| catboost | `amount_to_shrunk_category_mean` | single | -2.47 | -64.90 | -138.71 | -68.69 | -2,010 | 0/3 | no |
| catboost | `past_mean_amt` | single | -65.66 | -147.42 | -4.85 | -72.64 | -3,340 | 0/3 | no |
| catboost | `has_customer_history` | single | -145.04 | -59.69 | -43.61 | -82.78 | -4,330 | 0/3 | no |
| catboost | `amt` | single | +11.02 | -189.89 | -78.14 | -85.67 | -2,770 | 1/3 | no |
| catboost | `past_count` | single | -49.25 | -158.24 | -51.49 | -86.32 | -3,495 | 0/3 | no |
| catboost | `category_past_share` | single | -81.40 | -177.47 | -77.53 | -112.13 | -4,665 | 0/3 | no |
| catboost | `hour_unusualness` | single | -255.90 | -180.27 | -42.40 | -159.52 | -8,290 | 0/3 | no |
| catboost | `mean_7d_to_past_mean` | single | -125.48 | -395.39 | -155.06 | -225.31 | -9,005 | 0/3 | no |
| catboost | `hour` | single | -1,623.98 | -1,614.83 | -777.12 | -1,338.64 | -62,680 | 0/3 | no |
| xgboost | `new_merchant`, `age`, `distinct_merchants_1h` | joint | +312.79 | +104.96 | -53.91 | +121.28 | +7,820 | 2/3 | yes |
| xgboost | `age` | single | +269.84 | +112.97 | -39.98 | +114.28 | +7,080 | 2/3 | yes |
| xgboost | `distinct_merchants_1h` | single | +0.67 | +68.90 | +0.61 | +23.39 | +880 | 3/3 | yes |
| xgboost | `new_merchant` | single | +44.30 | +34.45 | -26.65 | +17.37 | +1,195 | 2/3 | yes |
| xgboost | `has_customer_history` | single | +0.00 | +0.00 | +0.00 | +0.00 | +0 | 0/3 | no |
| xgboost | `amount_to_mean_7d` | single | -25.41 | +56.89 | -39.37 | -2.63 | -180 | 1/3 | no |
| xgboost | `weekday` | single | +38.23 | +35.25 | -85.40 | -3.97 | +585 | 2/3 | no |
| xgboost | `amount_to_past_mean` | single | +44.07 | -4.41 | -62.99 | -7.78 | +405 | 1/3 | no |
| xgboost | `amount_to_shrunk_category_mean` | single | -18.89 | -104.16 | -38.16 | -53.73 | -2,035 | 0/3 | no |
| xgboost | `count_1h` | single | -200.58 | +0.80 | -60.57 | -86.78 | -4,950 | 1/3 | no |
| xgboost | `count_24h` | single | -183.72 | -13.62 | -113.87 | -103.74 | -5,195 | 0/3 | no |
| xgboost | `merchant` | single | -124.13 | -132.60 | -82.98 | -113.24 | -5,100 | 0/3 | no |
| xgboost | `amt` | single | -150.21 | -73.31 | -118.11 | -113.88 | -5,230 | 0/3 | no |
| xgboost | `past_mean_amt` | single | -301.32 | -53.28 | -23.02 | -125.87 | -7,555 | 0/3 | no |
| xgboost | `category_past_share` | single | -415.78 | +41.66 | -113.87 | -162.66 | -9,665 | 1/3 | no |
| xgboost | `mean_7d_to_past_mean` | single | -428.59 | -62.49 | -69.05 | -186.71 | -10,880 | 0/3 | no |
| xgboost | `gap_seconds` | single | -319.76 | -96.55 | -144.16 | -186.82 | -9,505 | 0/3 | no |
| xgboost | `past_count` | single | -881.92 | -156.64 | -133.26 | -390.60 | -22,665 | 0/3 | no |
| xgboost | `hour` | single | -483.24 | -825.64 | -602.68 | -637.19 | -26,025 | 0/3 | no |

## Locked choices

- catboost: score leader `CB__combined`; robust leader `CB__combined`.
- xgboost: score leader `XGB__joint__mean_positive`; robust leader `XGB__joint__mean_positive`.

## June reused diagnosis (not selection)

| Family | Run | Score | FP | FN |
|---|---|---:|---:|---:|
| catboost | `CB__combined` | 19,335 | 273 | 20 |
| xgboost | `XGB__combined` | 19,775 | 215 | 19 |
| xgboost | `XGB__joint__mean_positive` | 20,075 | 305 | 14 |

Verification: 3046/3046 checks passed; 26,640,185 saved prediction rows checked.

Joint drops test only the prespecified positive/robust-positive unions, not every combination or iterative backward elimination. Reused development windows imply selection optimism; June was previously observed. Official unlabeled test is not loaded or scored.

## Secondary total-score view

Same frozen candidates, no new models or changed primary selection. Pooled leaders were recorded before generating June predictions. Different fold lengths make pooled total and equal-window mean genuinely different weights.

| Family | View | Removed | Mean score/10000 | Total score | Total gain | Positive windows | June score |
|---|---|---|---:|---:|---:|---:|---:|
| catboost | baseline | none | 2,926.45 | 114,840 | +0 | 0/3 | 19,335 |
| catboost | equal-window score leader | none | 2,926.45 | 114,840 | +0 | 0/3 | 19,335 |
| catboost | robust score leader | none | 2,926.45 | 114,840 | +0 | 0/3 | 19,335 |
| catboost | pooled total score leader | none | 2,926.45 | 114,840 | +0 | 0/3 | 19,335 |
| xgboost | baseline | none | 2,839.21 | 107,455 | +0 | 0/3 | 19,775 |
| xgboost | equal-window score leader | `new_merchant`, `age`, `distinct_merchants_1h` | 2,960.49 | 115,275 | +7,820 | 2/3 | 20,075 |
| xgboost | robust score leader | `new_merchant`, `age`, `distinct_merchants_1h` | 2,960.49 | 115,275 | +7,820 | 2/3 | 20,075 |
| xgboost | pooled total score leader | `new_merchant`, `age`, `distinct_merchants_1h` | 2,960.49 | 115,275 | +7,820 | 2/3 | 20,075 |
