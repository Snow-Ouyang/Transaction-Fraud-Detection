# Data

原始文件由项目所有者提供，位于仓库根目录。使用 Git LFS 存储原 CSV，不压缩、不改写、不转码。精确 SHA-256 和字节数见 [manifest.json](manifest.json)。

两个文件均超过普通 Git 的 100 MiB 单文件限制，因此使用 LFS；见 [GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

| File | Rows | Columns | Unix-time date range | Label |
|---|---:|---:|---|---|
| `fraudTrain.csv` | 1402996 | 23 | 2024-01-01–2025-06-30 | `is_fraud`，正例 10593（0.755%） |
| `fraudTest.csv` | 469843 | 22 | 2025-07-01–2025-12-31 | 无标签 |

```bash
git lfs pull
python scripts/verify_data.py
```

GitHub 显示 LFS pointer 时，不能把它当作 CSV 读取。校验脚本会识别这种情况并提示下载。

数据包含模拟交易的客户、商户、时间、金额和身份字段。文件原样保留供复现；模型只使用 `configs/current_models.json` 中指定的输入，原始姓名、详细地址、卡号和交易编号不是直接模型输入。

质量问题和处理依据见 [EDA 要点](../eda/README.md)。两个文件的 ID 会重置，不能跨文件直接用 ID 连接交易。训练按 `unix_time, ID` 稳定排序；历史特征排除同秒事件。

原课程项目说明 PDF 的 SHA-256 为 `dd91700489d33a41c7bf567c9f5ad18f4988b957769d3f64fed273cb387c8a26`，保留在所有者本地；评分和验证要求已整理为 Markdown。本仓库不替数据或课程材料另行声明许可证。
