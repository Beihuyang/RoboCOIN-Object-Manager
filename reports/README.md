# 效果评估报告

这里保存每次模型、提示、筛选规则或图像处理改动后的 A/B 评估结果。
每项改动使用独立的日期目录，至少包含：

- `summary.json`：机器可读的汇总指标和判定条件；
- `datasets.csv`：每条数据的新旧指标；
- `REPORT.md`：便于阅读的结论与限制；
- `comparisons/all/`：永久保留全部明显变化样本；
- `comparisons/highlights/`：从完整结果中按变化类型精选的样本；
- `human_review.csv`：人工填写 better / worse / same 的复核表。

报告不得修改正式检测、审核、跟踪或物体库数据。纯缓存比较只读取已有缓存；需要重新推理的可控实验必须把中间掩码写入自己的报告目录。

重新生成严格超分对比（默认计入全部 `review_hidden`，排除任一缓存 `revision > 0` 的数据）：

```bash
python reports/compare_super_resolution.py
```

仅用于补充诊断、需要连人工调整数据也纳入时，可显式添加 `--include-manual-revisions`。

重新运行数据集名称语义提示对比（会加载 SAM3，但只写报告目录）：

```bash
python reports/compare_semantic_prompts.py
```

它排除 `revision > 0` 的人工调整数据，共用一次图像编码比较 `object` 与 `object + 数据集名词`，并支持从 `raw/` 断点续跑。
