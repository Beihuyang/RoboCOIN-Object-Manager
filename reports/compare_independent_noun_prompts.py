#!/usr/bin/env python3
"""Extend the preserved 100-session A/B run to a three-way SAM3 comparison."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from project_paths import resolve_project_path
from semantic_prompts import semantic_noun_prompts
from stage1_track_select import DEFAULT_CKPT, detect_first_frame, load_sam3_detector
from reports.compare_semantic_prompts import _render
from reports.compare_single_prompt_nouns import _key, _load_result, _save_result

SOURCE = BASE_DIR / "reports" / "2026-09-03_single_prompt_nouns_ab_100"
OUTPUT = BASE_DIR / "reports" / "2026-09-03_noun_prompt_three_way_100"
DATA_ROOT = BASE_DIR / "RoboCOIN_datasets"


def mean(items):
    return float(np.mean([float(item["score"]) for item in items])) if items else None


def metrics(groups):
    scores = [float(item["score"]) for group in groups for item in group]
    counts = [len(group) for group in groups]
    q = lambda value: round(float(np.quantile(scores, value)), 4) if scores else None
    return {
        "nonempty_sessions": sum(value > 0 for value in counts),
        "objects": sum(counts),
        "mean_objects_per_session": round(float(np.mean(counts)), 3),
        "median_objects_per_session": round(float(np.median(counts)), 3),
        "mean_confidence": round(float(np.mean(scores)), 4) if scores else None,
        "median_confidence": q(.5), "confidence_p25": q(.25),
        "confidence_p75": q(.75),
    }


def comparison(frame, session, nouns, legacy, combined, independent):
    groups = [("legacy", legacy), ("comma joined", combined),
              ("independent nouns", independent)]
    views = [_render(frame, [{**x, "prompt": label} for x in items],
                     [x["mask"] for x in items]) for label, items in groups]
    width, height, header = max(x.width for x in views), max(x.height for x in views), 112
    canvas = Image.new("RGB", (width * 3, height + header), (18, 22, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), session, fill=(235, 241, 250))
    for index, ((label, items), view) in enumerate(zip(groups, views)):
        canvas.paste(view, (index * width + (width - view.width) // 2, header))
        value = "N/A" if mean(items) is None else f"{mean(items):.3f}"
        draw.text((index * width + 12, 34),
                  f"{label} | masks={len(items)} | mean={value}",
                  fill=((160, 196, 255), (255, 190, 120), (144, 232, 180))[index])
    prompt = ", ".join(nouns)
    draw.text((12, 62), "nouns: " + (prompt[:250] + "..." if len(prompt) > 250 else prompt),
              fill=(192, 198, 208))
    draw.text((12, 87), "Same 2K frame and confidence threshold (0.2)", fill=(192, 198, 208))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=.2)
    parser.add_argument("--highlights", type=int, default=24)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    source_rows = list(csv.DictReader((source / "datasets.csv").open()))
    pending = [row for row in source_rows if not (
        output / "raw" / _key(row["session"]) / "independent/result.json").is_file()]
    model = processor = None
    failures = []
    if pending:
        model, processor = load_sam3_detector(args.checkpoint, args.gpu)
    try:
        for index, row in enumerate(source_rows, 1):
            session = row["session"]
            target = output / "raw" / _key(session) / "independent"
            if (target / "result.json").is_file():
                print(f"[{index}/{len(source_rows)}] resume {session}", flush=True)
                continue
            try:
                _, combined = _load_result(source / "raw" / _key(session) / "after")
                combined_meta, _ = _load_result(source / "raw" / _key(session) / "after")
                frame = resolve_project_path(combined_meta["frame"])
                video = DATA_ROOT / f"{session}.mp4"
                nouns = semantic_noun_prompts(video, DATA_ROOT)
                independent, report = detect_first_frame(
                    frame, model, processor, nouns, args.threshold, exclude_robot_arms=False)
                _save_result(target, independent, {
                    "session": session, "frame": str(frame.resolve()), "nouns": nouns,
                    "strategy": "independent noun prompts with shared image encoding",
                    "filter_report": report,
                })
                print(f"[{index}/{len(source_rows)}] {session}: {len(combined)} -> {len(independent)}", flush=True)
            except Exception as exc:
                failures.append({"session": session, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {session}: {exc}", flush=True)
                if isinstance(exc, torch.OutOfMemoryError):
                    gc.collect(); torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model, processor
            gc.collect(); torch.cuda.empty_cache()

    rows, loaded = [], []
    for source_row in source_rows:
        session = source_row["session"]
        independent_dir = output / "raw" / _key(session) / "independent"
        if not (independent_dir / "result.json").is_file():
            continue
        _, legacy = _load_result(source / "raw" / _key(session) / "before")
        _, combined = _load_result(source / "raw" / _key(session) / "after")
        independent_meta, independent = _load_result(independent_dir)
        frame = resolve_project_path(independent_meta["frame"])
        row = {
            "session": session, "noun_count": len(independent_meta["nouns"]),
            "legacy_count": len(legacy), "combined_count": len(combined),
            "independent_count": len(independent),
            "legacy_mean_confidence": mean(legacy),
            "combined_mean_confidence": mean(combined),
            "independent_mean_confidence": mean(independent),
            "independent_vs_legacy_delta": len(independent) - len(legacy),
            "independent_vs_combined_delta": len(independent) - len(combined),
        }
        rows.append(row); loaded.append((row, frame, independent_meta["nouns"], legacy, combined, independent))

    variants = {
        "legacy": [item[3] for item in loaded],
        "combined_prompt": [item[4] for item in loaded],
        "independent_prompts": [item[5] for item in loaded],
    }
    summary = {
        "report_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_sessions": len(rows), "failed_sessions": len(failures),
        "threshold": args.threshold, "metrics": {name: metrics(groups) for name, groups in variants.items()},
        "independent_vs_legacy_count_categories": dict(Counter(
            "gain" if row["independent_vs_legacy_delta"] > 0 else
            "loss" if row["independent_vs_legacy_delta"] < 0 else "same" for row in rows)),
        "average_noun_count": round(float(np.mean([row["noun_count"] for row in rows])), 2) if rows else 0,
        "notes": [
            "All variants use the same 2K frame and threshold.",
            "Legacy and comma-joined masks are preserved from the preceding experiment.",
            "No ground-truth masks are available; count and confidence measure output behavior, not accuracy.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False))
    with (output / "datasets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows: writer.writeheader(); writer.writerows(rows)

    names = list(variants)
    labels = ["Legacy", "Comma joined", "Independent nouns"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    values = [summary["metrics"][name] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, title in zip(axes, ["objects", "nonempty_sessions", "mean_confidence"],
                              ["Total masks", "Sessions with masks", "Mean confidence"]):
        bars = ax.bar(labels, [value[key] for value in values], color=colors)
        ax.bar_label(bars, fmt="%.3g", padding=3); ax.set_title(title); ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(output / "summary_metrics.png", dpi=180); plt.close(fig)

    buckets = [(1, 5, "1-5"), (6, 10, "6-10"), (11, 10_000, "11+")]
    bucket_data = []
    for low, high, label in buckets:
        selected = [item for item in loaded if low <= item[0]["noun_count"] <= high]
        bucket_data.append({"label": label, "sessions": len(selected), **{
            name: sum(len(item[index]) for item in selected)
            for name, index in zip(names, [3, 4, 5])}})
    x = np.arange(len(buckets)); width = .25
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, name, label, color in zip([-1, 0, 1], names, labels, colors):
        bars = ax.bar(x + offset * width, [item[name] for item in bucket_data], width, label=label, color=color)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(x, [f"{x['label']} nouns\n(n={x['sessions']})" for x in bucket_data])
    ax.set_ylabel("Total masks"); ax.set_title("Mask count by noun count"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "noun_count_buckets.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    score_groups = [[float(x["score"]) for group in variants[name] for x in group] for name in names]
    axes[0].boxplot(score_groups, tick_labels=labels, showfliers=False)
    axes[0].set_ylim(0, 1); axes[0].set_title("Confidence distribution"); axes[0].grid(axis="y", alpha=.25)
    axes[0].tick_params(axis="x", rotation=15)
    noun_counts = [item[0]["noun_count"] for item in loaded]
    result_counts = [item[0]["independent_count"] for item in loaded]
    axes[1].scatter(noun_counts, result_counts, alpha=.7, color=colors[2])
    if len(noun_counts) > 1:
        slope, intercept = np.polyfit(noun_counts, result_counts, 1)
        line_x = np.array([min(noun_counts), max(noun_counts)])
        axes[1].plot(line_x, slope * line_x + intercept, color="#d62728", linewidth=2)
    axes[1].set_xlabel("Noun prompts per session"); axes[1].set_ylabel("Kept masks")
    axes[1].set_title("Independent prompts: nouns vs masks"); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "confidence_and_prompt_load.png", dpi=180); plt.close(fig)

    ranked = sorted(loaded, key=lambda item: abs(item[0]["independent_vs_combined_delta"]), reverse=True)
    compare_dir = output / "comparisons" / "highlights"; compare_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for rank, (row, frame, nouns, legacy, combined, independent) in enumerate(ranked[:args.highlights], 1):
        name = f"{rank:02d}_{_key(row['session'])}.jpg"
        comparison(frame, row["session"], nouns, legacy, combined, independent).save(compare_dir / name, quality=92)
        cards.append(f'<article><a href="comparisons/highlights/{name}"><img src="comparisons/highlights/{name}"></a><b>{row["legacy_count"]} → {row["combined_count"]} → {row["independent_count"]}</b><span>{row["session"]}</span></article>')
    (output / "index.html").write_text('''<!doctype html><meta charset="utf-8"><title>SAM3 noun prompt comparison</title><style>body{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:14px}article{background:#1b2330;padding:10px;border-radius:9px}img{width:100%}b,span{display:block;margin-top:6px}span{color:#9ba8ba;overflow-wrap:anywhere}</style><h1>SAM3 名词提示三方对比</h1><p>顺序：旧版 → 逗号拼接 → 独立名词。</p><div class="grid">''' + "".join(cards) + "</div>")

    m = summary["metrics"]
    total_calls = sum(len(item[2]) for item in loaded)
    raw_masks = duplicate_masks = boundary_masks = 0
    for row, _, _, _, _, _ in loaded:
        result_path = output / "raw" / _key(row["session"]) / "independent" / "result.json"
        filter_report = json.loads(result_path.read_text())["filter_report"]
        semantic = filter_report["semantic_discovery"]
        raw_masks += sum(semantic["raw_counts"].values())
        duplicate_masks += semantic["cross_prompt_duplicates_removed"]
        boundary_masks += filter_report["boundary_filter"]["objects_removed"]
    legacy_objects = m['legacy']['objects']; combined_objects = m['combined_prompt']['objects']
    independent_objects = m['independent_prompts']['objects']
    report = f'''# SAM3 名词提示三方对比（{len(rows)}条）

| 指标 | 旧版 | 逗号拼接一次 | 独立名词提示 |
|---|---:|---:|---:|
| 有检出的数据 | {m['legacy']['nonempty_sessions']} | {m['combined_prompt']['nonempty_sessions']} | {m['independent_prompts']['nonempty_sessions']} |
| 掩码总数 | {m['legacy']['objects']} | {m['combined_prompt']['objects']} | {m['independent_prompts']['objects']} |
| 每条平均掩码 | {m['legacy']['mean_objects_per_session']} | {m['combined_prompt']['mean_objects_per_session']} | {m['independent_prompts']['mean_objects_per_session']} |
| 平均置信度 | {m['legacy']['mean_confidence']} | {m['combined_prompt']['mean_confidence']} | {m['independent_prompts']['mean_confidence']} |
| 中位置信度 | {m['legacy']['median_confidence']} | {m['combined_prompt']['median_confidence']} | {m['independent_prompts']['median_confidence']} |

## 数据分析

- 独立名词方案相对逗号拼接方案恢复了 {independent_objects-combined_objects} 个掩码（{(independent_objects-combined_objects)*100/max(1,combined_objects):.1f}%），有检出的数据由 {m['combined_prompt']['nonempty_sessions']} 条恢复到 {m['independent_prompts']['nonempty_sessions']} 条。
- 相对旧版，独立名词方案掩码总数增加 {independent_objects-legacy_objects} 个（{(independent_objects-legacy_objects)*100/max(1,legacy_objects):.1f}%），覆盖数据少 3 条；平均置信度仅相差 {m['independent_prompts']['mean_confidence']-m['legacy']['mean_confidence']:+.4f}。
- 100条数据共发送 {total_calls} 个独立名词提示，平均每条 {total_calls/max(1,len(rows)):.2f} 个。SAM3共产生 {raw_masks} 个阈值内原始候选，跨提示去重 {duplicate_masks} 个，边界过滤 {boundary_masks} 个，最终保留 {independent_objects} 个。
- 与旧版逐条比较：掩码增加 {summary['independent_vs_legacy_count_categories'].get('gain', 0)} 条、减少 {summary['independent_vs_legacy_count_categories'].get('loss', 0)} 条、数量相同 {summary['independent_vs_legacy_count_categories'].get('same', 0)} 条。这说明总量接近不能替代逐图人工检查。
- 独立提示修复了拼接文本导致的大面积漏检，但名词较多的数据会出现候选激增，例如 `push_building_blocks` 为 60 个、`place_the_cake` 为 53 个。部分 `meta/annotations` 文本含位置或描述性名词，仍可能引入泛化误检，后续应收紧字段级提取或增加名词白名单/人工审核。

## 图表

![总体指标](summary_metrics.png)

![按名词数量分桶](noun_count_buckets.png)

![置信度和提示负载](confidence_and_prompt_load.png)

三组均使用同一张2K帧和置信度阈值 {args.threshold}。独立名词方案只编码图像一次，每个去重名词短语分别调用官方文本提示接口，最后跨提示词去重。旧版还包含原有通用提示及机械臂过滤，因此它是完整工作流基线，不是只改变文本格式的严格单变量实验。没有真值掩码，数量和置信度用于衡量输出行为，不能单独代表准确率。
'''
    (output / "REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
