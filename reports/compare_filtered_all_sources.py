#!/usr/bin/env python3
"""Compare object+meta, legacy all-text nouns, and filtered all-source nouns."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import stage1_track_select as stage1
from hardware_profiles import PROFILES, default_profile_name, get_profile
from project_paths import resolve_project_path
from semantic_prompts import PROMPT_STRATEGY, semantic_noun_prompts
from reports.compare_semantic_prompts import _render
from reports.compare_single_prompt_nouns import _key, _load_result, _save_result

OBJECT_META_ROOT = BASE_DIR / "reports/2026-09-03_object_meta_prompt_100"
ALL_TEXT_ROOT = BASE_DIR / "reports/2026-09-03_noun_prompt_three_way_100"
DEFAULT_OUTPUT = BASE_DIR / "reports/2026-09-04_filtered_all_sources_100"
DATA_ROOT = BASE_DIR / "RoboCOIN_datasets"


def mean(items: list[dict]) -> float | None:
    return float(np.mean([float(item["score"]) for item in items])) if items else None


def metrics(groups: list[list[dict]]) -> dict:
    scores = [float(item["score"]) for group in groups for item in group]
    counts = [len(group) for group in groups]
    quantile = lambda value: round(float(np.quantile(scores, value)), 4) if scores else None
    return {
        "nonempty_sessions": sum(bool(group) for group in groups),
        "objects": sum(counts),
        "mean_objects": round(float(np.mean(counts)), 3),
        "median_objects": round(float(np.median(counts)), 3),
        "mean_confidence": round(float(np.mean(scores)), 4) if scores else None,
        "median_confidence": quantile(0.5),
        "confidence_p25": quantile(0.25),
        "confidence_p75": quantile(0.75),
    }


def panel(frame: Path, session: str, prompts: list[str], variants) -> Image.Image:
    colors = ["#54a24b", "#9467bd", "#e45756"]
    views = [
        _render(frame, [{**item, "prompt": label} for item in items],
                [item["mask"] for item in items])
        for label, items in variants
    ]
    width, height, header = max(x.width for x in views), max(x.height for x in views), 120
    canvas = Image.new("RGB", (3 * width, height + header), (18, 22, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), session, fill="white")
    for index, ((label, items), view) in enumerate(zip(variants, views)):
        canvas.paste(view, (index * width + (width - view.width) // 2, header))
        confidence = "N/A" if mean(items) is None else f"{mean(items):.3f}"
        draw.text(
            (index * width + 12, 34),
            f"{label} | masks={len(items)} | mean={confidence}",
            fill=colors[index],
        )
    prompt_text = ", ".join(prompts)
    if len(prompt_text) > 330:
        prompt_text = prompt_text[:327] + "..."
    draw.text((12, 66), f"new prompts ({len(prompts)}): {prompt_text}", fill=(195, 202, 214))
    draw.text((12, 94), "Same 100 sessions, 2K frames, threshold=0.2", fill=(195, 202, 214))
    return canvas


def write_prompt_analysis(source_rows: list[dict], output: Path) -> dict:
    """Write the comparison that does not require loading SAM3."""
    all_text_rows = {
        row["session"]: row
        for row in csv.DictReader((ALL_TEXT_ROOT / "datasets.csv").open())
    }
    rows = []
    for row in source_rows:
        session = row["session"]
        prompts = semantic_noun_prompts(DATA_ROOT / f"{session}.mp4", DATA_ROOT)
        rows.append({
            "session": session,
            "object_meta_prompts": int(row["prompt_count"]),
            "all_text_prompts": int(all_text_rows[session]["noun_count"]),
            "new_prompts": len(prompts),
            "new_modifier_prompts": 0,
            "new_prompt_text": ", ".join(prompts),
        })
    # Import the vocabulary directly here to keep the report's provenance clear.
    from semantic_prompts import COLOR_AND_SIZE_WORDS
    for row in rows:
        prompts = row["new_prompt_text"].split(", ")
        row["new_modifier_prompts"] = sum(
            prompt.split(" ", 1)[0] in COLOR_AND_SIZE_WORDS for prompt in prompts
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prompt_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    keys = ["object_meta_prompts", "all_text_prompts", "new_prompts"]
    labels = ["Object + meta", "All-text nouns", "Filtered all sources"]
    colors = ["#54a24b", "#9467bd", "#e45756"]
    summary = {
        "sessions": len(rows),
        "total": {key: sum(row[key] for row in rows) for key in keys},
        "average": {key: round(float(np.mean([row[key] for row in rows])), 2) for key in keys},
        "median": {key: round(float(np.median([row[key] for row in rows])), 2) for key in keys},
        "new_modifier_prompts": sum(row["new_modifier_prompts"] for row in rows),
    }
    (output / "prompt_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = axes[0].bar(labels, [summary["total"][key] for key in keys], color=colors)
    axes[0].bar_label(bars, padding=3); axes[0].set_title("Total text prompt calls (100 sessions)")
    axes[0].tick_params(axis="x", rotation=15); axes[0].grid(axis="y", alpha=0.25)
    axes[1].boxplot([[row[key] for row in rows] for key in keys], tick_labels=labels, showfliers=True)
    axes[1].set_title("Prompt count per session"); axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "prompt_load_comparison.png", dpi=180); plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=stage1.DEFAULT_CKPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--highlights", type=int, default=30)
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Generate noun-source statistics without loading SAM3",
    )
    parser.add_argument(
        "--hardware-profile", choices=tuple(PROFILES), default=default_profile_name()
    )
    args = parser.parse_args()
    stage1.RUNTIME_PROFILE = get_profile(args.hardware_profile)
    output = args.output.resolve()
    source_rows = list(csv.DictReader((OBJECT_META_ROOT / "datasets.csv").open()))
    prompt_summary = write_prompt_analysis(source_rows, output)
    print("Prompt analysis:", json.dumps(prompt_summary, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    pending = [
        row for row in source_rows
        if not (output / "raw" / _key(row["session"]) / "result.json").is_file()
    ]
    model = processor = None
    failures = []
    if pending:
        model, processor = stage1.load_sam3_detector(args.checkpoint, args.gpu)
    try:
        for index, row in enumerate(source_rows, 1):
            session = row["session"]
            target = output / "raw" / _key(session)
            if (target / "result.json").is_file():
                print(f"[{index}/100] resume {session}", flush=True)
                continue
            try:
                old_meta, _ = _load_result(OBJECT_META_ROOT / "raw" / _key(session))
                frame = resolve_project_path(old_meta["frame"])
                video = DATA_ROOT / f"{session}.mp4"
                prompts = semantic_noun_prompts(video, DATA_ROOT)
                detections, report = stage1.detect_first_frame(
                    frame, model, processor, prompts, args.threshold,
                    exclude_robot_arms=False,
                )
                _save_result(target, detections, {
                    "session": session,
                    "frame": str(frame.resolve()),
                    "prompts": prompts,
                    "strategy": PROMPT_STRATEGY,
                    "filter_report": report,
                })
                print(
                    f"[{index}/100] {session}: {len(prompts)} prompts -> "
                    f"{len(detections)} masks", flush=True,
                )
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
        new_dir = output / "raw" / _key(session)
        if not (new_dir / "result.json").is_file():
            continue
        new_meta, new = _load_result(new_dir)
        _, object_meta = _load_result(OBJECT_META_ROOT / "raw" / _key(session))
        _, all_text = _load_result(ALL_TEXT_ROOT / "raw" / _key(session) / "independent")
        frame = resolve_project_path(new_meta["frame"])
        record = {
            "session": session,
            "new_prompt_count": len(new_meta["prompts"]),
            "object_meta_count": len(object_meta),
            "all_text_count": len(all_text),
            "filtered_all_sources_count": len(new),
            "object_meta_mean_confidence": mean(object_meta),
            "all_text_mean_confidence": mean(all_text),
            "filtered_all_sources_mean_confidence": mean(new),
        }
        rows.append(record)
        loaded.append((record, frame, new_meta["prompts"], object_meta, all_text, new))

    names = ["object_meta", "all_text", "filtered_all_sources"]
    labels = ["Object + meta", "All-text nouns", "Filtered all sources"]
    groups = {name: [item[index] for item in loaded]
              for name, index in zip(names, [3, 4, 5])}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_sessions": len(rows),
        "failed_sessions": len(failures),
        "threshold": args.threshold,
        "strategy": PROMPT_STRATEGY,
        "metrics": {name: metrics(groups[name]) for name in names},
        "average_new_prompt_count": round(float(np.mean(
            [row["new_prompt_count"] for row in rows]
        )), 2) if rows else 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False))
    with (output / "datasets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)

    colors = ["#54a24b", "#9467bd", "#e45756"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, key, title in zip(
        axes,
        ["objects", "nonempty_sessions", "mean_confidence"],
        ["Total kept masks", "Sessions with masks", "Mean confidence"],
    ):
        values = [summary["metrics"][name][key] for name in names]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.3g", padding=3)
        ax.set_title(title); ax.grid(axis="y", alpha=0.25); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(output / "summary_metrics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scores = [[float(x["score"]) for group in groups[name] for x in group] for name in names]
    axes[0].boxplot(scores, tick_labels=labels, showfliers=False)
    axes[0].set_ylim(0, 1); axes[0].set_title("Confidence distribution")
    axes[0].grid(axis="y", alpha=0.25); axes[0].tick_params(axis="x", rotation=15)
    axes[1].scatter(
        [row["object_meta_count"] for row in rows],
        [row["filtered_all_sources_count"] for row in rows],
        alpha=0.65, color=colors[-1],
    )
    maximum = max([1, *[row["object_meta_count"] for row in rows],
                   *[row["filtered_all_sources_count"] for row in rows]])
    axes[1].plot([0, maximum], [0, maximum], "--", color="#333")
    axes[1].set_xlabel("Object + meta masks"); axes[1].set_ylabel("New method masks")
    axes[1].set_title("Per-session mask counts"); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "confidence_and_per_session.png", dpi=180); plt.close(fig)

    ranked = sorted(
        loaded,
        key=lambda item: abs(item[0]["filtered_all_sources_count"] - item[0]["object_meta_count"]),
        reverse=True,
    )
    compare_dir = output / "comparisons" / "highlights"
    compare_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for rank, (row, frame, prompts, object_meta, all_text, new) in enumerate(
        ranked[:args.highlights], 1
    ):
        name = f"{rank:02d}_{_key(row['session'])}.jpg"
        panel(frame, row["session"], prompts, list(zip(
            labels, [object_meta, all_text, new]
        ))).save(compare_dir / name, quality=92)
        cards.append(
            f'<article><a href="comparisons/highlights/{name}"><img '
            f'src="comparisons/highlights/{name}"></a><b>'
            f'{len(object_meta)} / {len(all_text)} / {len(new)}</b>'
            f'<span>{row["session"]}</span></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>'
        'body{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(720px,1fr));gap:14px}'
        'article{background:#1b2330;padding:10px;border-radius:9px}img{width:100%}'
        'b,span{display:block;margin-top:6px}span{color:#9ba8ba}</style>'
        '<h1>Three noun-source strategies</h1><div class="grid">' + "".join(cards) + "</div>"
    )

    if rows:
        old, broad, new = (summary["metrics"][name] for name in names)
        (output / "REPORT.md").write_text(f'''# SAM3 三种名词来源对比（{len(rows)}条）

| 指标 | object + meta | all-text nouns | 新版全来源过滤 |
|---|---:|---:|---:|
| 有检出数据 | {old['nonempty_sessions']} | {broad['nonempty_sessions']} | {new['nonempty_sessions']} |
| 掩码总数 | {old['objects']} | {broad['objects']} | {new['objects']} |
| 每条平均掩码 | {old['mean_objects']} | {broad['mean_objects']} | {new['mean_objects']} |
| 平均置信度 | {old['mean_confidence']} | {broad['mean_confidence']} | {new['mean_confidence']} |
| 中位置信度 | {old['median_confidence']} | {broad['median_confidence']} | {new['median_confidence']} |

新版使用数据集名称、当前 episode 的 meta/scene，以及 subtask 文本；仅保留物理实体，颜色和尺寸作为物体修饰词，同时保留基础物体名词。平均每条使用 {summary['average_new_prompt_count']} 个提示词。

![总体对比](summary_metrics.png)

![置信度与逐条数量](confidence_and_per_session.png)

三组使用同一批100条数据、相同2K帧和0.2阈值。没有真值掩码，因此数量和置信度只能衡量输出行为，最终准确率仍需逐图审核。详细掩码对比见 `index.html`。
''')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
