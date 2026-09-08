#!/usr/bin/env python3
"""Compare three representative-mask policies on identical SAM3 raw outputs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
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
from project_paths import resolve_project_path
from semantic_prompts import semantic_noun_prompts
from reports.compare_semantic_prompts import _render
from reports.compare_single_prompt_nouns import _key

SOURCE = BASE_DIR / "reports/2026-09-03_object_meta_prompt_100/datasets.csv"
DATA_ROOT = BASE_DIR / "RoboCOIN_datasets"
OUTPUT = BASE_DIR / "reports/2026-09-04_mask_representative_three_way_100"
POLICIES = ("quality_consensus", "highest_confidence", "semantic_highest_confidence")
LABELS = ("Quality + consensus", "Highest confidence", "Non-object highest confidence")


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / max(1, union)


def touches_boundary(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def raw_inference(frame: Path, model, processor, prompts: list[str], threshold: float):
    """Encode once, then preserve every per-prompt post-NMS detection."""
    with Image.open(frame) as source:
        image = source.convert("RGB")
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        state = stage1._set_sam3_image(processor, image)
    processor.set_confidence_threshold(max(0.0, threshold - 1e-7))
    groups = []
    for prompt_index, prompt in enumerate(prompts):
        processor.reset_all_prompts(state)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            output = processor.set_text_prompt(state=state, prompt=prompt)
        detections = stage1._decode_grounding_detections(
            output, image.height, image.width, prompt, threshold
        )
        for rank, detection in enumerate(detections):
            detection["prompt_index"] = prompt_index
            detection["prompt_rank"] = rank
            detection["prompt_count"] = len(detections)
        groups.append((prompt, detections))
    return groups


def save_raw(directory: Path, session: str, frame: Path, prompts: list[str], groups) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for prompt, detections in groups:
        for item in detections:
            name = f"mask_{len(records):05d}.png"
            Image.fromarray((item["mask"] * 255).astype(np.uint8)).save(directory / name)
            records.append({
                "mask": name, "prompt": prompt, "score": float(item["score"]),
                "bbox": [float(value) for value in item["bbox"]],
                "prompt_index": int(item["prompt_index"]),
                "prompt_rank": int(item["prompt_rank"]),
                "prompt_count": int(item["prompt_count"]),
            })
    (directory / "result.json").write_text(json.dumps({
        "session": session, "frame": str(frame.resolve()), "prompts": prompts,
        "raw_detections": records,
    }, indent=2, ensure_ascii=False))


def load_raw(directory: Path):
    metadata = json.loads((directory / "result.json").read_text())
    groups = [(prompt, []) for prompt in metadata["prompts"]]
    by_prompt = {prompt: items for prompt, items in groups}
    for record in metadata["raw_detections"]:
        with Image.open(directory / record["mask"]) as source:
            mask = np.asarray(source.convert("L")) > 127
        by_prompt[record["prompt"]].append({**record, "mask": mask})
    return metadata, groups


def boundary_filter(items: list[dict]) -> tuple[list[dict], int]:
    kept = [item for item in items if not touches_boundary(item["mask"])]
    return kept, len(items) - len(kept)


def connected_components(items: list[dict]) -> list[list[int]]:
    parent = list(range(len(items)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if stage1._mask_duplicate(items[left]["mask"], items[right]["mask"])[0]:
                union(left, right)
    result = {}
    for index in range(len(items)):
        result.setdefault(root(index), []).append(index)
    return list(result.values())


def quality_consensus_policy(groups) -> tuple[list[dict], dict]:
    items = [item for _prompt, detections in groups for item in detections]
    selected = []
    cluster_details = []
    for component in connected_components(items):
        cluster = [items[index] for index in component]
        areas = np.asarray([max(1, int(item["mask"].sum())) for item in cluster], float)
        median_area = float(np.median(areas))
        candidates = []
        for index, item in enumerate(cluster):
            others = [mask_iou(item["mask"], other["mask"]) for other in cluster if other is not item]
            consensus = float(np.mean(others)) if others else 1.0
            area_agreement = math.exp(-abs(math.log(areas[index] / median_area)))
            boundary_quality = 0.0 if touches_boundary(item["mask"]) else 1.0
            non_object = 0.0 if item["prompt"] == "object" else 1.0
            phrase_specificity = min(1.0, len(item["prompt"].split()) / 3.0)
            count = max(1, int(item["prompt_count"]))
            rank_quality = 1.0 if count == 1 else 1.0 - int(item["prompt_rank"]) / (count - 1)
            quality = (
                0.40 * consensus + 0.20 * area_agreement + 0.20 * boundary_quality
                + 0.10 * non_object + 0.05 * phrase_specificity + 0.05 * rank_quality
            )
            candidates.append((quality, consensus, area_agreement, item))
        quality, consensus, area_agreement, representative = max(
            candidates, key=lambda value: (value[0], value[3]["score"])
        )
        chosen = dict(representative)
        chosen["selection_quality"] = quality
        chosen["cluster_consensus"] = consensus
        chosen["cluster_size"] = len(cluster)
        chosen["matched_prompts"] = [
            {"prompt": item["prompt"], "score": float(item["score"])} for item in cluster
        ]
        selected.append(chosen)
        cluster_details.append({
            "size": len(cluster), "quality": quality, "consensus": consensus,
            "area_agreement": area_agreement, "prompt": chosen["prompt"],
        })
    selected, boundary_removed = boundary_filter(selected)
    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected, {
        "duplicates": len(items) - len(cluster_details),
        "boundary_removed": boundary_removed,
        "clusters": cluster_details,
    }


def confidence_policy(groups, prefer_semantic: bool) -> tuple[list[dict], dict]:
    """Choose the highest-score representative, optionally preferring non-object candidates."""
    items = [item for _prompt, detections in groups for item in detections]
    selected = []
    cluster_details = []
    for component in connected_components(items):
        cluster = [items[index] for index in component]
        eligible = cluster
        if prefer_semantic:
            semantic = [item for item in cluster if item["prompt"] != "object"]
            if semantic:
                eligible = semantic
        representative = max(eligible, key=lambda item: float(item["score"]))
        chosen = dict(representative)
        chosen["cluster_size"] = len(cluster)
        chosen["matched_prompts"] = [
            {"prompt": item["prompt"], "score": float(item["score"])} for item in cluster
        ]
        selected.append(chosen)
        cluster_details.append({
            "size": len(cluster), "prompt": chosen["prompt"],
            "score": float(chosen["score"]),
        })
    selected, boundary_removed = boundary_filter(selected)
    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected, {
        "duplicates": len(items) - len(cluster_details),
        "boundary_removed": boundary_removed,
        "clusters": cluster_details,
    }


def apply_policies(groups):
    return {
        "quality_consensus": quality_consensus_policy(groups),
        "highest_confidence": confidence_policy(groups, prefer_semantic=False),
        "semantic_highest_confidence": confidence_policy(groups, prefer_semantic=True),
    }


def result_metrics(results: list[tuple[list[dict], dict]]) -> dict:
    groups = [items for items, _report in results]
    scores = [float(item["score"]) for group in groups for item in group]
    areas = [int(item["mask"].sum()) for group in groups for item in group]
    return {
        "sessions_with_masks": sum(bool(group) for group in groups),
        "total_masks": sum(len(group) for group in groups),
        "mean_masks": round(float(np.mean([len(group) for group in groups])), 3),
        "mean_confidence": round(float(np.mean(scores)), 4) if scores else None,
        "median_confidence": round(float(np.median(scores)), 4) if scores else None,
        "mean_mask_area": round(float(np.mean(areas)), 1) if areas else None,
        "non_object_representatives": sum(
            item["prompt"] != "object" for group in groups for item in group
        ),
        "boundary_removed": sum(report["boundary_removed"] for _items, report in results),
    }


def comparison_panel(frame: Path, session: str, results) -> Image.Image:
    colors = ((87, 154, 229), (237, 174, 73), (97, 190, 132))
    variants = [(label, results[name][0]) for name, label in zip(POLICIES, LABELS)]
    views = [_render(frame, [{**item, "prompt": item["prompt"]} for item in items],
                     [item["mask"] for item in items]) for _label, items in variants]
    width, height, header = max(x.width for x in views), max(x.height for x in views), 92
    canvas = Image.new("RGB", (3 * width, height + header), (18, 22, 29))
    draw = ImageDraw.Draw(canvas); draw.text((12, 8), session, fill="white")
    for index, ((label, items), view) in enumerate(zip(variants, views)):
        canvas.paste(view, (index * width + (width - view.width) // 2, header))
        avg = np.mean([item["score"] for item in items]) if items else float("nan")
        draw.text((index * width + 12, 38), f"{label} | masks={len(items)} | conf={avg:.3f}", fill=colors[index])
        semantic = sum(item["prompt"] != "object" for item in items)
        draw.text((index * width + 12, 64), f"specific-prompt reps={semantic}", fill=colors[index])
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=stage1.DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--highlights", type=int, default=30)
    args = parser.parse_args(); output = args.output.resolve()
    source_rows = list(csv.DictReader(SOURCE.open()))[:args.limit]
    pending = [row for row in source_rows if not (
        output / "raw" / _key(row["session"]) / "result.json"
    ).is_file()]
    model = processor = None; failures = []
    if pending:
        model, processor = stage1.load_sam3_detector(args.checkpoint, args.gpu)
    try:
        for index, row in enumerate(source_rows, 1):
            session = row["session"]; target = output / "raw" / _key(session)
            if (target / "result.json").is_file():
                print(f"[{index}/{len(source_rows)}] resume {session}", flush=True); continue
            try:
                old_meta = json.loads((
                    BASE_DIR / "reports/2026-09-03_object_meta_prompt_100/raw"
                    / _key(session) / "result.json"
                ).read_text())
                frame = resolve_project_path(old_meta["frame"])
                prompts = semantic_noun_prompts(DATA_ROOT / f"{session}.mp4", DATA_ROOT)
                groups = raw_inference(frame, model, processor, prompts, args.threshold)
                save_raw(target, session, frame, prompts, groups)
                print(f"[{index}/{len(source_rows)}] {session}: {sum(len(x) for _, x in groups)} raw", flush=True)
            except Exception as exc:
                failures.append({"session": session, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {session}: {exc}", flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    gc.collect(); torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model, processor; gc.collect(); torch.cuda.empty_cache()

    all_results = {name: [] for name in POLICIES}; sessions = []
    for row in source_rows:
        target = output / "raw" / _key(row["session"])
        if not (target / "result.json").is_file(): continue
        metadata, groups = load_raw(target); results = apply_policies(groups)
        sessions.append((metadata, results))
        for name in POLICIES: all_results[name].append(results[name])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_sessions": len(sessions), "failed_sessions": len(failures),
        "threshold": args.threshold,
        "metrics": {name: result_metrics(all_results[name]) for name in POLICIES},
        "quality_formula": "0.40 consensus + 0.20 area agreement + 0.20 non-boundary + 0.10 non-object + 0.05 phrase specificity + 0.05 within-prompt rank",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False))
    rows = []
    for metadata, results in sessions:
        row = {"session": metadata["session"], "prompt_count": len(metadata["prompts"]),
               "raw_count": len(metadata["raw_detections"])}
        for name in POLICIES:
            items, report = results[name]
            row[f"{name}_count"] = len(items)
            row[f"{name}_mean_confidence"] = round(float(np.mean([x["score"] for x in items])), 4) if items else None
            row[f"{name}_specific_representatives"] = sum(x["prompt"] != "object" for x in items)
            row[f"{name}_boundary_removed"] = report["boundary_removed"]
        rows.append(row)
    with (output / "datasets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows: writer.writeheader(); writer.writerows(rows)

    colors = ["#4c78a8", "#f2a93b", "#54a24b"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, key, title in zip(axes, ["total_masks", "mean_confidence", "non_object_representatives"],
                              ["Total kept masks", "Mean selected confidence", "Specific-prompt representatives"]):
        values = [summary["metrics"][name][key] for name in POLICIES]
        bars = ax.bar(LABELS, values, color=colors); ax.bar_label(bars, fmt="%.4g", padding=3)
        ax.set_title(title); ax.grid(axis="y", alpha=.25); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(output / "summary_metrics.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    score_groups = [[item["score"] for items, _ in all_results[name] for item in items] for name in POLICIES]
    axes[0].boxplot(score_groups, tick_labels=LABELS, showfliers=False); axes[0].set_ylim(0, 1)
    axes[0].set_title("Selected confidence distribution"); axes[0].grid(axis="y", alpha=.25)
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].scatter([row["highest_confidence_count"] for row in rows],
                    [row["semantic_highest_confidence_count"] for row in rows], alpha=.65)
    maximum = max([1, *[row["highest_confidence_count"] for row in rows],
                   *[row["semantic_highest_confidence_count"] for row in rows]])
    axes[1].plot([0, maximum], [0, maximum], "--", color="#d62728")
    axes[1].set_xlabel("Highest-confidence kept masks")
    axes[1].set_ylabel("Non-object-highest kept masks")
    axes[1].set_title("Per-session retained counts"); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "confidence_and_counts.png", dpi=180); plt.close(fig)

    ranked = sorted(sessions, key=lambda value: sum(
        len({item["prompt"] for item in value[1][name][0]})
        for name in POLICIES
    ), reverse=True)
    compare_dir = output / "comparisons"; compare_dir.mkdir(exist_ok=True); cards = []
    for rank, (metadata, results) in enumerate(ranked[:args.highlights], 1):
        name = f"{rank:02d}_{_key(metadata['session'])}.jpg"
        comparison_panel(resolve_project_path(metadata["frame"]), metadata["session"], results).save(compare_dir / name, quality=92)
        cards.append(f'<article><img src="comparisons/{name}"><span>{metadata["session"]}</span></article>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><style>body{background:#10141b;color:#eef;font:14px sans-serif;padding:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(700px,1fr));gap:14px}article{background:#1b2330;padding:10px}img{width:100%}span{display:block;margin-top:6px;color:#abc}</style><h1>Representative mask policies</h1><div class="grid">'+''.join(cards)+'</div>')
    if sessions:
        m = summary["metrics"]
        raw_total = sum(row["raw_count"] for row in rows)
        prompt_total = sum(row["prompt_count"] for row in rows)
        quality_specific_rate = 100 * m['quality_consensus']['non_object_representatives'] / max(1, m['quality_consensus']['total_masks'])
        highest_specific_rate = 100 * m['highest_confidence']['non_object_representatives'] / max(1, m['highest_confidence']['total_masks'])
        semantic_specific_rate = 100 * m['semantic_highest_confidence']['non_object_representatives'] / max(1, m['semantic_highest_confidence']['total_masks'])
        (output / "REPORT.md").write_text(f'''# 相近掩码代表选择三方案对比（{len(sessions)}条）

| 指标 | 质量共识 | 最高置信度 | 非 object 最高置信度 |
|---|---:|---:|---:|
| 最终掩码 | {m['quality_consensus']['total_masks']} | {m['highest_confidence']['total_masks']} | {m['semantic_highest_confidence']['total_masks']} |
| 平均置信度 | {m['quality_consensus']['mean_confidence']} | {m['highest_confidence']['mean_confidence']} | {m['semantic_highest_confidence']['mean_confidence']} |
| 具体提示词代表 | {m['quality_consensus']['non_object_representatives']} | {m['highest_confidence']['non_object_representatives']} | {m['semantic_highest_confidence']['non_object_representatives']} |
| 具体提示词代表占比 | {quality_specific_rate:.1f}% | {highest_specific_rate:.1f}% | {semantic_specific_rate:.1f}% |
| 边缘掩码删除 | {m['quality_consensus']['boundary_removed']} | {m['highest_confidence']['boundary_removed']} | {m['semantic_highest_confidence']['boundary_removed']} |

共执行 {prompt_total} 个提示词，得到 {raw_total} 个原始候选。三种方法共享完全相同的逐提示词原始输出，所以差异只来自相近掩码的处理策略。

- 质量共识不直接横比不同提示词的原始置信度，而主要依据组内 IoU 共识、面积一致性、是否触边、提示具体程度和提示词内部排名；原始置信度只用于平分时决胜。
- 最高置信度法在每个相似掩码组中直接选择分数最高者，不区分提示词。
- 非 object 最高置信度法只要组内存在具体提示词，就在具体提示词候选中选择最高分；仅在没有具体提示词时回退到 `object`。
- 本次没有真值掩码，置信度是模型自评而不是准确率。最终方法选择仍需结合 `index.html` 的逐图轮廓审核。

![总体指标](summary_metrics.png)

![置信度和逐条数量](confidence_and_counts.png)
''')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
