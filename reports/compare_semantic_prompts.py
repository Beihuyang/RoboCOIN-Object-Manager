#!/usr/bin/env python3
"""Controlled SAM3 A/B: `object` versus `object` plus dataset-name nouns."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from project_paths import resolve_project_path
from stage1_track_select import (
    DEFAULT_CKPT,
    DEFAULT_ROBOT_ARM_OVERLAP,
    DEFAULT_ROBOT_ARM_THRESHOLD,
    detect_first_frame,
    load_sam3_detector,
    semantic_discovery_prompts,
)


DEFAULT_TRACKS = BASE_DIR / "objects" / "tracks"
DEFAULT_OUTPUT = BASE_DIR / "reports" / "2026-08-31_semantic_prompts_ab_v3"
PALETTE = (
    (255, 91, 91), (83, 181, 255), (91, 225, 139), (255, 196, 87),
    (191, 121, 255), (255, 123, 211), (80, 221, 218), (211, 229, 91),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _result_key(session: str) -> str:
    return hashlib.sha1(session.encode()).hexdigest()[:12]


def _write_result(path: Path, payload: dict, detections: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    objects = []
    for index, detection in enumerate(detections):
        mask_name = f"mask_{index:04d}.png"
        Image.fromarray((detection["mask"] * 255).astype(np.uint8)).save(path / mask_name)
        objects.append({
            "mask": mask_name,
            "bbox": [float(value) for value in detection["bbox"]],
            "score": float(detection["score"]),
            "prompt": detection["prompt"],
            "matched_prompts": detection.get("matched_prompts", []),
        })
    (path / "result.json").write_text(json.dumps({
        **payload, "objects": objects,
    }, indent=2, ensure_ascii=False))


def _load_result(path: Path) -> dict:
    payload = json.loads((path / "result.json").read_text())
    masks = []
    for item in payload["objects"]:
        with Image.open(path / item["mask"]) as source:
            masks.append(np.asarray(source.convert("L")) > 127)
    payload["masks"] = masks
    return payload


def _mean_score(objects: list[dict]) -> float | None:
    return float(np.mean([float(item["score"]) for item in objects])) if objects else None


def _render(frame_path: Path, objects: list[dict], masks: list[np.ndarray], width=900) -> Image.Image:
    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, width / image.width, 700 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = np.asarray(image.resize(size, Image.Resampling.LANCZOS)).copy()
    tinted = canvas.copy()
    labels = []
    for index, (item, mask) in enumerate(zip(objects, masks)):
        resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        color = PALETTE[index % len(PALETTE)]
        tinted[resized] = color
        edge = cv2.morphologyEx(resized.astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)).astype(bool)
        canvas[edge] = color
        x1, y1, _, _ = _bbox(resized)
        labels.append((x1, y1, f"{item['prompt']} {float(item['score']):.2f}"))
    canvas = np.where(np.any(tinted != canvas, axis=2, keepdims=True),
                      (canvas * 0.68 + tinted * 0.32).astype(np.uint8), canvas)
    output = Image.fromarray(canvas)
    draw = ImageDraw.Draw(output)
    for x, y, label in labels:
        text_box = draw.textbbox((x + 4, y + 1), label)
        draw.rectangle((x, y, text_box[2] + 4, y + 17), fill=(0, 0, 0))
        draw.text((x + 4, y + 1), label, fill=(255, 255, 255))
    return output


def _comparison(result: dict) -> Image.Image:
    baseline_indices = [
        index for index, item in enumerate(result["objects"])
        if item["prompt"] == result["prompts"][0]
    ]
    before_objects = [result["objects"][index] for index in baseline_indices]
    before_masks = [result["masks"][index] for index in baseline_indices]
    left = _render(Path(result["frame"]), before_objects, before_masks)
    right = _render(Path(result["frame"]), result["objects"], result["masks"])
    panel_w, panel_h = max(left.width, right.width), max(left.height, right.height)
    header = 108
    canvas = Image.new("RGB", (panel_w * 2, panel_h + header), (18, 22, 29))
    canvas.paste(left, ((panel_w - left.width) // 2, header))
    canvas.paste(right, (panel_w + (panel_w - right.width) // 2, header))
    draw = ImageDraw.Draw(canvas)
    before_score = _mean_score(before_objects)
    after_score = _mean_score(result["objects"])
    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    draw.text((12, 8), result["session"], fill=(235, 241, 250))
    draw.text((12, 32), f"BEFORE object | masks={len(before_objects)} | mean confidence={fmt(before_score)}", fill=(160, 196, 255))
    draw.text((panel_w + 12, 32), f"AFTER object + dataset nouns | masks={len(result['objects'])} | mean confidence={fmt(after_score)}", fill=(144, 232, 180))
    draw.text((12, 56), "semantic prompts: " + ", ".join(result["prompts"][1:]), fill=(192, 198, 208))
    draw.text((12, 81), "Label = source prompt + confidence; semantic additions do not replace object masks", fill=(192, 198, 208))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--highlight-count", type=int, default=12)
    parser.add_argument("--no-robot-filter", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    raw_dir = output / "raw"
    manifests = sorted(args.tracks_root.rglob("initial_sam3_sr_2k/manifest.json"))
    eligible, excluded = [], []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        session = manifest_path.parent.parent.relative_to(args.tracks_root).as_posix()
        if int(manifest.get("revision", 0)) > 0:
            excluded.append({"session": session, "revision": manifest.get("revision", 0)})
            continue
        frame = resolve_project_path(manifest["frame"])
        if not frame.is_file():
            excluded.append({"session": session, "revision": 0, "reason": "missing frame"})
            continue
        eligible.append((session, manifest, frame))
    if args.limit:
        eligible = eligible[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    pending = [item for item in eligible if not (raw_dir / _result_key(item[0]) / "result.json").is_file()]
    detector = processor = None
    if pending:
        print(f"Loading SAM3 once; pending {len(pending)}/{len(eligible)}", flush=True)
        detector, processor = load_sam3_detector(args.checkpoint, args.gpu)
    try:
        for completed, (session, manifest, frame) in enumerate(eligible, start=1):
            result_dir = raw_dir / _result_key(session)
            if (result_dir / "result.json").is_file():
                print(f"[{completed}/{len(eligible)}] resume {session}", flush=True)
                continue
            video = BASE_DIR / "RoboCOIN_datasets" / f"{session}.mp4"
            prompts = semantic_discovery_prompts(video, "object")
            robot_filter = manifest.get("robot_arm_filter", {})
            try:
                detections, filter_report = detect_first_frame(
                    frame, detector, processor, prompts, args.threshold,
                    exclude_robot_arms=(
                        not args.no_robot_filter
                        and bool(robot_filter.get("enabled", True))
                    ),
                    robot_arm_threshold=float(robot_filter.get(
                        "threshold", DEFAULT_ROBOT_ARM_THRESHOLD
                    )),
                    robot_arm_overlap=float(robot_filter.get(
                        "overlap_threshold", DEFAULT_ROBOT_ARM_OVERLAP
                    )),
                )
                _write_result(result_dir, {
                    "session": session,
                    "frame": str(frame.resolve()),
                    "prompts": prompts,
                    "threshold": args.threshold,
                    "filter_report": filter_report,
                }, detections)
                base_count = sum(item["prompt"] == prompts[0] for item in detections)
                print(f"[{completed}/{len(eligible)}] {session}: {base_count} -> {len(detections)}", flush=True)
            except Exception as exc:
                failures.append({"session": session, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{completed}/{len(eligible)}] FAILED {session}: {exc}", flush=True)
                if isinstance(exc, torch.OutOfMemoryError):
                    gc.collect(); torch.cuda.empty_cache()
    finally:
        if detector is not None:
            del processor, detector
            gc.collect(); torch.cuda.empty_cache()

    results = []
    prompt_additions = Counter()
    for session, _, _ in eligible:
        result_path = raw_dir / _result_key(session)
        if not (result_path / "result.json").is_file():
            continue
        result = _load_result(result_path)
        base = [item for item in result["objects"] if item["prompt"] == result["prompts"][0]]
        added = [item for item in result["objects"] if item["prompt"] != result["prompts"][0]]
        prompt_additions.update(item["prompt"] for item in added)
        results.append((result, {
            "session": session,
            "prompts": " | ".join(result["prompts"]),
            "prompt_count": len(result["prompts"]),
            "before_count": len(base),
            "after_count": len(result["objects"]),
            "added_count": len(added),
            "before_mean_score": _mean_score(base),
            "after_mean_score": _mean_score(result["objects"]),
            "semantic_added_mean_score": _mean_score(added),
            "category": (
                "newly_detected" if not base and added
                else "large_addition" if len(added) >= 6
                else "moderate_addition" if added
                else "unchanged"
            ),
        }))
    changed = [(result, row) for result, row in results if row["added_count"] > 0]
    all_dir, highlight_dir = output / "comparisons" / "all", output / "comparisons" / "highlights"
    all_dir.mkdir(parents=True, exist_ok=True); highlight_dir.mkdir(parents=True, exist_ok=True)
    for index, (result, row) in enumerate(changed, start=1):
        name = f"{result['session'].split('/')[0]}__{_result_key(result['session'])}.jpg"
        image_path = all_dir / name
        _comparison(result).save(image_path, quality=92, optimize=True)
        row["comparison"] = image_path.relative_to(output).as_posix()
        if index % 25 == 0:
            print(f"Exported comparisons {index}/{len(changed)}", flush=True)
    highlights = []
    categories = ("newly_detected", "moderate_addition", "large_addition")
    per_category = max(1, args.highlight_count // len(categories))
    for category in categories:
        candidates = [pair for pair in changed if pair[1]["category"] == category]
        highlights.extend(sorted(
            candidates,
            key=lambda pair: (
                pair[1]["added_count"],
                pair[1]["semantic_added_mean_score"] or 0,
            ),
            reverse=True,
        )[:per_category])
    if len(highlights) < args.highlight_count:
        selected = {pair[1]["session"] for pair in highlights}
        remainder = [pair for pair in changed if pair[1]["session"] not in selected]
        highlights.extend(sorted(
            remainder, key=lambda pair: pair[1]["semantic_added_mean_score"] or 0,
            reverse=True,
        )[:args.highlight_count - len(highlights)])
    for rank, (_, row) in enumerate(highlights, start=1):
        source = output / row["comparison"]
        target = highlight_dir / f"{rank:02d}_{source.name}"
        shutil.copy2(source, target)
        row["highlight"] = target.relative_to(output).as_posix()

    rows = [row for _, row in results]
    for row in rows:
        row.setdefault("comparison", "")
        row.setdefault("highlight", "")
    before_total = sum(row["before_count"] for row in rows)
    after_total = sum(row["after_count"] for row in rows)
    before_nonempty = sum(row["before_count"] > 0 for row in rows)
    after_nonempty = sum(row["after_count"] > 0 for row in rows)
    weighted = lambda key, count_key: (
        sum((row[key] or 0) * row[count_key] for row in rows) /
        max(1, sum(row[count_key] for row in rows))
    )
    summary = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "change": "SAM3 object prompt + nouns/compound nouns extracted from dataset name",
        "controlled_design": "one shared image encoding; object masks retained first; semantic prompts only add cross-prompt non-duplicates",
        "eligible_sessions": len(eligible),
        "completed_sessions": len(rows),
        "excluded_manual_revision_sessions": len(excluded),
        "failed_sessions": len(failures),
        "before": {"nonempty_sessions": before_nonempty, "objects": before_total,
                   "mean_confidence": round(weighted("before_mean_score", "before_count"), 4)},
        "after": {"nonempty_sessions": after_nonempty, "objects": after_total,
                  "mean_confidence": round(weighted("after_mean_score", "after_count"), 4)},
        "delta": {
            "nonempty_sessions": after_nonempty - before_nonempty,
            "coverage_percentage_points": round((after_nonempty - before_nonempty) * 100 / max(1, len(rows)), 2),
            "objects": after_total - before_total,
            "sessions_with_additions": len(changed),
        },
        "average_prompt_count": round(float(np.mean([row["prompt_count"] for row in rows])), 2) if rows else 0,
        "top_semantic_prompts_by_added_masks": prompt_additions.most_common(30),
        "comparison_images": {"all_changed": len(changed), "highlights": len(highlights)},
        "change_categories": dict(Counter(row["category"] for row in rows)),
        "limitations": [
            "There are no ground-truth masks, so added candidates measure recall opportunity, not proven accuracy.",
            "SAM3 confidence is conditioned on each text prompt and is not calibrated across prompts.",
            "Sessions with manifest revision > 0 are excluded to avoid manual-edit contamination.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    fields = list(rows[0]) if rows else []
    with (output / "datasets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    for name, data in (("excluded_manual_revisions.json", excluded), ("failures.json", failures)):
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    with (output / "human_review.csv").open("w", newline="") as stream:
        fields = ["session", "category", "comparison", "verdict", "false_positive_after", "false_negative_after", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for _, row in highlights:
            writer.writerow({
                "session": row["session"], "category": row["category"],
                "comparison": row.get("highlight", row.get("comparison", "")),
                "verdict": "", "false_positive_after": "",
                "false_negative_after": "", "notes": "",
            })
    cards = "".join(
        f'<article><a href="{html.escape(row.get("highlight", row.get("comparison", "")))}"><img src="{html.escape(row.get("highlight", row.get("comparison", "")))}" loading="lazy"></a><b>+{row["added_count"]} masks</b><span>{html.escape(row["session"])}</span></article>'
        for _, row in highlights
    )
    all_cards = "".join(
        f'<article><a href="{html.escape(row.get("comparison", ""))}"><img src="{html.escape(row.get("comparison", ""))}" loading="lazy"></a><b>+{row["added_count"]} masks</b><span>{html.escape(row["session"])}</span></article>'
        for _, row in changed
    )
    (output / "index.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>SAM3 semantic prompt A/B</title>
<style>body{{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}}a{{color:#78b9ff}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}article{{background:#1b2330;padding:10px;border-radius:9px}}img{{width:100%}}b,span{{display:block;margin-top:6px}}span{{color:#9ba8ba;overflow-wrap:anywhere}}</style>
<h1>SAM3 数据集名语义提示 A/B</h1><p>完成 {len(rows)} 条；物体数 {before_total} → {after_total}；新增 {after_total-before_total}；<a href="all_comparisons.html">全部变化图</a>。</p><div class="grid">{cards}</div>''')
    (output / "all_comparisons.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>SAM3 semantic prompt A/B all</title>
<style>body{{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}}a{{color:#78b9ff}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}article{{background:#1b2330;padding:10px;border-radius:9px}}img{{width:100%}}b,span{{display:block;margin-top:6px}}span{{color:#9ba8ba;overflow-wrap:anywhere}}</style>
<h1>全部有语义新增候选的对比</h1><p><a href="index.html">返回精选</a> · 共 {len(changed)} 张</p><div class="grid">{all_cards}</div>''')
    report = f"""# SAM3 数据集名称语义提示 A/B 报告

- 可控样本：{len(rows)} 条（排除人工 revision {len(excluded)} 条，失败 {len(failures)} 条）
- 有掩码数据：{before_nonempty} → {after_nonempty}（覆盖率 {before_nonempty / max(1, len(rows)) * 100:.2f}% → {after_nonempty / max(1, len(rows)) * 100:.2f}%，{summary['delta']['coverage_percentage_points']:+.2f} 个百分点）
- 掩码总数：{before_total} → {after_total}（{after_total-before_total:+d}）
- 按掩码加权的平均置信度：{summary['before']['mean_confidence']:.4f} → {summary['after']['mean_confidence']:.4f}
- 获得语义新增候选的数据：{len(changed)} 条
- 平均提示词数量：{summary['average_prompt_count']}
- 全部变化对比图：{len(changed)} 张；精选：{len(highlights)} 张

## 解释

修改前只用 `object`。修改后先保留全部 `object` 掩码，再使用数据集名称中的短名词和复合名词补检；跨提示 IoU/包含率去重后才加入，因此不会用不可直接比较的语义置信度替换原掩码。

这里的平均置信度是“所有掩码分数之和 / 掩码总数”，不是逐数据平均。当前没有人工真值，因此新增掩码表示“可能补回的漏检”，不等于全部正确，也不能据此计算准确率；请结合精选图和 `datasets.csv` 做人工判断。不同文本提示下的置信度不完全可比。
"""
    (output / "REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
