#!/usr/bin/env python3
"""Compare original-resolution and 2K-super-resolved SAM3 discovery caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS = BASE_DIR / "objects" / "tracks"
DEFAULT_OUTPUT = BASE_DIR / "reports" / "2026-08-27_super_resolution_ab_controlled"
PALETTE = (
    (255, 91, 91), (83, 181, 255), (91, 225, 139), (255, 196, 87),
    (191, 121, 255), (255, 123, 211), (80, 221, 218), (211, 229, 91),
)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def load_cache(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    masks = []
    objects = manifest.get("objects", [])
    for item in objects:
        mask_path = manifest_path.parent / Path(item["mask"]).name
        with Image.open(mask_path) as source:
            masks.append(np.asarray(source.convert("L")) > 127)
    return {
        "manifest": manifest,
        "objects": objects,
        "masks": masks,
        "frame": project_path(manifest["frame"]),
    }


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def mask_matches(old_masks: list[np.ndarray], new_masks: list[np.ndarray]) -> dict:
    if not old_masks and not new_masks:
        return {"matched": 0, "mean_matched_iou": None, "penalized_iou": 1.0,
                "matches_iou_50": 0, "pairs": []}
    if not old_masks or not new_masks:
        return {"matched": 0, "mean_matched_iou": None, "penalized_iou": 0.0,
                "matches_iou_50": 0, "pairs": []}
    target_h, target_w = new_masks[0].shape
    resized_old = [
        cv2.resize(mask.astype(np.uint8), (target_w, target_h),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
        if mask.shape != (target_h, target_w) else mask
        for mask in old_masks
    ]
    old_boxes = [mask_bbox(mask) for mask in resized_old]
    new_boxes = [mask_bbox(mask) for mask in new_masks]
    old_areas = [int(mask.sum()) for mask in resized_old]
    new_areas = [int(mask.sum()) for mask in new_masks]
    candidates = []
    for old_index, old_mask in enumerate(resized_old):
        ox1, oy1, ox2, oy2 = old_boxes[old_index]
        for new_index, new_mask in enumerate(new_masks):
            nx1, ny1, nx2, ny2 = new_boxes[new_index]
            x1, y1, x2, y2 = max(ox1, nx1), max(oy1, ny1), min(ox2, nx2), min(oy2, ny2)
            if x2 <= x1 or y2 <= y1:
                continue
            intersection = int(np.logical_and(
                old_mask[y1:y2, x1:x2], new_mask[y1:y2, x1:x2]
            ).sum())
            if not intersection:
                continue
            union = old_areas[old_index] + new_areas[new_index] - intersection
            candidates.append((intersection / max(1, union), old_index, new_index))
    used_old, used_new, matches, pairs = set(), set(), [], []
    for iou, old_index, new_index in sorted(candidates, reverse=True):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        matches.append(iou)
        pairs.append({"old_index": old_index, "new_index": new_index, "iou": iou})
    return {
        "matched": len(matches),
        "mean_matched_iou": round(float(np.mean(matches)), 4) if matches else None,
        "penalized_iou": round(sum(matches) / max(len(old_masks), len(new_masks)), 4),
        "matches_iou_50": sum(value >= 0.5 for value in matches),
        "pairs": pairs,
    }


def render_overlay(frame_path: Path, masks: list[np.ndarray], objects: list[dict],
                   color_indices: list[int], width: int = 900) -> Image.Image:
    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, width / image.width, 700 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    canvas = np.asarray(image).copy()
    overlay = canvas.copy()
    draw_labels = []
    for index, mask in enumerate(masks):
        display_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        color = PALETTE[color_indices[index] % len(PALETTE)]
        overlay[display_mask] = color
        boundary = cv2.morphologyEx(
            display_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        ).astype(bool)
        canvas[boundary] = color
        x1, y1, _, _ = mask_bbox(display_mask)
        object_id = objects[index].get("object_id", index)
        score = float(objects[index].get("score", 0))
        draw_labels.append((x1, y1, f"#{object_id} {score:.2f}"))
    canvas = np.where(np.any(overlay != canvas, axis=2, keepdims=True),
                      (canvas * 0.68 + overlay * 0.32).astype(np.uint8), canvas)
    output = Image.fromarray(canvas)
    draw = ImageDraw.Draw(output)
    for x, y, label in draw_labels:
        label_box = draw.textbbox((x + 4, y + 1), label)
        draw.rectangle((x, y, label_box[2] + 4, y + 17), fill=(0, 0, 0))
        draw.text((x + 4, y + 1), label, fill=(255, 255, 255))
    return output


def mean_confidence(cache: dict) -> float | None:
    scores = [float(item.get("score", 0)) for item in cache["objects"]]
    return float(np.mean(scores)) if scores else None


def display_confidence(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def object_mean_score(objects: list[dict]) -> float | None:
    scores = [float(item.get("score", 0)) for item in objects]
    return round(float(np.mean(scores)), 4) if scores else None


def comparison_image(old: dict, new: dict, title: str, match_result: dict) -> Image.Image:
    old_colors = list(range(len(old["masks"])))
    new_colors = [len(old_colors) + index for index in range(len(new["masks"]))]
    for pair in match_result["pairs"]:
        new_colors[pair["new_index"]] = old_colors[pair["old_index"]]
    left = render_overlay(old["frame"], old["masks"], old["objects"], old_colors)
    right = render_overlay(new["frame"], new["masks"], new["objects"], new_colors)
    panel_w, panel_h = max(left.width, right.width), max(left.height, right.height)
    header = 98
    result = Image.new("RGB", (panel_w * 2, panel_h + header), (18, 22, 29))
    result.paste(left, ((panel_w - left.width) // 2, header))
    result.paste(right, (panel_w + (panel_w - right.width) // 2, header))
    draw = ImageDraw.Draw(result)
    old_confidence, new_confidence = mean_confidence(old), mean_confidence(new)
    confidence_delta = (
        f"{new_confidence-old_confidence:+.3f}"
        if old_confidence is not None and new_confidence is not None else "N/A"
    )
    draw.text((12, 8), title, fill=(235, 241, 250))
    draw.text((12, 31), f"BEFORE original | masks={len(old['masks'])} | mean confidence(all masks)={display_confidence(old_confidence)}", fill=(160, 196, 255))
    draw.text((panel_w + 12, 31), f"AFTER RealESRGAN 2K | masks={len(new['masks'])} | mean confidence(all masks)={display_confidence(new_confidence)} | delta={confidence_delta}", fill=(144, 232, 180))
    draw.text((12, 55), f"matched={match_result['matched']}  mean IoU={match_result['mean_matched_iou']}  penalized IoU={match_result['penalized_iou']}", fill=(192, 198, 208))
    draw.text((12, 77), "Mask label: #object_id confidence | matched masks use the same color", fill=(192, 198, 208))
    return result


def category_for(old_count: int, new_count: int, penalized_iou: float,
                 min_count_delta: int, mask_change_iou: float) -> str | None:
    if old_count == 0 < new_count:
        return "newly_detected"
    if new_count == 0 < old_count:
        return "lost_detection"
    delta = new_count - old_count
    if delta >= min_count_delta:
        return "count_gain"
    if delta <= -min_count_delta:
        return "count_loss"
    if old_count and new_count and penalized_iou < mask_change_iou:
        return "mask_changed"
    return None


def select_highlights(rows: list[dict]) -> list[dict]:
    """Pick a small, stratified set of the clearest changes."""
    selections = []
    rules = (
        ("newly_detected", 3, lambda row: (row["new_mean_score"] or 0, row["new_count"]), True),
        ("lost_detection", 3, lambda row: (row["old_mean_score"] or 0, row["old_count"]), True),
        ("count_gain", 2, lambda row: (row["count_delta"], row["new_mean_score"] or 0), True),
        ("count_loss", 2, lambda row: (-row["count_delta"], row["old_mean_score"] or 0), True),
        ("mask_changed", 2, lambda row: row["penalized_iou"], False),
    )
    for category, limit, key, reverse in rules:
        candidates = [row for row in rows if row["category"] == category]
        selections.extend(sorted(candidates, key=key, reverse=reverse)[:limit])
    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-count-delta", type=int, default=2)
    parser.add_argument("--mask-change-iou", type=float, default=0.35)
    parser.add_argument(
        "--include-manual-revisions", action="store_true",
        help="include sessions whose old or new manifest revision is greater than zero",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    comparisons = output / "comparisons"
    rows, excluded_manual = [], []
    category_counts = Counter()
    paired = sorted(args.tracks_root.rglob("initial_sam3_sr_2k/manifest.json"))
    for completed, new_manifest in enumerate(paired, start=1):
        session_dir = new_manifest.parent.parent
        old_manifest = session_dir / "initial_sam3" / "manifest.json"
        if not old_manifest.is_file():
            continue
        old, new = load_cache(old_manifest), load_cache(new_manifest)
        old_revision = int(old["manifest"].get("revision", 0))
        new_revision = int(new["manifest"].get("revision", 0))
        if not args.include_manual_revisions and (old_revision > 0 or new_revision > 0):
            excluded_manual.append({
                "session": session_dir.relative_to(args.tracks_root).as_posix(),
                "old_revision": old_revision,
                "new_revision": new_revision,
            })
            continue
        match_result = mask_matches(old["masks"], new["masks"])
        match_result.pop("pairs")
        metrics = match_result
        old_count, new_count = len(old["masks"]), len(new["masks"])
        session_key = session_dir.relative_to(args.tracks_root).as_posix()
        category = category_for(
            old_count, new_count, metrics["penalized_iou"],
            args.min_count_delta, args.mask_change_iou,
        )
        row = {
            "session": session_key,
            "old_count": old_count,
            "new_count": new_count,
            "count_delta": new_count - old_count,
            "old_mean_score": object_mean_score(old["objects"]),
            "new_mean_score": object_mean_score(new["objects"]),
            "old_review_hidden_included": sum(bool(item.get("review_hidden", False)) for item in old["objects"]),
            "new_review_hidden_included": sum(bool(item.get("review_hidden", False)) for item in new["objects"]),
            "old_revision": old_revision,
            "new_revision": new_revision,
            "category": category or "not_exported",
            "highlight_rank": "",
            "highlight_comparison": "",
            "comparison": "",
            **metrics,
        }
        if category:
            category_counts[category] += 1
        rows.append(row)
        if completed % 50 == 0:
            print(f"Compared {completed}/{len(paired)}", flush=True)

    highlights = select_highlights(rows)
    all_dir = comparisons / "all"
    highlight_dir = comparisons / "highlights"
    all_dir.mkdir(parents=True, exist_ok=True)
    highlight_dir.mkdir(parents=True, exist_ok=True)
    highlight_ranks = {
        row["session"]: rank for rank, row in enumerate(highlights, start=1)
    }
    all_cards, highlight_cards = [], []
    significant_rows = [row for row in rows if row["category"] != "not_exported"]
    for completed, row in enumerate(significant_rows, start=1):
        session_dir = args.tracks_root / row["session"]
        old = load_cache(session_dir / "initial_sam3" / "manifest.json")
        new = load_cache(session_dir / "initial_sam3_sr_2k" / "manifest.json")
        match_result = mask_matches(old["masks"], new["masks"])
        digest = hashlib.sha1(row["session"].encode()).hexdigest()[:8]
        name = f"{row['session'].split('/')[0]}__{digest}.jpg"
        category_dir = all_dir / row["category"]
        category_dir.mkdir(exist_ok=True)
        relative_image = Path("comparisons") / "all" / row["category"] / name
        image = comparison_image(old, new, row["session"], match_result)
        image.save(output / relative_image, quality=92, optimize=True)
        row["comparison"] = relative_image.as_posix()
        all_cards.append((row["category"], row["session"], relative_image.as_posix(), row))
        rank = highlight_ranks.get(row["session"])
        if rank is not None:
            highlight_name = f"{rank:02d}_{row['category']}__{name}"
            highlight_relative = Path("comparisons") / "highlights" / highlight_name
            shutil.copy2(output / relative_image, output / highlight_relative)
            row["highlight_rank"] = rank
            row["highlight_comparison"] = highlight_relative.as_posix()
            highlight_cards.append((
                row["category"], row["session"], highlight_relative.as_posix(), row
            ))
        if completed % 25 == 0:
            print(f"Exported {completed}/{len(significant_rows)} comparisons", flush=True)

    old_nonempty = sum(row["old_count"] > 0 for row in rows)
    new_nonempty = sum(row["new_count"] > 0 for row in rows)
    old_objects = sum(row["old_count"] for row in rows)
    new_objects = sum(row["new_count"] for row in rows)
    both_nonempty = [row for row in rows if row["old_count"] and row["new_count"]]
    old_only = sum(row["old_count"] > 0 and row["new_count"] == 0 for row in rows)
    new_only = sum(row["new_count"] > 0 and row["old_count"] == 0 for row in rows)
    old_weighted_score = sum(
        (row["old_mean_score"] or 0) * row["old_count"] for row in rows
    ) / max(1, old_objects)
    new_weighted_score = sum(
        (row["new_mean_score"] or 0) * row["new_count"] for row in rows
    ) / max(1, new_objects)
    old_hidden_included = sum(row["old_review_hidden_included"] for row in rows)
    new_hidden_included = sum(row["new_review_hidden_included"] for row in rows)
    summary = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "change": "RealESRGAN_x2plus preprocessing to 2K before SAM3 discovery",
        "source_sessions_considered": len(rows) + len(excluded_manual),
        "paired_sessions": len(rows),
        "excluded_manual_revision_sessions": len(excluded_manual),
        "manual_revision_policy": "excluded when old_revision > 0 or new_revision > 0",
        "review_hidden_policy": {
            "mode": "included_as_normal_masks",
            "old_flagged_masks_included": old_hidden_included,
            "new_flagged_masks_included": new_hidden_included,
        },
        "old": {"nonempty_sessions": old_nonempty, "objects": old_objects},
        "new": {"nonempty_sessions": new_nonempty, "objects": new_objects},
        "delta": {
            "nonempty_sessions": new_nonempty - old_nonempty,
            "coverage_percentage_points": round((new_nonempty - old_nonempty) * 100 / max(1, len(rows)), 2),
            "objects": new_objects - old_objects,
        },
        "transitions": {
            "newly_detected_sessions": new_only,
            "lost_detection_sessions": old_only,
        },
        "confidence": {
            "all_stored_masks_including_review_hidden": {
                "old_mean": round(old_weighted_score, 4),
                "new_mean": round(new_weighted_score, 4),
            },
        },
        "mask_consistency": {
            "both_nonempty_sessions": len(both_nonempty),
            "mean_penalized_iou": round(float(np.mean([
                row["penalized_iou"] for row in both_nonempty
            ])), 4) if both_nonempty else None,
        },
        "significant_change_rules": {
            "detection_status_changed": True,
            "absolute_object_count_delta_at_least": args.min_count_delta,
            "penalized_mask_iou_below": args.mask_change_iou,
        },
        "significant_candidates": sum(category_counts.values()),
        "exported_comparisons": len(all_cards),
        "highlight_comparisons": len(highlight_cards),
        "comparison_categories": dict(category_counts),
        "highlight_categories": dict(Counter(row["category"] for row in highlights)),
        "limitations": [
            "No ground-truth masks are available, so more detections do not prove higher precision.",
            "Mask IoU measures consistency between runs, not correctness.",
            "All review_hidden masks are intentionally included as normal candidates.",
            "Sessions with revision > 0 in either cache are excluded from this report.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    fieldnames = list(rows[0]) if rows else []
    with (output / "datasets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    with (output / "excluded_manual_revisions.csv").open("w", newline="") as stream:
        fields = ["session", "old_revision", "new_revision"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(excluded_manual)
    with (output / "human_review.csv").open("w", newline="") as stream:
        fields = ["session", "category", "comparison", "verdict", "false_positive_after", "false_negative_after", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            if row["highlight_comparison"]:
                review_row = {key: row.get(key, "") for key in fields}
                review_row["comparison"] = row["highlight_comparison"]
                writer.writerow(review_row)
    report = f"""# SAM3 超分前后严格可控 A/B 报告

- 原始配对数据：{len(rows) + len(excluded_manual)} 条
- 排除人工 revision：{len(excluded_manual)} 条
- 严格可控数据：{len(rows)} 条
- `review_hidden`：全部作为普通掩码计入（旧 {old_hidden_included} 个，新 {new_hidden_included} 个）
- 有掩码数据：{old_nonempty} → {new_nonempty}（{summary['delta']['coverage_percentage_points']:+.2f} 个百分点）
- 掩码数量：{old_objects} → {new_objects}（{new_objects-old_objects:+d}）
- 新增检出 / 丢失检出：{new_only} / {old_only}
- 全部掩码平均置信度：{old_weighted_score:.4f} → {new_weighted_score:.4f}
- 两次均检出数据的平均惩罚 IoU：{summary['mask_consistency']['mean_penalized_iou']}
- 明显变化候选：{summary['significant_candidates']} 条
- 完整对比图：{summary['exported_comparisons']} 张
- 精选对比图：{summary['highlight_comparisons']} 张
- 精选分类：{json.dumps(summary['highlight_categories'], ensure_ascii=False)}

## 如何判断是否提升

1. 检出率和物体数用于衡量召回变化。
2. 新旧掩码匹配 IoU 只衡量结果变化幅度，不能替代真值 IoU。
3. 精选图同时显示每个掩码置信度、新旧平均置信度和差值。
4. 请在 `human_review.csv` 中填写 `better`、`worse` 或 `same`。
5. 最终以人工有效率、误检率和漏检率判断超分是否真正提升效果。

## 限制

当前没有人工真值掩码；候选变多也可能意味着误检增加。本报告已经排除超分前或超分后 `revision > 0` 的全部数据。
"""
    (output / "REPORT.md").write_text(report)
    highlight_card_html = "".join(
        f'<article><a href="{html.escape(path)}"><img src="{html.escape(path)}" loading="lazy"></a>'
        f'<b>{html.escape(category)}</b><span>{html.escape(session)}</span></article>'
        for category, session, path, _ in highlight_cards
    )
    all_card_html = "".join(
        f'<article><a href="{html.escape(path)}"><img src="{html.escape(path)}" loading="lazy"></a>'
        f'<b>{html.escape(category)}</b><span>{html.escape(session)}</span></article>'
        for category, session, path, _ in all_cards
    )
    (output / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>SAM3 SR A/B</title>
<style>body{{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}article{{background:#1b2330;padding:10px;border-radius:9px}}img{{width:100%}}b,span{{display:block;margin-top:6px}}span{{color:#9ba8ba;overflow-wrap:anywhere}}</style>
<h1>SAM3 超分前后严格可控精选变化</h1><p>已排除 {len(excluded_manual)} 条人工 revision 数据；`review_hidden` 全部计入。完整保留 {len(all_cards)} 张对比图，<a href="all_comparisons.html" style="color:#78b9ff">查看全部</a>。本页分层精选 {len(highlight_cards)} 张。</p><div class="grid">{highlight_card_html}</div>""")
    (output / "all_comparisons.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>SAM3 SR A/B All</title>
<style>body{{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}}a{{color:#78b9ff}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}article{{background:#1b2330;padding:10px;border-radius:9px}}img{{width:100%}}b,span{{display:block;margin-top:6px}}span{{color:#9ba8ba;overflow-wrap:anywhere}}</style>
<h1>严格可控数据的全部明显变化对比</h1><p><a href="index.html">返回精选</a> · 已计入全部 `review_hidden` 掩码 · 共 {len(all_cards)} 张</p><div class="grid">{all_card_html}</div>""")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
