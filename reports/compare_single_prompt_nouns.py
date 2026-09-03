#!/usr/bin/env python3
"""Compare cached legacy discovery with one combined dataset/text noun prompt."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from project_paths import resolve_project_path
from stage1_track_select import DEFAULT_CKPT, detect_first_frame, load_sam3_detector
from semantic_prompts import combined_semantic_prompt
from reports.compare_semantic_prompts import _render

DEFAULT_OUTPUT = BASE_DIR / "reports" / "2026-09-03_single_prompt_nouns_ab_100"
TRACKS_ROOT = BASE_DIR / "objects" / "tracks"
DATA_ROOT = BASE_DIR / "RoboCOIN_datasets"


def _key(session: str) -> str:
    return hashlib.sha1(session.encode()).hexdigest()[:12]


def _load_cached_before(manifest_path: Path, manifest: dict) -> list[dict]:
    result = []
    for item in manifest.get("objects", []):
        path = manifest_path.parent / item["mask"]
        if not path.is_file():
            continue
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L")) > 127
        result.append({
            "mask": mask,
            "bbox": item["bbox"],
            "score": float(item["score"]),
            "prompt": str(item.get("prompt", "legacy")),
        })
    return result


def _save_result(directory: Path, detections: list[dict], metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    objects = []
    for index, item in enumerate(detections):
        name = f"mask_{index:04d}.png"
        Image.fromarray((item["mask"] * 255).astype(np.uint8)).save(directory / name)
        objects.append({
            "mask": name,
            "bbox": [float(value) for value in item["bbox"]],
            "score": float(item["score"]),
            "prompt": str(item.get("prompt", "")),
        })
    (directory / "result.json").write_text(json.dumps(
        {**metadata, "objects": objects}, indent=2, ensure_ascii=False
    ))


def _load_result(directory: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "result.json").read_text())
    detections = []
    for item in metadata["objects"]:
        with Image.open(directory / item["mask"]) as source:
            mask = np.asarray(source.convert("L")) > 127
        detections.append({**item, "mask": mask})
    return metadata, detections


def _scores(items: list[dict]) -> list[float]:
    return [float(item["score"]) for item in items]


def _mean(items: list[dict]) -> float | None:
    values = _scores(items)
    return float(np.mean(values)) if values else None


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / max(1, union)


def _greedy_iou(before: list[dict], after: list[dict]) -> float | None:
    if not before or not after:
        return None
    pairs = sorted(
        (
            (_mask_iou(left["mask"], right["mask"]), li, ri)
            for li, left in enumerate(before)
            for ri, right in enumerate(after)
        ),
        reverse=True,
    )
    used_left, used_right, values = set(), set(), []
    for iou, li, ri in pairs:
        if li in used_left or ri in used_right:
            continue
        used_left.add(li); used_right.add(ri); values.append(iou)
    denominator = max(len(before), len(after))
    return sum(values) / denominator


def _comparison(frame: Path, session: str, prompt: str,
                before: list[dict], after: list[dict]) -> Image.Image:
    before_view = [{**item, "prompt": "before"} for item in before]
    after_view = [{**item, "prompt": "after"} for item in after]
    left = _render(frame, before_view, [item["mask"] for item in before_view])
    right = _render(frame, after_view, [item["mask"] for item in after_view])
    panel_w, panel_h = max(left.width, right.width), max(left.height, right.height)
    header = 110
    canvas = Image.new("RGB", (panel_w * 2, panel_h + header), (18, 22, 29))
    canvas.paste(left, ((panel_w - left.width) // 2, header))
    canvas.paste(right, (panel_w + (panel_w - right.width) // 2, header))
    draw = ImageDraw.Draw(canvas)
    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    draw.text((12, 8), session, fill=(235, 241, 250))
    draw.text((12, 32), f"BEFORE legacy | masks={len(before)} | mean={fmt(_mean(before))}", fill=(160, 196, 255))
    draw.text((panel_w + 12, 32), f"AFTER one noun prompt | masks={len(after)} | mean={fmt(_mean(after))}", fill=(144, 232, 180))
    visible_prompt = prompt if len(prompt) <= 210 else prompt[:207] + "..."
    draw.text((12, 58), "one prompt: " + visible_prompt, fill=(192, 198, 208))
    draw.text((12, 83), "Same 2K frame and threshold; labels show side and confidence", fill=(192, 198, 208))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--highlights", type=int, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    raw = output / "raw"
    candidates = []
    excluded = []
    for manifest_path in sorted(TRACKS_ROOT.rglob("initial_sam3_sr_2k/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        session = manifest_path.parent.parent.relative_to(TRACKS_ROOT).as_posix()
        frame = resolve_project_path(manifest.get("frame", ""))
        if int(manifest.get("revision", 0)) > 0 or not frame.is_file():
            excluded.append(session)
            continue
        if manifest.get("prompt_strategy") not in {"dataset_nouns_v3", None}:
            excluded.append(session)
            continue
        candidates.append((session, manifest_path, manifest, frame))
        if len(candidates) >= args.limit:
            break
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    pending = [item for item in candidates if not (raw / _key(item[0]) / "after/result.json").is_file()]
    detector = processor = None
    if pending:
        print(f"Loading SAM3; pending {len(pending)}/{len(candidates)}", flush=True)
        detector, processor = load_sam3_detector(args.checkpoint, args.gpu)
    try:
        for index, (session, manifest_path, manifest, frame) in enumerate(candidates, 1):
            directory = raw / _key(session)
            after_result = directory / "after/result.json"
            if after_result.is_file():
                print(f"[{index}/{len(candidates)}] resume {session}", flush=True)
                continue
            video = DATA_ROOT / f"{session}.mp4"
            try:
                prompt = combined_semantic_prompt(video, DATA_ROOT)
                before = _load_cached_before(manifest_path, manifest)
                after, filter_report = detect_first_frame(
                    frame, detector, processor, prompt, args.threshold,
                    exclude_robot_arms=False,
                )
                metadata = {"session": session, "frame": str(frame.resolve())}
                _save_result(directory / "before", before, {
                    **metadata, "strategy": "legacy cached discovery",
                })
                _save_result(directory / "after", after, {
                    **metadata, "strategy": "one combined noun prompt",
                    "prompt": prompt, "filter_report": filter_report,
                })
                print(f"[{index}/{len(candidates)}] {session}: {len(before)} -> {len(after)}", flush=True)
            except Exception as exc:
                failures.append({"session": session, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(candidates)}] FAILED {session}: {exc}", flush=True)
                if isinstance(exc, torch.OutOfMemoryError):
                    gc.collect(); torch.cuda.empty_cache()
    finally:
        if detector is not None:
            del processor, detector
            gc.collect(); torch.cuda.empty_cache()

    rows, loaded = [], []
    for session, _, _, frame in candidates:
        directory = raw / _key(session)
        if not (directory / "after/result.json").is_file():
            continue
        _, before = _load_result(directory / "before")
        after_meta, after = _load_result(directory / "after")
        row = {
            "session": session,
            "before_count": len(before), "after_count": len(after),
            "count_delta": len(after) - len(before),
            "before_mean_confidence": _mean(before),
            "after_mean_confidence": _mean(after),
            "confidence_delta": (
                _mean(after) - _mean(before) if before and after else None
            ),
            "penalized_greedy_iou": _greedy_iou(before, after),
            "noun_count": len(after_meta["prompt"].split(", ")),
            "prompt": after_meta["prompt"],
        }
        if not before and after:
            row["category"] = "newly_detected"
        elif before and not after:
            row["category"] = "lost_detection"
        elif row["count_delta"] > 0:
            row["category"] = "count_gain"
        elif row["count_delta"] < 0:
            row["category"] = "count_loss"
        else:
            row["category"] = "same_count"
        rows.append(row); loaded.append((row, frame, before, after))

    all_scores_before = [score for _, _, before, _ in loaded for score in _scores(before)]
    all_scores_after = [score for _, _, _, after in loaded for score in _scores(after)]
    before_total = len(all_scores_before); after_total = len(all_scores_after)
    before_nonempty = sum(row["before_count"] > 0 for row in rows)
    after_nonempty = sum(row["after_count"] > 0 for row in rows)
    q = lambda values, value: round(float(np.quantile(values, value)), 4) if values else None
    summary = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "legacy cached multi-prompt discovery vs one combined dataset/meta/annotation noun prompt",
        "sample_selection": "first 100 sorted revision=0 legacy discovery sessions with existing 2K frames",
        "requested_sessions": args.limit,
        "completed_sessions": len(rows),
        "failed_sessions": len(failures),
        "threshold": args.threshold,
        "before": {
            "nonempty_sessions": before_nonempty, "objects": before_total,
            "mean_confidence": round(float(np.mean(all_scores_before)), 4) if all_scores_before else None,
            "median_confidence": q(all_scores_before, 0.5),
            "confidence_p25": q(all_scores_before, 0.25),
            "confidence_p75": q(all_scores_before, 0.75),
        },
        "after": {
            "nonempty_sessions": after_nonempty, "objects": after_total,
            "mean_confidence": round(float(np.mean(all_scores_after)), 4) if all_scores_after else None,
            "median_confidence": q(all_scores_after, 0.5),
            "confidence_p25": q(all_scores_after, 0.25),
            "confidence_p75": q(all_scores_after, 0.75),
        },
        "delta": {
            "nonempty_sessions": after_nonempty - before_nonempty,
            "coverage_percentage_points": round((after_nonempty-before_nonempty)*100/max(1,len(rows)), 2),
            "objects": after_total-before_total,
            "objects_percent": round((after_total-before_total)*100/max(1,before_total), 2),
        },
        "categories": dict(Counter(row["category"] for row in rows)),
        "average_noun_count": round(float(np.mean([row["noun_count"] for row in rows])), 2) if rows else 0,
        "mean_penalized_greedy_iou": round(float(np.mean([
            row["penalized_greedy_iou"] for row in rows
            if row["penalized_greedy_iou"] is not None
        ])), 4) if any(row["penalized_greedy_iou"] is not None for row in rows) else None,
        "limitations": [
            "No ground-truth masks: counts and confidence do not prove correctness.",
            "The before side is the preserved legacy cache; the after side is freshly inferred on the same 2K frame.",
            "The new pipeline intentionally removes extra robot-arm prompt calls, so the comparison measures the complete workflow change.",
        ],
    }

    comparisons = output / "comparisons" / "all"
    highlights = output / "comparisons" / "highlights"
    comparisons.mkdir(parents=True, exist_ok=True); highlights.mkdir(parents=True, exist_ok=True)
    visibly_changed = sorted(
        loaded,
        key=lambda item: (
            abs(item[0]["count_delta"]),
            abs(item[0]["confidence_delta"] or 0),
        ), reverse=True,
    )
    visibly_changed = [item for item in visibly_changed if (
        item[0]["count_delta"] != 0
        or abs(item[0]["confidence_delta"] or 0) >= 0.08
        or (item[0]["penalized_greedy_iou"] or 1) < 0.5
    )]
    for index, (row, frame, before, after) in enumerate(visibly_changed, 1):
        name = f"{index:03d}_{row['category']}__{row['session'].split('/')[0]}__{_key(row['session'])}.jpg"
        target = comparisons / name
        _comparison(frame, row["session"], row["prompt"], before, after).save(target, quality=92, optimize=True)
        row["comparison"] = target.relative_to(output).as_posix()
    for rank, (row, _, _, _) in enumerate(visibly_changed[:args.highlights], 1):
        source = output / row["comparison"]
        target = highlights / f"{rank:02d}_{source.name}"
        shutil.copy2(source, target)
        row["highlight"] = target.relative_to(output).as_posix()

    fields = list(rows[0]) if rows else []
    for row in rows:
        row.setdefault("comparison", ""); row.setdefault("highlight", "")
    with (output / "datasets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "comparison", "highlight"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False))
    cards = "".join(
        f'<article><a href="{html.escape(row.get("highlight", row.get("comparison", "")))}"><img src="{html.escape(row.get("highlight", row.get("comparison", "")))}" loading="lazy"></a><b>{row["before_count"]} → {row["after_count"]} masks</b><span>{html.escape(row["session"])}</span></article>'
        for row, _, _, _ in visibly_changed[:args.highlights]
    )
    (output / "index.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>Single noun prompt A/B</title>
<style>body{{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}}a{{color:#78b9ff}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px}}article{{background:#1b2330;padding:10px;border-radius:9px}}img{{width:100%}}b,span{{display:block;margin-top:6px}}span{{color:#9ba8ba;overflow-wrap:anywhere}}</style>
<h1>单次组合名词提示 A/B（{len(rows)}条）</h1><p>掩码 {before_total} → {after_total}；有检出数据 {before_nonempty} → {after_nonempty}。</p><div class="grid">{cards}</div>''')
    report = f"""# SAM3 单次组合名词提示 A/B（100条）

- 完成：{len(rows)} 条；失败：{len(failures)} 条
- 有掩码数据：{before_nonempty} → {after_nonempty}（{summary['delta']['coverage_percentage_points']:+.2f}个百分点）
- 掩码总数：{before_total} → {after_total}（{after_total-before_total:+d}，{summary['delta']['objects_percent']:+.2f}%）
- 平均置信度：{summary['before']['mean_confidence']} → {summary['after']['mean_confidence']}
- 中位置信度：{summary['before']['median_confidence']} → {summary['after']['median_confidence']}
- 置信度P25–P75：{summary['before']['confidence_p25']}–{summary['before']['confidence_p75']} → {summary['after']['confidence_p25']}–{summary['after']['confidence_p75']}
- 平均组合名词数：{summary['average_noun_count']}
- 变化分类：{json.dumps(summary['categories'], ensure_ascii=False)}
- 平均惩罚贪心IoU：{summary['mean_penalized_greedy_iou']}
- 明显变化图：{len(visibly_changed)} 张；精选：{min(args.highlights, len(visibly_changed))} 张

## 说明

修改前读取相同2K关键帧的旧版未人工修改缓存；修改后使用数据名称和 `meta/annotations` 文本名词去重拼接，只调用一次SAM3。阈值均为 {args.threshold}。当前没有真值掩码，因此数量和置信度只能衡量输出变化，不能单独证明准确率。
"""
    (output / "REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
