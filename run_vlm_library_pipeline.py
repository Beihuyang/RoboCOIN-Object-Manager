#!/usr/bin/env python3
"""Run attribute annotation, CLIP deduplication, and tree building with progress."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TREE_PATH = BASE_DIR / "objects" / "new_library_work" / "dedup_tree_layout.json"
ATTR_CACHE = BASE_DIR / "objects" / "new_library_work" / "attributes.jsonl"
PROGRESS_PREFIX = "@@PROGRESS "


def emit_progress(percent: float, message: str) -> None:
    payload = {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "message": message,
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def run_stage(command: list[str], start: float, end: float, label: str) -> None:
    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if PROGRESS_PREFIX in line:
            prefix_text, event_text = line.rsplit(PROGRESS_PREFIX, 1)
            try:
                event = json.loads(event_text)
                if prefix_text:
                    print(prefix_text, end="", flush=True)
                child_percent = float(event["percent"])
                percent = start + (end - start) * child_percent / 100
                emit_progress(percent, str(event["message"]))
                continue
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"{label} failed with exit code {return_code}")


def preserve_existing_tree_fingerprint_baseline() -> None:
    """Capture old image fingerprints before attribute cache refresh changes them."""
    if not TREE_PATH.is_file() or not ATTR_CACHE.is_file():
        return
    from dedup_tree import (
        backfill_layout_instance_fingerprints,
        load_layout,
        save_layout,
    )

    items = []
    for line in ATTR_CACHE.read_text().splitlines():
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    layout, added = backfill_layout_instance_fingerprints(load_layout(), items)
    if added:
        save_layout(layout)
        print(f"Captured fingerprint baseline for {added} existing tree object(s).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("sam3",), default="sam3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preserve_existing_tree_fingerprint_baseline()

    annotation = [
        sys.executable,
        str(BASE_DIR / "stage3_attribute.py"),
        "--tracker",
        args.tracker,
    ]
    if args.force:
        annotation.append("--force")
    emit_progress(0, "开始检查VLM属性缓存")
    run_stage(annotation, 0, 70, "VLM attribute annotation")

    emit_progress(70, "开始计算CLIP特征和去重候选")
    run_stage(
        [sys.executable, str(BASE_DIR / "stage4_dedup.py")],
        70,
        95,
        "CLIP deduplication",
    )

    clip_threshold = 0.82
    attribute_threshold = 0.50
    if TREE_PATH.is_file():
        try:
            settings = json.loads(TREE_PATH.read_text()).get("settings", {})
            clip_threshold = float(settings.get("clip_threshold", clip_threshold))
            attribute_threshold = float(
                settings.get("attribute_threshold", attribute_threshold)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    emit_progress(95, "正在生成可视化类别树和物体库")
    run_stage(
        [
            sys.executable,
            str(BASE_DIR / "dedup_tree.py"),
            "--regenerate",
            "--clip-threshold",
            str(clip_threshold),
            "--attribute-threshold",
            str(attribute_threshold),
        ],
        95,
        100,
        "Dedup tree generation",
    )
    emit_progress(100, "VLM属性、CLIP和类别树已全部更新")


if __name__ == "__main__":
    main()
