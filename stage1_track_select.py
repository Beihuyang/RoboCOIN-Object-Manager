#!/usr/bin/env python3
"""Discover first-frame objects and export representative tracks.

SAM 3 discovers objects on source frame zero by default. The review interface
can replace it with a manually chosen keyframe and rerun discovery. The official
SAM 3.1 multiplex tracker propagates the reviewed masks over uniformly sampled
frames starting at the active discovery frame. For every track the script exports
``best_quality``: the frame with the best configurable
combination of sharpness, area, confidence, temporal area stability, and
image-boundary score.

Usage:
    python stage1_track_select.py --stage discover --limit 1
    python stage1_track_select.py --stage refine --limit 1
    python stage1_track_select.py --stage track --limit 1
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from project_paths import portable_path, resolve_project_path, same_project_path
from semantic_prompts import (
    DEFAULT_MAX_SEMANTIC_PROMPTS,
    PROMPT_STRATEGY,
    dataset_name_from_video,
    discovery_prompts,
)


BASE_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "RoboCOIN_datasets"
OUTPUT_ROOT = BASE_DIR / "objects" / "tracks"
SAM3_REPO = BASE_DIR / "sam3"
DEFAULT_CKPT = BASE_DIR / "sam3_weights" / "sam3.pt"
DEFAULT_SAM31_CKPT = BASE_DIR / "sam3_weights" / "sam3.1_multiplex.pt"
BPE_PATH = SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CAMERA_KEYWORDS = (
    "high", "head", "front", "chest", "center", "ego", "left", "right"
)
FIRST_FRAME_EXTRACTION_METHOD = "source_frame_selected_sr_2k_v1"
DISCOVERY_CACHE_VERSION = 14
TRACK_FRAME_EXTRACTION_METHOD = "source_discovery_frames_interval_v3"
PROGRESS_PREFIX = "@@PROGRESS "
ROBOT_ARM_PROMPTS = ("robot arm", "robot gripper", "robot hand")
DEFAULT_ROBOT_ARM_THRESHOLD = 0.10
DEFAULT_ROBOT_ARM_OVERLAP = 0.50
DEFAULT_CROSS_PROMPT_IOU = 0.55
DEFAULT_CROSS_PROMPT_CONTAINMENT = 0.80


def _set_sam3_image(processor, image: Image.Image):
    """Run resident 2K Real-ESRGAN, but return masks at source resolution."""
    from super_resolution import upscale_for_sam3

    original_size = image.size
    enhanced = upscale_for_sam3(image)
    return processor.set_image(enhanced, original_size=original_size)


def emit_progress(percent: float, message: str) -> None:
    """Write a machine-readable progress event while keeping normal logs useful."""
    payload = {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "message": message,
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


@dataclass
class FrameCandidate:
    frame_index: int
    frame_path: str
    mask_area: int
    area_ratio: float
    sharpness: float
    confidence: float
    touches_boundary: bool
    area_stability: float = 1.0
    quality_score: float = 0.0


def candidate_metadata(candidate: FrameCandidate) -> dict:
    """Serialize a runtime candidate without tying it to this checkout path."""
    data = asdict(candidate)
    data["frame_path"] = portable_path(candidate.frame_path)
    return data


QUALITY_WEIGHTS = {
    "sharpness": 0.45,
    "area": 0.20,
    "confidence": 0.15,
    "stability": 0.10,
    "boundary": 0.10,
}
QUALITY_FILTERS = {
    "min_relative_area": 0.25,
    "min_stability": 0.50,
    "min_sharpness_quantile": 0.10,
    "exclude_boundary": True,
}


def _camera_priority(video_path: Path) -> tuple[int, str] | None:
    """Rank a camera using only its stream directory, not the dataset name."""
    camera_name = video_path.parent.name.lower()
    for priority, keyword in enumerate(CAMERA_KEYWORDS):
        if keyword in camera_name:
            return priority, keyword
    return None


def find_videos(video_root: Path = VIDEO_ROOT) -> tuple[list[Path], dict]:
    """Choose exactly one episode-0 camera video from each dataset."""
    candidates_by_dataset: dict[str, list[Path]] = {}
    for path in sorted(video_root.rglob("episode_000000.mp4")):
        try:
            relative = path.relative_to(video_root)
        except ValueError:
            continue
        if "chunk-000" not in relative.parts or not relative.parts:
            continue
        candidates_by_dataset.setdefault(relative.parts[0], []).append(path)

    selected = []
    report = {
        "datasets_scanned": len(candidates_by_dataset),
        "selected": [],
        "no_priority_match": [],
    }
    for dataset, candidates in sorted(candidates_by_dataset.items()):
        ranked = [
            (*priority, path)
            for path in candidates
            if (priority := _camera_priority(path)) is not None
        ]
        if not ranked:
            report["no_priority_match"].append({
                "dataset": dataset,
                "candidates": candidates,
            })
            continue
        priority, keyword, chosen = min(
            ranked, key=lambda item: (item[0], str(item[2]))
        )
        selected.append(chosen)
        report["selected"].append({
            "dataset": dataset,
            "keyword": keyword,
            "priority": priority,
            "path": chosen,
            "skipped": [path for path in candidates if path != chosen],
        })
    return selected, report


def print_video_selection_report(report: dict, run_videos: list[Path]) -> None:
    run_paths = {path.resolve() for path in run_videos}
    selected_items = report.get("selected", [])
    limited_out = [
        item for item in selected_items if item["path"].resolve() not in run_paths
    ]
    alternatives = sum(len(item["skipped"]) for item in selected_items)
    print("\n=== Video selection report ===")
    print(f"Priority:         {' > '.join(CAMERA_KEYWORDS)}")
    print(f"Datasets scanned: {report.get('datasets_scanned', 0)}")
    print(f"Selected to run:  {len(run_paths)}")
    print(f"Other views skip: {alternatives}")
    print(f"Limited out:      {len(limited_out)}")
    print(f"No keyword match: {len(report.get('no_priority_match', []))}")
    for item in selected_items:
        if item["path"].resolve() not in run_paths:
            continue
        camera = item["path"].parent.name
        print(
            f"  - {item['dataset']}: [{item['keyword']}] {camera}"
            f"; skipped {len(item['skipped'])} other view(s)"
        )
    for item in limited_out:
        print(f"  - limit skipped dataset: {item['dataset']}")
    for item in report.get("no_priority_match", []):
        cameras = ", ".join(path.parent.name for path in item["candidates"])
        print(f"  - no matching camera: {item['dataset']} ({cameras})")


def video_key(video_path: Path) -> Path:
    """Return a stable, filesystem-safe relative key for output directories."""
    try:
        relative = video_path.resolve().relative_to(VIDEO_ROOT.resolve())
        return relative.with_suffix("")
    except ValueError:
        return Path(video_path.stem)


def _video_stream_geometry(video_path: Path) -> tuple[float, int, int]:
    """Read video FPS and geometry without relying on OpenCV codec support."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height",
        "-of", "json", str(video_path),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    try:
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
        source_fps = float(numerator) / float(denominator)
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Invalid ffprobe metadata for {video_path}") from exc
    if not math.isfinite(source_fps) or source_fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video stream metadata for {video_path}")
    return source_fps, width, height


def extract_source_frame_image(
    video_path: Path, source_frame_index: int, output_path: Path
) -> None:
    """Decode one exact source frame to a JPEG without OpenCV codec assumptions."""
    source_frame_index = max(0, int(source_frame_index))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".extracting.jpg")
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{source_frame_index})",
        "-frames:v", "1", "-q:v", "2", str(temp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr.strip()}")
    if not temp_path.is_file() or temp_path.stat().st_size == 0:
        raise RuntimeError(
            f"Could not extract source frame {source_frame_index} from {video_path}"
        )
    temp_path.replace(output_path)


def enhance_frame_to_2k(frame_path: Path) -> dict:
    """Replace one extracted frame with its cached 2K review representation."""
    from super_resolution import status, upscale_for_sam3

    with Image.open(frame_path) as source:
        original = source.convert("RGB")
    original_size = original.size
    enhanced = upscale_for_sam3(original)
    enhanced_size = enhanced.size
    temp_path = frame_path.with_name(frame_path.stem + ".enhancing.jpg")
    enhanced.save(temp_path, quality=95)
    temp_path.replace(frame_path)
    return {
        **status(),
        "source_size": [int(original_size[0]), int(original_size[1])],
        "review_size": [int(enhanced_size[0]), int(enhanced_size[1])],
    }


def discovery_frame_metadata(
    video_path: Path, source_frame_index: int, reason: str
) -> dict:
    source_fps, _, _ = _video_stream_geometry(video_path)
    source_frame_index = max(0, int(source_frame_index))
    return {
        "source": portable_path(video_path),
        "source_frame_index": source_frame_index,
        "timestamp_seconds": source_frame_index / source_fps,
        "source_fps": source_fps,
        "method": FIRST_FRAME_EXTRACTION_METHOD,
        "reason": reason,
        "frame_count": 1,
    }


def extract_discovery_frame(
    video_path: Path,
    frame_dir: Path,
    source_frame_index: int = 0,
    reason: str = "source_frame_zero_default",
) -> Path:
    """Extract one exact source frame for SAM3 discovery."""
    source_frame_index = max(0, int(source_frame_index))
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"source_{source_frame_index:06d}.jpg"
    marker = frame_dir / "extraction.json"
    if marker.exists() and frame_path.is_file():
        try:
            metadata = json.loads(marker.read_text())
            if (
                same_project_path(metadata.get("source", ""), video_path)
                and metadata.get("method") == FIRST_FRAME_EXTRACTION_METHOD
                and int(metadata.get("source_frame_index", -1))
                == source_frame_index
            ):
                return frame_path
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    extract_source_frame_image(video_path, source_frame_index, frame_path)
    enhancement = enhance_frame_to_2k(frame_path)
    selection = discovery_frame_metadata(video_path, source_frame_index, reason)
    selection["image_representation"] = "realesrgan_2k"
    selection["super_resolution"] = enhancement
    marker.write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    print(
        f"Discovery frame {source_frame_index} at "
        f"{selection['timestamp_seconds']:.2f}s ({selection['reason']}): {video_path}",
        flush=True,
    )
    return frame_path


def extract_first_frame(video_path: Path, frame_dir: Path) -> Path:
    """Use source frame zero by default; review may replace it manually later."""
    return extract_discovery_frame(video_path, frame_dir, 0)


def extract_sampled_frames(
    video_path: Path,
    frame_dir: Path,
    sample_fps: float,
    start_source_frame_index: int = 0,
    required_source_frame_indices: list[int] | None = None,
) -> list[Path]:
    """Extract uniform samples and inject every exact discovery keyframe."""
    required_source_frame_indices = sorted({
        max(0, int(value)) for value in (required_source_frame_indices or [])
        if int(value) >= start_source_frame_index
    })
    frame_dir.mkdir(parents=True, exist_ok=True)
    marker = frame_dir / "extraction.json"
    existing = sorted(frame_dir.glob("*.jpg"))
    if marker.exists() and existing:
        try:
            metadata = json.loads(marker.read_text())
            if (
                same_project_path(metadata.get("source", ""), video_path)
                and float(metadata.get("sample_fps")) == sample_fps
                and metadata.get("method") == TRACK_FRAME_EXTRACTION_METHOD
                and int(metadata.get("start_source_frame_index", 0))
                == start_source_frame_index
                and [int(value) for value in metadata.get(
                    "required_source_frame_indices", []
                )] == required_source_frame_indices
            ):
                return existing
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # Use a temporary sibling directory so an interrupted extraction never
    # looks like a complete cache on the next run.
    temp_dir = frame_dir.with_name(frame_dir.name + ".extracting")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for stale in temp_dir.glob("*.jpg"):
        stale.unlink()

    sample_interval = 1.0 / sample_fps
    select_filter = (
        f"select=eq(n\\,{start_source_frame_index})+"
        f"gt(n\\,{start_source_frame_index})*"
        f"gte(t-prev_selected_t\\,{sample_interval:.12g})"
    )
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-vf", select_filter, "-vsync", "vfr", "-q:v", "2",
        "-start_number", "0",
        str(temp_dir / "%06d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr.strip()}")

    extracted = sorted(temp_dir.glob("*.jpg"))
    if not extracted:
        raise RuntimeError(f"No frames extracted from {video_path}")

    source_fps, _, _ = _video_stream_geometry(video_path)
    uniform_source_indices = []
    previous_index = start_source_frame_index - 1
    for sample_index in range(len(extracted)):
        source_index = start_source_frame_index + round(
            sample_index * source_fps / sample_fps
        )
        source_index = max(previous_index + 1, source_index)
        uniform_source_indices.append(source_index)
        previous_index = source_index
    frames_by_source = dict(zip(uniform_source_indices, extracted))
    for source_index in required_source_frame_indices:
        required_path = temp_dir / f"required_{source_index:09d}.jpg"
        extract_source_frame_image(video_path, source_index, required_path)
        frames_by_source[source_index] = required_path

    ordered_dir = temp_dir / "ordered"
    ordered_dir.mkdir()
    source_frame_indices = sorted(frames_by_source)
    for output_index, source_index in enumerate(source_frame_indices):
        frames_by_source[source_index].replace(ordered_dir / f"{output_index:06d}.jpg")

    for old in frame_dir.glob("*.jpg"):
        old.unlink()
    for path in sorted(ordered_dir.glob("*.jpg")):
        path.rename(frame_dir / path.name)
    shutil.rmtree(temp_dir)
    marker.write_text(json.dumps({
        "source": portable_path(video_path),
        "sample_fps": sample_fps,
        "method": TRACK_FRAME_EXTRACTION_METHOD,
        "start_source_frame_index": start_source_frame_index,
        "required_source_frame_indices": required_source_frame_indices,
        "source_frame_indices": source_frame_indices,
        "frame_count": len(source_frame_indices),
    }, indent=2))
    return sorted(frame_dir.glob("*.jpg"))


def load_sam31_mask_tracker(checkpoint: Path, gpu: int):
    """Load the official SAM 3.1 multiplex mask tracker without text grounding."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 3.1 checkpoint not found: {checkpoint}\n"
            "Download facebook/sam3.1/sam3.1_multiplex.pt or pass "
            "--sam31-checkpoint /path/to/sam3.1_multiplex.pt"
        )
    torch.cuda.set_device(gpu)
    sys.path.insert(0, str(SAM3_REPO))
    from sam3.model_builder import build_sam3_multiplex_video_model

    tracker = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        strict_state_dict_loading=False,
        device="cpu",
        compile=False,
    )
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_state = checkpoint_state.get("model", checkpoint_state)
    model_state = tracker.state_dict()
    tracker_state = {}
    for key, value in checkpoint_state.items():
        candidates = [key]
        if key.startswith("tracker.model."):
            candidates.append(key.removeprefix("tracker.model."))
        if key.startswith("tracker."):
            candidates.append(key.removeprefix("tracker."))
        if key.startswith("detector.backbone."):
            candidates.append("backbone." + key.removeprefix("detector.backbone."))
        for mapped_key in candidates:
            expected = model_state.get(mapped_key)
            if expected is not None and expected.shape == value.shape:
                tracker_state[mapped_key] = value
                break
    missing, unexpected = tracker.load_state_dict(tracker_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "SAM 3.1 tracker checkpoint mismatch: "
            f"loaded={len(tracker_state)}/{len(model_state)}, "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    del checkpoint_state, tracker_state, model_state
    return tracker.to(device=f"cuda:{gpu}").eval()


def load_sam3_detector(
    checkpoint: Path,
    gpu: int,
    enable_inst_interactivity: bool = False,
):
    """Load the SAM 3 image model used for discovery-frame segmentation."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {checkpoint}")
    torch.cuda.set_device(gpu)
    sys.path.insert(0, str(SAM3_REPO))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        bpe_path=str(BPE_PATH),
        enable_inst_interactivity=enable_inst_interactivity,
    ).cuda().eval()
    return model, Sam3Processor(
        model,
        confidence_threshold=0.0,
        mask_upsample_chunk_size=8,
        offload_masks_to_cpu=True,
        retain_mask_logits=False,
    )


def _decode_grounding_detections(
    output: dict,
    height: int,
    width: int,
    prompt: str,
    threshold: float,
) -> list[dict]:
    """Convert one SAM3 text-grounding result into NMS-filtered masks."""
    from torchvision.ops import nms

    boxes = _as_numpy(output.get("boxes", [])).reshape(-1, 4)
    masks = _as_numpy(output.get("masks", []))
    scores = _as_numpy(output.get("scores", [])).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    valid_masks = []
    for index in range(min(len(boxes), len(masks), len(scores))):
        mask = masks[index].astype(bool)
        if mask.shape == (height, width) and mask.any():
            valid_masks.append(index)
    valid = [
        index for index in valid_masks if float(scores[index]) >= threshold
    ]
    if not valid:
        return []

    keep_local = nms(
        torch.as_tensor(boxes[valid], dtype=torch.float32),
        torch.as_tensor(scores[valid], dtype=torch.float32),
        iou_threshold=0.5,
    ).tolist()
    results = []
    for local_index in keep_local:
        index = valid[local_index]
        results.append({
            "bbox": [float(value) for value in boxes[index]],
            "mask": masks[index].astype(bool),
            "score": float(scores[index]),
            "prompt": prompt,
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _detect_robot_regions_from_state(
    state: dict,
    processor,
    height: int,
    width: int,
    threshold: float,
) -> list[dict]:
    """Run all robot-arm prompts while reusing one encoded first frame."""
    regions = []
    processor.set_confidence_threshold(max(0.0, threshold - 1e-7))
    for prompt in ROBOT_ARM_PROMPTS:
        processor.reset_all_prompts(state)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            output = processor.set_text_prompt(state=state, prompt=prompt)
        regions.extend(_decode_grounding_detections(
            output, height, width, prompt, threshold
        ))
    return regions


def _filter_robot_arm_overlaps(
    detections: list[dict],
    robot_regions: list[dict],
    overlap_threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Remove object masks substantially covered by the union of robot regions."""
    if not robot_regions:
        return detections, []
    robot_union = np.logical_or.reduce([item["mask"] for item in robot_regions])
    kept = []
    removed = []
    for detection in detections:
        mask = detection["mask"]
        covered = int(np.logical_and(mask, robot_union).sum())
        coverage = covered / max(1, int(mask.sum()))
        if coverage >= overlap_threshold:
            removed.append({
                "bbox": detection["bbox"],
                "score": detection["score"],
                "robot_overlap": coverage,
            })
        else:
            kept.append(detection)
    return kept, removed


def _filter_image_boundary_masks(
    detections: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove first-frame masks touching any outer image edge."""
    kept = []
    removed = []
    for detection in detections:
        mask = detection["mask"]
        touched_edges = []
        if mask[0, :].any():
            touched_edges.append("top")
        if mask[-1, :].any():
            touched_edges.append("bottom")
        if mask[:, 0].any():
            touched_edges.append("left")
        if mask[:, -1].any():
            touched_edges.append("right")
        if touched_edges:
            removed.append({
                "bbox": detection["bbox"],
                "score": detection["score"],
                "touched_edges": touched_edges,
            })
        else:
            kept.append(detection)
    return kept, removed


def _mask_duplicate(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, float]:
    intersection = int(np.logical_and(a, b).sum())
    if not intersection:
        return False, 0.0, 0.0
    area_a, area_b = int(a.sum()), int(b.sum())
    iou = intersection / max(1, area_a + area_b - intersection)
    containment = intersection / max(1, min(area_a, area_b))
    area_ratio = min(area_a, area_b) / max(1, max(area_a, area_b))
    duplicate = (
        iou >= DEFAULT_CROSS_PROMPT_IOU
        or (
            containment >= DEFAULT_CROSS_PROMPT_CONTAINMENT
            and area_ratio >= 0.50
        )
    )
    return duplicate, iou, containment


def merge_cross_prompt_detections(
    detections_by_prompt: list[tuple[str, list[dict]]],
) -> tuple[list[dict], list[dict]]:
    """Keep base-prompt masks first and add only novel semantic masks.

    Confidence scores from different text prompts are not calibrated against one
    another, so a later semantic result never replaces an earlier ``object`` mask.
    """
    kept: list[dict] = []
    duplicates: list[dict] = []
    for prompt, detections in detections_by_prompt:
        for detection in detections:
            matched = None
            for kept_index, existing in enumerate(kept):
                duplicate, iou, containment = _mask_duplicate(
                    detection["mask"], existing["mask"]
                )
                if duplicate:
                    matched = (kept_index, iou, containment)
                    break
            if matched is None:
                item = dict(detection)
                item["matched_prompts"] = [{
                    "prompt": prompt,
                    "score": float(detection["score"]),
                }]
                kept.append(item)
                continue
            kept_index, iou, containment = matched
            kept[kept_index].setdefault("matched_prompts", []).append({
                "prompt": prompt,
                "score": float(detection["score"]),
            })
            duplicates.append({
                "prompt": prompt,
                "score": float(detection["score"]),
                "kept_index": kept_index,
                "kept_prompt": kept[kept_index]["prompt"],
                "mask_iou": iou,
                "containment": containment,
            })
    return kept, duplicates


def semantic_discovery_prompts(
    video_path: Path,
    base_prompt: str = "object",
    max_semantic_prompts: int = DEFAULT_MAX_SEMANTIC_PROMPTS,
) -> list[str]:
    dataset_name = dataset_name_from_video(video_path, VIDEO_ROOT)
    return discovery_prompts(dataset_name, base_prompt, max_semantic_prompts)


def detect_first_frame(
    frame_path: Path,
    model,
    processor,
    prompt: str | list[str],
    threshold: float,
    exclude_robot_arms: bool = True,
    robot_arm_threshold: float = DEFAULT_ROBOT_ARM_THRESHOLD,
    robot_arm_overlap: float = DEFAULT_ROBOT_ARM_OVERLAP,
) -> tuple[list[dict], dict]:
    """Discover objects and remove candidates substantially covered by robot arms."""

    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    height, width = image.height, image.width
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    prompts = [value.strip() for value in prompts if value and value.strip()]
    if not prompts:
        prompts = ["object"]
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        state = _set_sam3_image(processor, image)
    detections_by_prompt = []
    for text_prompt in prompts:
        processor.set_confidence_threshold(max(0.0, threshold - 1e-7))
        processor.reset_all_prompts(state)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            output = processor.set_text_prompt(
                state=state, prompt=text_prompt
            )
        detections_by_prompt.append((text_prompt, _decode_grounding_detections(
            output, height, width, text_prompt, threshold
        )))
    detections, duplicate_detections = merge_cross_prompt_detections(
        detections_by_prompt
    )
    report = {
        "enabled": exclude_robot_arms,
        "prompts": list(ROBOT_ARM_PROMPTS),
        "threshold": robot_arm_threshold,
        "overlap_threshold": robot_arm_overlap,
        "regions_detected": 0,
        "objects_removed": 0,
        "objects_before_filter": len(detections),
        "removed": [],
        "semantic_discovery": {
            "strategy": PROMPT_STRATEGY,
            "prompts": prompts,
            "raw_counts": {
                text_prompt: len(items)
                for text_prompt, items in detections_by_prompt
            },
            "cross_prompt_duplicates_removed": len(duplicate_detections),
            "duplicate_matches": duplicate_detections,
        },
    }
    kept = detections
    if exclude_robot_arms and kept:
        robot_regions = _detect_robot_regions_from_state(
            state, processor, height, width, robot_arm_threshold
        )
        report["regions_detected"] = len(robot_regions)
        kept, report["removed"] = _filter_robot_arm_overlaps(
            kept, robot_regions, robot_arm_overlap
        )
    report["objects_removed"] = len(report["removed"])
    kept, boundary_removed = _filter_image_boundary_masks(kept)
    report["boundary_filter"] = {
        "enabled": True,
        "objects_removed": len(boundary_removed),
        "removed": boundary_removed,
    }
    return kept, report


def initial_mask_dir(video_path: Path) -> Path:
    return OUTPUT_ROOT / video_key(video_path) / "initial_sam3_sr_2k"


def discovery_frame_id(source_frame_index: int) -> str:
    return f"source_{max(0, int(source_frame_index)):06d}"


def manifest_discovery_frames(manifest: dict) -> list[dict]:
    """Return normalized multi-keyframe records, including legacy manifests."""
    frames = []
    seen = set()
    for raw in manifest.get("discovery_frames", []):
        try:
            source_index = int(raw["source_frame_index"])
            frame_path = str(raw["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        frame_id = str(raw.get("frame_id") or discovery_frame_id(source_index))
        if frame_id in seen:
            continue
        seen.add(frame_id)
        frames.append({**raw, "frame_id": frame_id, "frame": frame_path,
                       "source_frame_index": source_index})
    if not frames and manifest.get("frame"):
        metadata = dict(manifest.get("discovery_frame", {}))
        source_index = int(metadata.get("source_frame_index", 0))
        frames.append({
            **metadata,
            "frame_id": discovery_frame_id(source_index),
            "frame": manifest["frame"],
            "source_frame_index": source_index,
        })
    return sorted(frames, key=lambda item: int(item["source_frame_index"]))


def save_initial_masks(
    video_path: Path,
    frame_path: Path,
    detections: list[dict],
    prompt: str | list[str],
    threshold: float,
    robot_filter: dict | None = None,
    snapshot_existing: bool = True,
    revision: int = 0,
) -> None:
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    prompts = [value for value in prompts if value]
    base_prompt = prompts[0] if prompts else "object"
    directory = initial_mask_dir(video_path)
    directory.mkdir(parents=True, exist_ok=True)
    if snapshot_existing and (directory / "manifest.json").is_file():
        _snapshot_initial_masks(directory)
    if (directory / "manifest.json").is_file():
        for old_mask in directory.glob("object_*.png"):
            old_mask.unlink()
    objects = []
    for object_id, detection in enumerate(detections):
        mask_name = f"object_{object_id:04d}.png"
        Image.fromarray((detection["mask"] * 255).astype(np.uint8)).save(
            directory / mask_name
        )
        objects.append({
            "object_id": object_id,
            "mask": mask_name,
            "bbox": detection["bbox"],
            "score": detection["score"],
            "prompt": detection["prompt"],
            "matched_prompts": detection.get("matched_prompts", [{
                "prompt": detection["prompt"],
                "score": detection["score"],
            }]),
        })
    discovery_frame = {}
    extraction_marker = frame_path.parent / "extraction.json"
    try:
        extraction = json.loads(extraction_marker.read_text())
        discovery_frame = {
            key: extraction[key] for key in (
                "source_frame_index",
                "timestamp_seconds",
                "reason",
                "source_fps",
                "analysis_fps",
                "stable_duration_seconds",
                "settle_delay_seconds",
                "motion_threshold_ratio",
            ) if key in extraction
        }
    except (OSError, json.JSONDecodeError):
        pass
    source_frame_index = int(discovery_frame.get("source_frame_index", 0))
    frame_id = discovery_frame_id(source_frame_index)
    for item in objects:
        item["discovery_frame_id"] = frame_id
        item["source_frame_index"] = source_frame_index
        item["source_frame"] = portable_path(frame_path)
    discovery_frames = [{
        **discovery_frame,
        "frame_id": frame_id,
        "frame": portable_path(frame_path),
        "source_frame_index": source_frame_index,
        "robot_arm_filter": robot_filter or {"enabled": False},
        "boundary_filter": (
            (robot_filter or {}).get("boundary_filter", {"enabled": True})
        ),
    }]
    (directory / "manifest.json").write_text(json.dumps({
        "frame": portable_path(frame_path),
        "frame_extraction_method": FIRST_FRAME_EXTRACTION_METHOD,
        "discovery_frame": discovery_frame,
        "discovery_frames": discovery_frames,
        "active_discovery_frame_id": frame_id,
        "discovery_cache_version": DISCOVERY_CACHE_VERSION,
        "prompt": base_prompt,
        "prompts": prompts,
        "prompt_strategy": PROMPT_STRATEGY,
        "threshold": threshold,
        "robot_arm_filter": robot_filter or {"enabled": False},
        "boundary_filter": (
            (robot_filter or {}).get("boundary_filter", {"enabled": True})
        ),
        "revision": int(revision),
        "next_object_id": len(objects),
        "tracker_dirty": True,
        "objects": objects,
    }, indent=2, ensure_ascii=False))


def mark_initial_masks_tracked(video_path: Path, expected_revision: int) -> bool:
    """Mark masks current only when review did not change during tracking."""
    manifest_path = initial_mask_dir(video_path) / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("revision", 0)) != int(expected_revision):
        print(
            "Review changed while tracking; keeping tracker_dirty=true: "
            f"{video_path}",
            flush=True,
        )
        return False
    manifest["tracker_dirty"] = False
    manifest["tracked_revision"] = int(expected_revision)
    temp_path = manifest_path.with_suffix(".json.updating")
    temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    temp_path.replace(manifest_path)
    return True


def filter_dirty_tracking_videos(
    videos: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split videos into dirty, current, and unavailable review manifests."""
    dirty, current, unavailable = [], [], []
    for video_path in videos:
        manifest_path = initial_mask_dir(video_path) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Dirty-only mode is driven by saved review state. A video without
            # a readable review manifest has no reviewed mask change to track.
            unavailable.append(video_path)
            continue
        (dirty if bool(manifest.get("tracker_dirty", False)) else current).append(
            video_path
        )
    return dirty, current, unavailable


def _snapshot_initial_masks(directory: Path) -> None:
    """Create an undo-compatible snapshot before SAM3 changes manual masks."""
    snapshot = directory / ".history" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
        + uuid.uuid4().hex[:8]
    )
    snapshot.mkdir(parents=True)
    shutil.copy2(directory / "manifest.json", snapshot / "manifest.json")
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        frame_path = resolve_project_path(manifest["frame"])
        shutil.copy2(frame_path, snapshot / "discovery_frame.jpg")
        extraction = frame_path.parent / "extraction.json"
        if extraction.is_file():
            extraction_data = json.loads(extraction.read_text())
            extraction_data.update(manifest.get("discovery_frame", {}))
            extraction_data["method"] = manifest.get(
                "frame_extraction_method", extraction_data.get("method")
            )
            (snapshot / "extraction.json").write_text(
                json.dumps(extraction_data, indent=2, ensure_ascii=False)
            )
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    for mask_path in directory.glob("object_*.png"):
        shutil.copy2(mask_path, snapshot / mask_path.name)


def refine_manual_boxes(
    video_path: Path,
    model,
    processor,
    semantic_prompt: str,
    progress_callback: Callable[[int, int], None] | None = None,
    discovery_frame_id_filter: str | None = None,
) -> int:
    """Refine manual objects with SAM3's native box-and-point predictor."""
    directory = initial_mask_dir(video_path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No initial masks for {video_path}; run --stage discover first")
    manifest = json.loads(manifest_path.read_text())
    targets = [
        item for item in manifest.get("objects", [])
        if item.get("source") in {"manual_box", "sam3_box_refined"}
        and (
            discovery_frame_id_filter is None
            or str(item.get("discovery_frame_id") or discovery_frame_id(
                manifest.get("discovery_frame", {}).get("source_frame_index", 0)
            )) == discovery_frame_id_filter
        )
    ]
    if not targets:
        return 0

    frames = manifest_discovery_frames(manifest)
    frames_by_id = {item["frame_id"]: item for item in frames}
    target_frame_ids = {
        str(item.get("discovery_frame_id") or frames[0]["frame_id"])
        for item in targets
    }
    if len(target_frame_ids) != 1:
        raise RuntimeError(
            "Manual boxes span multiple discovery frames; refine one review frame at a time"
        )
    target_frame_id = next(iter(target_frame_ids))
    if target_frame_id not in frames_by_id:
        raise RuntimeError(f"Unknown discovery frame {target_frame_id}")
    with Image.open(resolve_project_path(frames_by_id[target_frame_id]["frame"])) as source:
        image = source.convert("RGB")
    width, height = image.size

    # Manual review is an explicit human target selection. Reuse one image
    # embedding for all boxes, but do not apply the automatic discovery-time
    # robot-arm overlap filter to these human-selected targets.
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        interactive_state = _set_sam3_image(processor, image)

    interactive_predictor = model.inst_interactive_predictor
    if interactive_predictor is None:
        raise RuntimeError(
            "SAM3 was loaded without its native interactive predictor; reload it "
            "with enable_inst_interactivity=True"
        )

    refined = []
    for target_index, item in enumerate(
        tqdm(targets, desc=f"SAM3 interactive refinement ({video_path.name})", leave=False),
        start=1,
    ):
            prompt_box = np.asarray(
                item.get("prompt_box", item["bbox"]), dtype=np.float32
            )
            x1, y1, x2, y2 = prompt_box
            if (
                not np.isfinite(prompt_box).all()
                or x2 <= x1
                or y2 <= y1
                or x2 <= 0
                or y2 <= 0
                or x1 >= width
                or y1 >= height
            ):
                raise RuntimeError(f"Invalid manual ROI {prompt_box.tolist()}")
            prompt_box = np.asarray([
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
                max(1, min(width, x2)),
                max(1, min(height, y2)),
            ], dtype=np.float32)

            positive_points = [
                (float(point[0]), float(point[1]))
                for point in item.get("prompt_points", [])
                if isinstance(point, (list, tuple)) and len(point) == 2
                and all(
                    isinstance(value, (int, float, np.number))
                    and math.isfinite(float(value))
                    for value in point
                )
            ]
            negative_points = [
                (float(point[0]), float(point[1]))
                for point in item.get("prompt_negative_points", [])
                if isinstance(point, (list, tuple)) and len(point) == 2
                and all(
                    isinstance(value, (int, float, np.number))
                    and math.isfinite(float(value))
                    for value in point
                )
            ]
            all_points = positive_points + negative_points
            point_coords = (
                np.asarray(all_points, dtype=np.float32) if all_points else None
            )
            point_labels = (
                np.asarray(
                    [1] * len(positive_points) + [0] * len(negative_points),
                    dtype=np.int32,
                )
                if all_points else None
            )
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, cache_enabled=False
            ):
                masks, scores, _ = model.predict_inst(
                    interactive_state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=prompt_box,
                    multimask_output=True,
                )
            masks = np.asarray(masks)
            scores = np.asarray(scores).reshape(-1)
            count = min(len(masks), len(scores))
            masks, scores = masks[:count], scores[:count]
            valid = np.asarray(
                [mask.shape == (height, width) and mask.any() for mask in masks],
                dtype=bool,
            )
            if not valid.any():
                raise RuntimeError(
                    f"SAM3 returned no interactive mask for {prompt_box.tolist()}"
                )
            ranking = scores.astype(np.float64).copy()
            ranking[~valid] = -np.inf
            best_index = int(np.argmax(ranking))
            full_mask = masks[best_index].astype(bool)
            mask_y, mask_x = np.nonzero(full_mask)
            full_box = np.asarray([
                mask_x.min(), mask_y.min(), mask_x.max() + 1, mask_y.max() + 1
            ], dtype=np.float64)
            refined.append((
                item,
                full_mask,
                full_box,
                scores[best_index],
                len(positive_points),
                len(negative_points),
                count,
            ))
            if progress_callback is not None:
                progress_callback(target_index, len(targets))

    _snapshot_initial_masks(directory)
    for (
        item, mask, box, score,
        positive_point_count,
        negative_point_count,
        candidate_count,
    ) in refined:
        mask_path = directory / Path(item["mask"]).name
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        item["prompt_box"] = [
            float(value) for value in item.get("prompt_box", item["bbox"])
        ]
        item["bbox"] = [float(value) for value in box]
        item["score"] = float(score)
        item["source"] = "sam3_box_refined"
        item["prompt"] = "interactive_box_points"
        item["refinement_prompt"] = {
            "mode": "native_interactive_box_points",
            "box_prompt": True,
            "positive_point_count": positive_point_count,
            "negative_point_count": negative_point_count,
            "multimask_output": True,
            "candidate_count": candidate_count,
            "ranking": "predicted_iou",
            "robot_overlap_filter": False,
        }
    manifest["revision"] = int(manifest.get("revision", 0)) + 1
    manifest["refined_at"] = datetime.now(timezone.utc).isoformat()
    manifest["tracker_dirty"] = True
    temp_path = manifest_path.with_suffix(".json.refining")
    temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    temp_path.replace(manifest_path)
    return len(refined)


def load_cached_initial_masks(
    video_path: Path,
    frame_path: Path,
    prompt: str,
    threshold: float,
    validate_discovery_settings: bool = True,
    exclude_robot_arms: bool = True,
    robot_arm_threshold: float = DEFAULT_ROBOT_ARM_THRESHOLD,
    robot_arm_overlap: float = DEFAULT_ROBOT_ARM_OVERLAP,
) -> list[dict] | None:
    expected_prompts = semantic_discovery_prompts(video_path, prompt)
    directory = initial_mask_dir(video_path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        extraction = json.loads((frame_path.parent / "extraction.json").read_text())
        robot_filter = manifest.get("robot_arm_filter", {})
        manual_revision = int(manifest.get("revision", 0)) > 0
        if (
            not same_project_path(manifest.get("frame", ""), frame_path)
            or int(manifest.get("discovery_frame", {}).get("source_frame_index", -1))
            != int(extraction.get("source_frame_index", -2))
            or (
                not manual_revision
                and (
                    manifest.get("frame_extraction_method")
                    != FIRST_FRAME_EXTRACTION_METHOD
                    or manifest.get("discovery_cache_version")
                    != DISCOVERY_CACHE_VERSION
                    or (
                        validate_discovery_settings
                        and (
                            manifest.get("prompt") != prompt
                            or manifest.get("prompts") != expected_prompts
                            or manifest.get("prompt_strategy") != PROMPT_STRATEGY
                            or float(manifest.get("threshold")) != threshold
                            or bool(robot_filter.get("enabled", False))
                            != exclude_robot_arms
                            or (
                                exclude_robot_arms
                                and (
                                    float(robot_filter.get("threshold", -1))
                                    != robot_arm_threshold
                                    or float(robot_filter.get(
                                        "overlap_threshold", -1
                                    )) != robot_arm_overlap
                                )
                            )
                        )
                    )
                )
            )
        ):
            return None
        detections = []
        for item in manifest.get("objects", []):
            mask_path = directory / item["mask"]
            if not mask_path.is_file():
                return None
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 127
            detections.append({
                "object_id": int(item.get("object_id", len(detections))),
                "bbox": item["bbox"],
                "mask": mask,
                "score": float(item["score"]),
                "prompt": item.get("prompt", prompt),
                "matched_prompts": item.get("matched_prompts", [{
                    "prompt": item.get("prompt", prompt),
                    "score": float(item["score"]),
                }]),
                "discovery_frame_id": item.get("discovery_frame_id"),
                "source_frame_index": int(item.get(
                    "source_frame_index",
                    manifest.get("discovery_frame", {}).get("source_frame_index", 0),
                )),
                "source_frame": item.get("source_frame", manifest.get("frame")),
            })
        return detections
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _base_discovery_cache_matches(
    video_path: Path,
    frame_path: Path,
    prompt: str,
    threshold: float,
) -> bool:
    """Return whether an old cache can be upgraded without replacing manual edits."""
    manifest_path = initial_mask_dir(video_path) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        extraction = json.loads((frame_path.parent / "extraction.json").read_text())
        return (
            same_project_path(manifest.get("frame", ""), frame_path)
            and manifest.get("frame_extraction_method")
            == FIRST_FRAME_EXTRACTION_METHOD
            and manifest.get("discovery_cache_version") == DISCOVERY_CACHE_VERSION
            and int(manifest.get("discovery_frame", {}).get("source_frame_index", -1))
            == int(extraction.get("source_frame_index", -2))
            and manifest.get("prompt") == prompt
            and manifest.get("prompts")
            == semantic_discovery_prompts(video_path, prompt)
            and manifest.get("prompt_strategy") == PROMPT_STRATEGY
            and float(manifest.get("threshold")) == threshold
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def filter_cached_robot_arms(
    video_path: Path,
    frame_path: Path,
    model,
    processor,
    robot_arm_threshold: float,
    robot_arm_overlap: float,
) -> tuple[list[dict], dict]:
    """Upgrade an existing reviewed cache by deleting only robot-arm masks."""
    directory = initial_mask_dir(video_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        state = _set_sam3_image(processor, image)
    robot_regions = _detect_robot_regions_from_state(
        state,
        processor,
        image.height,
        image.width,
        robot_arm_threshold,
    )

    detections = []
    for item in manifest.get("objects", []):
        mask_path = directory / item["mask"]
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 127
        detections.append({
            "bbox": item["bbox"],
            "mask": mask,
            "score": float(item["score"]),
            "prompt": item.get("prompt", manifest.get("prompt", "object")),
            "_manifest_item": item,
        })
    kept, removed = _filter_robot_arm_overlaps(
        detections, robot_regions, robot_arm_overlap
    )
    kept_detection_ids = {id(detection) for detection in kept}
    removed_items = [
        detection["_manifest_item"]
        for detection in detections
        if id(detection) not in kept_detection_ids
    ]
    report = {
        "enabled": True,
        "prompts": list(ROBOT_ARM_PROMPTS),
        "threshold": robot_arm_threshold,
        "overlap_threshold": robot_arm_overlap,
        "regions_detected": len(robot_regions),
        "objects_before_filter": len(detections),
        "objects_removed": len(removed_items),
        "removed": removed,
    }
    if removed_items:
        _snapshot_initial_masks(directory)
        for item in removed_items:
            mask_path = directory / Path(item["mask"]).name
            if mask_path.is_file():
                mask_path.unlink()
        manifest["objects"] = [
            item for item in manifest.get("objects", []) if item not in removed_items
        ]
        manifest["revision"] = int(manifest.get("revision", 0)) + 1
        manifest["tracker_dirty"] = True
    manifest["discovery_cache_version"] = DISCOVERY_CACHE_VERSION
    manifest["robot_arm_filter"] = report
    manifest["robot_arm_filtered_at"] = datetime.now(timezone.utc).isoformat()
    temp_path = manifest_path.with_suffix(".json.robot-filtering")
    temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    temp_path.replace(manifest_path)
    for detection in kept:
        detection.pop("_manifest_item", None)
    return kept, report


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def decode_outputs(outputs: dict) -> list[tuple[int, np.ndarray, float]]:
    ids = _as_numpy(outputs.get("out_obj_ids", [])).reshape(-1)
    masks = _as_numpy(outputs.get("out_binary_masks", []))
    probs = _as_numpy(outputs.get("out_probs", np.ones(len(ids)))).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    decoded = []
    for index, obj_id in enumerate(ids):
        confidence = float(probs[index]) if index < len(probs) else 1.0
        decoded.append((int(obj_id), masks[index].astype(bool), confidence))
    return decoded


def pack_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Store full-resolution masks at one bit per pixel while scoring tracks."""
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    return np.packbits(mask.reshape(-1)), shape


def unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    count = int(shape[0] * shape[1])
    return np.unpackbits(packed, count=count).reshape(shape).astype(bool)


def masked_sharpness(frame: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian inside the object, excluding its edge."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask_u8 = mask.astype(np.uint8)
    interior = cv2.erode(mask_u8, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if interior.sum() < 25:
        interior = mask
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    values = laplacian[interior]
    return float(values.var()) if values.size else 0.0


def touches_image_boundary(mask: np.ndarray, margin: int = 2) -> bool:
    if not mask.any():
        return True
    margin = max(1, min(margin, mask.shape[0] // 2, mask.shape[1] // 2))
    return bool(
        mask[:margin].any() or mask[-margin:].any()
        or mask[:, :margin].any() or mask[:, -margin:].any()
    )


def make_candidate(
    frame_index: int,
    frame_path: Path,
    frame: np.ndarray,
    mask: np.ndarray,
    confidence: float,
) -> FrameCandidate:
    area = int(mask.sum())
    return FrameCandidate(
        frame_index=frame_index,
        frame_path=str(frame_path),
        mask_area=area,
        area_ratio=float(area / mask.size),
        sharpness=masked_sharpness(frame, mask),
        confidence=max(0.0, min(1.0, confidence)),
        touches_boundary=touches_image_boundary(mask),
    )


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1 or np.allclose(values, values[0]):
        return np.ones_like(values, dtype=np.float64)
    low, high = np.percentile(values, [10, 90])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.ones_like(values, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def score_candidates(candidates: list[FrameCandidate]) -> None:
    """Set temporal stability and quality score in-place."""
    candidates.sort(key=lambda item: item.frame_index)
    areas = np.asarray([item.mask_area for item in candidates], dtype=np.float64)
    sharpness = np.asarray([item.sharpness for item in candidates], dtype=np.float64)
    area_scores = np.sqrt(areas / max(float(areas.max()), 1.0))
    sharpness_scores = _robust_normalize(np.log1p(sharpness))

    for index, item in enumerate(candidates):
        neighbor_areas = []
        if index > 0:
            neighbor_areas.append(areas[index - 1])
        if index + 1 < len(candidates):
            neighbor_areas.append(areas[index + 1])
        if neighbor_areas and item.mask_area > 0:
            reference = float(np.median(neighbor_areas))
            ratio = max(item.mask_area, reference) / max(min(item.mask_area, reference), 1.0)
            item.area_stability = float(math.exp(-abs(math.log(ratio))))
        else:
            item.area_stability = 1.0

        boundary_score = 0.0 if item.touches_boundary else 1.0
        total = sum(QUALITY_WEIGHTS.values())
        item.quality_score = float((
            QUALITY_WEIGHTS["sharpness"] * sharpness_scores[index]
            + QUALITY_WEIGHTS["area"] * area_scores[index]
            + QUALITY_WEIGHTS["confidence"] * item.confidence
            + QUALITY_WEIGHTS["stability"] * item.area_stability
            + QUALITY_WEIGHTS["boundary"] * boundary_score
        ) / total)


def filter_quality_candidates(candidates: list[FrameCandidate]) -> tuple[list[FrameCandidate], dict]:
    """Apply cheap hard filters; fall back safely if every frame is rejected."""
    if not candidates:
        return [], {"fallback": True, "rejected": {}}
    max_area = max(item.mask_area for item in candidates)
    sharpness_cutoff = float(np.quantile(
        [item.sharpness for item in candidates], QUALITY_FILTERS["min_sharpness_quantile"]
    ))
    accepted, rejected = [], {}
    for item in candidates:
        reasons = []
        if item.mask_area < max_area * QUALITY_FILTERS["min_relative_area"]:
            reasons.append("relative_area")
        if item.area_stability < QUALITY_FILTERS["min_stability"]:
            reasons.append("area_stability")
        if item.sharpness < sharpness_cutoff:
            reasons.append("sharpness")
        if QUALITY_FILTERS["exclude_boundary"] and item.touches_boundary:
            reasons.append("image_boundary")
        if reasons:
            rejected[str(item.frame_index)] = reasons
        else:
            accepted.append(item)
    fallback = not accepted
    return (accepted or list(candidates)), {
        "fallback": fallback,
        "input_count": len(candidates),
        "accepted_count": len(accepted) if not fallback else len(candidates),
        "sharpness_cutoff": sharpness_cutoff,
        "rejected": rejected,
        "settings": dict(QUALITY_FILTERS),
    }


def tight_crop(frame: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.08):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot crop an empty mask")
    mask_bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    x1, y1, x2, y2 = mask_bbox
    pad_x = max(4, int((x2 - x1) * pad_ratio))
    pad_y = max(4, int((y2 - y1) * pad_ratio))
    x1, x2 = max(0, x1 - pad_x), min(frame.shape[1], x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(frame.shape[0], y2 + pad_y)
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    masked = crop.copy()
    masked[~crop_mask] = 0
    return crop, masked, crop_mask, (x1, y1, x2, y2), mask_bbox


def export_choice(
    object_dir: Path,
    name: str,
    candidate: FrameCandidate,
    mask: np.ndarray,
    frame: np.ndarray,
    scene_path: Path,
) -> dict:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    context, masked, crop_mask, context_bbox, mask_bbox = tight_crop(frame, mask)
    Image.fromarray(context).save(object_dir / f"{name}_context.jpg", quality=95)
    # Keep the JPEG for existing caches and consumers, but use the lossless PNG in
    # human-facing tree cards so JPEG ringing cannot look like an enlarged mask.
    Image.fromarray(masked).save(object_dir / f"{name}.jpg", quality=95)
    Image.fromarray(masked).save(object_dir / f"{name}.png")
    Image.fromarray((crop_mask * 255).astype(np.uint8)).save(object_dir / f"{name}_mask.png")
    return {
        "scene_path": portable_path(scene_path),
        "scene_image_representation": "realesrgan_2k",
        "context_path": portable_path(object_dir / f"{name}_context.jpg"),
        "context_mask_path": portable_path(object_dir / f"{name}_mask.png"),
        "context_bbox": list(context_bbox),
        "mask_bbox": list(mask_bbox),
    }


def export_tracks(
    video_path: Path,
    frame_paths: list[Path],
    tracks: dict[int, list[FrameCandidate]],
    packed_masks: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]],
    sample_fps: float,
    prompt: str,
    tracker_backend: str,
    tracker_model: str,
) -> dict:
    output_dir = OUTPUT_ROOT / video_key(video_path) / f"tracker_{tracker_backend}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir.with_name(output_dir.name + f".building-{uuid.uuid4().hex}")
    build_dir.mkdir()
    scene_dir = build_dir / "representative_scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "video": portable_path(video_path),
        "sample_fps": sample_fps,
        "sampled_frame_count": len(frame_paths),
        "tracking_image_representation": "original_video_frames",
        "representative_image_representation": "realesrgan_2k",
        "prompt": prompt,
        "tracker_backend": tracker_backend,
        "tracker_model": tracker_model,
        "objects": [],
    }
    build_prefix = portable_path(build_dir)
    final_prefix = portable_path(output_dir)

    def final_paths(value):
        if isinstance(value, dict):
            return {key: final_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [final_paths(item) for item in value]
        if isinstance(value, str) and value.startswith(build_prefix + "/"):
            return final_prefix + value[len(build_prefix):]
        return value

    try:
        for obj_id, candidates in sorted(tracks.items()):
            score_candidates(candidates)
            eligible, filter_report = filter_quality_candidates(candidates)
            best = max(eligible, key=lambda item: (item.quality_score, item.mask_area))
            object_dir = build_dir / f"object_{obj_id:04d}"
            object_dir.mkdir(parents=True, exist_ok=True)
            best_packed, best_shape = packed_masks[(obj_id, best.frame_index)]
            scene_path = scene_dir / f"frame_{best.frame_index:06d}.jpg"
            if not scene_path.is_file():
                from super_resolution import upscale_for_sam3

                with Image.open(best.frame_path) as source:
                    enhanced = upscale_for_sam3(source.convert("RGB"))
                enhanced.save(scene_path, quality=95)
            with Image.open(scene_path) as source:
                representative_frame = np.asarray(source.convert("RGB")).copy()
            context_metadata = export_choice(
                object_dir,
                "best_quality",
                best,
                unpack_mask(best_packed, best_shape),
                representative_frame,
                scene_path,
            )
            track_data = final_paths({
                "object_id": obj_id,
                "best_quality": {**candidate_metadata(best), **context_metadata},
                "quality_weights": dict(QUALITY_WEIGHTS),
                "quality_filter": filter_report,
                "candidates": [candidate_metadata(item) for item in candidates],
            })
            (object_dir / "track.json").write_text(
                json.dumps(track_data, indent=2, ensure_ascii=False)
            )
            manifest["objects"].append(track_data)

        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        archive = None
        if output_dir.exists():
            archive_root = output_dir.parent / "_tracker_archive"
            archive_root.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive = archive_root / f"{output_dir.name}_{stamp}_{uuid.uuid4().hex[:6]}"
            output_dir.replace(archive)
        try:
            build_dir.replace(output_dir)
        except Exception:
            if archive is not None and archive.exists() and not output_dir.exists():
                archive.replace(output_dir)
            raise
        return manifest
    except Exception:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        raise


def process_video_sam31_mask(
    video_path: Path,
    tracker,
    frame_paths: list[Path],
    detections: list[dict],
    sample_fps: float,
    output_threshold: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Track the exact reviewed masks with the official SAM 3.1 multiplex tracker."""
    if not detections:
        raise RuntimeError(f"No reviewed objects to track in {video_path}")
    tracks: dict[int, list[FrameCandidate]] = {}
    packed_masks: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]] = {}
    from sam3.model.io_utils import load_video_frames

    images, video_height, video_width = load_video_frames(
        video_path=str(frame_paths[0].parent),
        image_size=tracker.image_size,
        offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    inference_state = tracker.init_state(
        video_height=video_height,
        video_width=video_width,
        num_frames=len(images),
        offload_state_to_cpu=True,
        offload_video_to_cpu=True,
    )
    inference_state["images"] = images
    try:
        try:
            extraction = json.loads(
                (frame_paths[0].parent / "extraction.json").read_text()
            )
            source_frame_indices = [
                int(value) for value in extraction["source_frame_indices"]
            ]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            source_frame_indices = list(range(len(frame_paths)))
        if len(source_frame_indices) != len(frame_paths):
            raise RuntimeError("Sampled-frame source index map is inconsistent")
        source_to_sample = {
            source_index: sample_index
            for sample_index, source_index in enumerate(source_frame_indices)
        }
        masks_by_sample_frame: dict[int, list[tuple[int, dict, np.ndarray]]] = {}
        for fallback_obj_id, detection in enumerate(detections):
            obj_id = int(detection.get("object_id", fallback_obj_id))
            source_frame_index = int(detection.get(
                "source_frame_index", source_frame_indices[0]
            ))
            if source_frame_index not in source_to_sample:
                raise RuntimeError(
                    f"Discovery source frame {source_frame_index} was not extracted"
                )
            mask = detection["mask"].astype(bool)
            if not mask.any():
                raise RuntimeError(f"Reviewed mask {obj_id} is empty")
            masks_by_sample_frame.setdefault(source_to_sample[source_frame_index], []).append(
                (obj_id, detection, mask)
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for sample_frame_index, frame_detections in sorted(
                masks_by_sample_frame.items()
            ):
                with Image.open(frame_paths[sample_frame_index]) as image:
                    discovery_image = np.asarray(image.convert("RGB"))
                obj_ids = []
                initial_masks = []
                for obj_id, detection, mask in frame_detections:
                    if mask.shape != discovery_image.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (discovery_image.shape[1], discovery_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    obj_ids.append(obj_id)
                    initial_masks.append(torch.from_numpy(mask.copy()))
                    tracks[obj_id] = [make_candidate(
                        sample_frame_index,
                        frame_paths[sample_frame_index],
                        discovery_image,
                        mask,
                        float(detection["score"]),
                    )]
                    packed_masks[(obj_id, sample_frame_index)] = pack_mask(mask)
                tracker.add_new_masks(
                    inference_state=inference_state,
                    frame_idx=sample_frame_index,
                    obj_ids=obj_ids,
                    masks=torch.stack(initial_masks),
                    add_mask_to_memory=True,
                )
            tracker.propagate_in_video_preflight(
                inference_state,
                run_mem_encoder=True,
            )
            if progress_callback is not None:
                progress_callback(1, len(frame_paths))

            stream = tracker.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=max(0, len(frame_paths) - 1),
                reverse=False,
                tqdm_disable=True,
            )
            for output in tqdm(
                stream, total=len(frame_paths), desc=video_path.name, leave=False
            ):
                frame_index, object_ids, _, masks, object_scores = output
                frame_index = int(frame_index)
                if frame_index == 0 or not 0 <= frame_index < len(frame_paths):
                    continue
                with Image.open(frame_paths[frame_index]) as image:
                    frame = np.asarray(image.convert("RGB"))
                masks = _as_numpy(masks)
                scores = torch.sigmoid(object_scores).detach().float().cpu().numpy().reshape(-1)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                for index, obj_id in enumerate(object_ids):
                    obj_id = int(obj_id)
                    if (obj_id, frame_index) in packed_masks:
                        continue
                    mask = masks[index] > 0
                    if not mask.any():
                        continue
                    confidence = float(scores[index]) if index < len(scores) else 1.0
                    tracks.setdefault(obj_id, []).append(make_candidate(
                        frame_index,
                        frame_paths[frame_index],
                        frame,
                        mask,
                        confidence,
                    ))
                    packed_masks[(obj_id, frame_index)] = pack_mask(mask)
                if progress_callback is not None:
                    progress_callback(frame_index + 1, len(frame_paths))
    finally:
        del inference_state
        gc.collect()
        torch.cuda.empty_cache()

    if progress_callback is not None:
        progress_callback(len(frame_paths), len(frame_paths))
    return export_tracks(
        video_path,
        frame_paths,
        tracks,
        packed_masks,
        sample_fps,
        "reviewed_object_masks",
        "sam3",
        "sam3.1_multiplex",
    )


def discover_initial_masks(
    videos: list[Path],
    first_frame_paths_by_video: dict[Path, Path],
    checkpoint: Path,
    gpu: int,
    prompt: str,
    threshold: float,
    exclude_robot_arms: bool,
    robot_arm_threshold: float,
    robot_arm_overlap: float,
) -> tuple[dict[Path, list[dict]], dict[str, list]]:
    """Discover uncached first frames and report videos that could not finish."""
    detections_by_video = {}
    report: dict[str, list] = {
        "reused": [],
        "detected": [],
        "no_objects": [],
        "failed": [],
        "robot_arms_removed": [],
        "boundary_masks_removed": [],
    }
    uncached = []
    cached_to_filter = []
    for video_path in videos:
        frame_path = first_frame_paths_by_video[video_path]
        detections = load_cached_initial_masks(
            video_path,
            frame_path,
            prompt,
            threshold,
            exclude_robot_arms=exclude_robot_arms,
            robot_arm_threshold=robot_arm_threshold,
            robot_arm_overlap=robot_arm_overlap,
        )
        if detections is None:
            if exclude_robot_arms and _base_discovery_cache_matches(
                video_path, frame_path, prompt, threshold
            ):
                cached_to_filter.append(video_path)
            else:
                uncached.append(video_path)
        else:
            detections_by_video[video_path] = detections
            report["reused"].append(video_path)
            if not detections:
                report["no_objects"].append(video_path)
            try:
                cached_manifest = json.loads(
                    (initial_mask_dir(video_path) / "manifest.json").read_text()
                )
                removed = int(
                    cached_manifest.get("robot_arm_filter", {}).get(
                        "objects_removed", 0
                    )
                )
                if removed:
                    report["robot_arms_removed"].append((video_path, removed))
                boundary_removed = int(
                    cached_manifest.get("boundary_filter", {}).get(
                        "objects_removed", 0
                    )
                )
                if boundary_removed:
                    report["boundary_masks_removed"].append(
                        (video_path, boundary_removed)
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            print(f"Reusing {len(detections)} initial masks: {video_path.name}")

    if not uncached and not cached_to_filter:
        return detections_by_video, report
    print("Loading SAM3 image detector for first-frame discovery…", flush=True)
    detector, detector_processor = load_sam3_detector(checkpoint, gpu)
    try:
        for video_path in tqdm(
            cached_to_filter, desc="SAM3 robot-arm cache cleanup"
        ):
            frame_path = first_frame_paths_by_video[video_path]
            try:
                detections, robot_filter = filter_cached_robot_arms(
                    video_path,
                    frame_path,
                    detector,
                    detector_processor,
                    robot_arm_threshold,
                    robot_arm_overlap,
                )
            except Exception as exc:
                reason = f"robot-arm cache cleanup: {type(exc).__name__}: {exc}"
                report["failed"].append((video_path, reason))
                tqdm.write(f"  Cache cleanup failed; skipped: {video_path}\n    {reason}")
                if isinstance(exc, torch.OutOfMemoryError):
                    gc.collect()
                    torch.cuda.empty_cache()
                continue
            detections_by_video[video_path] = detections
            report["reused"].append(video_path)
            if not detections:
                report["no_objects"].append(video_path)
            if robot_filter["objects_removed"]:
                report["robot_arms_removed"].append(
                    (video_path, robot_filter["objects_removed"])
                )
            tqdm.write(
                f"  Preserved reviewed cache: {len(detections)} objects, "
                f"removed {robot_filter['objects_removed']} robot-arm object(s): "
                f"{video_path}"
            )
        for video_path in tqdm(uncached, desc="SAM3 first-frame discovery"):
            frame_path = first_frame_paths_by_video[video_path]
            prompts = semantic_discovery_prompts(video_path, prompt)
            try:
                detections, robot_filter = detect_first_frame(
                    frame_path,
                    detector,
                    detector_processor,
                    prompts,
                    threshold,
                    exclude_robot_arms=exclude_robot_arms,
                    robot_arm_threshold=robot_arm_threshold,
                    robot_arm_overlap=robot_arm_overlap,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                report["failed"].append((video_path, reason))
                tqdm.write(f"  Detection failed; skipped: {video_path}\n    {reason}")
                if isinstance(exc, torch.OutOfMemoryError):
                    gc.collect()
                    torch.cuda.empty_cache()
                continue
            if not detections:
                report["no_objects"].append(video_path)
                # Persist the new discovery frame even when SAM3 finds nothing.
                # Otherwise an incompatible old first-frame manifest could stay
                # visible in review and be tracked against the wrong source frame.
                save_initial_masks(
                    video_path,
                    frame_path,
                    detections,
                    prompts,
                    threshold,
                    robot_filter,
                )
                if robot_filter["objects_removed"]:
                    report["robot_arms_removed"].append(
                        (video_path, robot_filter["objects_removed"])
                    )
                    tqdm.write(
                        f"  Removed {robot_filter['objects_removed']} robot-arm "
                        f"object(s); no task objects remain: {video_path}"
                    )
                else:
                    tqdm.write(f"  No objects detected; skipped: {video_path}")
                boundary_removed = int(
                    robot_filter.get("boundary_filter", {}).get("objects_removed", 0)
                )
                if boundary_removed:
                    report["boundary_masks_removed"].append(
                        (video_path, boundary_removed)
                    )
                continue
            save_initial_masks(
                video_path,
                frame_path,
                detections,
                prompts,
                threshold,
                robot_filter,
            )
            if robot_filter["objects_removed"]:
                report["robot_arms_removed"].append(
                    (video_path, robot_filter["objects_removed"])
                )
            boundary_removed = int(
                robot_filter.get("boundary_filter", {}).get("objects_removed", 0)
            )
            if boundary_removed:
                report["boundary_masks_removed"].append(
                    (video_path, boundary_removed)
                )
            detections_by_video[video_path] = detections
            report["detected"].append(video_path)
            tqdm.write(
                f"  {video_path.name}: {len(detections)} initial objects, "
                f"removed {robot_filter['objects_removed']} robot-arm object(s)"
            )
    finally:
        del detector_processor
        del detector
        gc.collect()
        torch.cuda.empty_cache()
    return detections_by_video, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--exclude-robot-arms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="detect robot arms/grippers/hands with SAM3 and remove overlapping objects",
    )
    parser.add_argument(
        "--robot-arm-threshold",
        type=float,
        default=DEFAULT_ROBOT_ARM_THRESHOLD,
        help="SAM3 confidence threshold for robot-arm exclusion prompts",
    )
    parser.add_argument(
        "--robot-arm-overlap",
        type=float,
        default=DEFAULT_ROBOT_ARM_OVERLAP,
        help="remove an object when this fraction of its mask overlaps robot-arm masks",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quality-sharpness", type=float, default=0.45)
    parser.add_argument("--quality-area", type=float, default=0.20)
    parser.add_argument("--quality-confidence", type=float, default=0.15)
    parser.add_argument("--quality-stability", type=float, default=0.10)
    parser.add_argument("--quality-boundary", type=float, default=0.10)
    parser.add_argument("--quality-min-relative-area", type=float, default=0.25)
    parser.add_argument("--quality-min-stability", type=float, default=0.50)
    parser.add_argument("--quality-min-sharpness-quantile", type=float, default=0.10)
    parser.add_argument(
        "--quality-exclude-boundary", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="skip videos that fail during batch tracking and report them at the end",
    )
    parser.add_argument(
        "--dirty-only",
        action="store_true",
        help="during batch tracking, process only manifests marked tracker_dirty",
    )
    parser.add_argument(
        "--stage", choices=("discover", "refine", "track"), default="discover",
        help=(
            "discover: SAM3 source frame zero only; refine: SAM3 re-segments manual boxes; "
            "track: track the currently saved masks"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM3_CKPT", str(DEFAULT_CKPT))),
        help="SAM3 checkpoint used for first-frame discovery and box refinement.",
    )
    parser.add_argument(
        "--sam31-checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM31_CKPT", str(DEFAULT_SAM31_CKPT))),
        help="SAM 3.1 multiplex checkpoint used for reviewed-mask tracking.",
    )
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if not 0 <= args.robot_arm_threshold <= 1:
        parser.error("--robot-arm-threshold must be between 0 and 1")
    if not 0 <= args.robot_arm_overlap <= 1:
        parser.error("--robot-arm-overlap must be between 0 and 1")
    if not 0 <= args.quality_min_relative_area <= 1:
        parser.error("--quality-min-relative-area must be between 0 and 1")
    if not 0 <= args.quality_min_stability <= 1:
        parser.error("--quality-min-stability must be between 0 and 1")
    if not 0 <= args.quality_min_sharpness_quantile <= 1:
        parser.error("--quality-min-sharpness-quantile must be between 0 and 1")
    weights = {
        "sharpness": args.quality_sharpness,
        "area": args.quality_area,
        "confidence": args.quality_confidence,
        "stability": args.quality_stability,
        "boundary": args.quality_boundary,
    }
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        parser.error("quality weights must be non-negative and not all zero")
    QUALITY_WEIGHTS.update(weights)
    QUALITY_FILTERS.update({
        "min_relative_area": args.quality_min_relative_area,
        "min_stability": args.quality_min_stability,
        "min_sharpness_quantile": args.quality_min_sharpness_quantile,
        "exclude_boundary": args.quality_exclude_boundary,
    })
    print(f"Quality weights: {QUALITY_WEIGHTS}")

    selection_report = None
    if args.video:
        videos = [args.video]
    else:
        videos, selection_report = find_videos()
    videos = [path.resolve() for path in videos if path and path.is_file()]
    if args.limit:
        videos = videos[:args.limit]
    if selection_report is not None:
        atexit.register(print_video_selection_report, selection_report, videos)
    if not videos:
        raise SystemExit(
            "No input videos found. Download RoboCOIN data or pass --video /path/to/video.mp4."
        )

    current_videos_skipped: list[Path] = []
    unavailable_videos_skipped: list[Path] = []
    if args.stage == "track" and args.dirty_only:
        (
            videos,
            current_videos_skipped,
            unavailable_videos_skipped,
        ) = filter_dirty_tracking_videos(videos)
        print(
            f"Dirty-only tracking: {len(videos)} need updates; "
            f"{len(current_videos_skipped)} already current; "
            f"{len(unavailable_videos_skipped)} without review state. Skipped both.",
            flush=True,
        )
        if not videos:
            emit_progress(100, "所有数据的跟踪均已是最新")
            print("\n=== Tracking report ===")
            print("Tracked videos:   0")
            print(f"Already current:  {len(current_videos_skipped)}")
            print(f"No review state:  {len(unavailable_videos_skipped)}")
            print("No SAM 3.1 model was loaded.")
            return

    requested_videos = list(videos)
    tracking_failures: list[tuple[Path, str]] = []
    frame_paths_by_video: dict[Path, list[Path]] = {}
    first_frame_paths_by_video: dict[Path, Path] = {}
    discovery_preflight_failures: list[tuple[Path, str]] = []
    if args.stage == "discover":
        for video_path in videos:
            frame_dir = OUTPUT_ROOT / video_key(video_path) / "first_frame_sr_2k"
            try:
                first_frame_paths_by_video[video_path] = extract_first_frame(
                    video_path, frame_dir
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                discovery_preflight_failures.append((video_path, reason))
                print(f"First-frame extraction failed; skipped: {video_path}")
                print(f"  {reason}")
    elif args.stage == "track":
        emit_progress(0, f"准备 {len(videos)} 个视频")
        for video_index, video_path in enumerate(videos, start=1):
            if (
                args.continue_on_error
                and not (initial_mask_dir(video_path) / "manifest.json").is_file()
            ):
                tracking_failures.append(
                    (video_path, "no initial masks; run --stage discover first")
                )
                print(f"No initial masks; skipped before frame extraction: {video_path}")
                emit_progress(
                    8 * video_index / max(1, len(requested_videos)),
                    f"抽帧与检查 {video_index}/{len(requested_videos)}",
                )
                continue
            frame_dir = OUTPUT_ROOT / video_key(video_path) / "frames"#在输出创建相应输出文件夹
            try:
                manifest_path = initial_mask_dir(video_path) / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                discovery_frames = manifest_discovery_frames(manifest)
                required_source_frame_indices = sorted({
                    int(item["source_frame_index"]) for item in discovery_frames
                }) or [0]
                start_source_frame_index = min(required_source_frame_indices)
                frame_paths_by_video[video_path] = extract_sampled_frames(
                    video_path,
                    frame_dir,
                    args.sample_fps,
                    start_source_frame_index=start_source_frame_index,
                    required_source_frame_indices=required_source_frame_indices,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                reason = f"frame extraction: {type(exc).__name__}: {exc}"
                tracking_failures.append((video_path, reason))
                print(f"Tracking preflight failed; skipped: {video_path}")
                print(f"  {reason}")
            emit_progress(
                8 * video_index / max(1, len(requested_videos)),
                f"抽帧与检查 {video_index}/{len(requested_videos)}",
            )
        if args.continue_on_error:
            videos = list(frame_paths_by_video)

    if args.stage == "discover":
        discover_videos = list(first_frame_paths_by_video)
        detections, discovery_report = discover_initial_masks(
            discover_videos,
            first_frame_paths_by_video,
            args.checkpoint,
            args.gpu,
            args.prompt,
            args.threshold,
            args.exclude_robot_arms,
            args.robot_arm_threshold,
            args.robot_arm_overlap,
        )
        discovery_report["failed"].extend(discovery_preflight_failures)
        count = sum(len(items) for items in detections.values())
        reused_count = len(discovery_report["reused"])
        detected_count = len(discovery_report["detected"])
        no_object_videos = discovery_report["no_objects"]
        failed_videos = discovery_report["failed"]
        robot_arms_removed = discovery_report["robot_arms_removed"]
        boundary_masks_removed = discovery_report["boundary_masks_removed"]
        print("\n=== Discovery report ===")
        print(f"Requested videos: {len(videos)}")
        print(f"Newly detected:   {detected_count}")
        print(f"Reused cache:     {reused_count}")
        print(f"No objects:       {len(no_object_videos)}")
        print(f"Failed/skipped:   {len(failed_videos)}")
        print(f"Object masks:     {count}")
        print(
            "Robot-arm masks removed: "
            f"{sum(removed for _, removed in robot_arms_removed)}"
        )
        if robot_arms_removed:
            print("Robot-arm removals by video:")
            for video_path, removed in robot_arms_removed:
                print(f"  - {video_path}: {removed}")
        print(
            "Boundary-touching masks removed: "
            f"{sum(removed for _, removed in boundary_masks_removed)}"
        )
        if no_object_videos:
            print("Videos with no detected objects:")
            for video_path in no_object_videos:
                print(f"  - {video_path}")
        if failed_videos:
            print("Videos skipped after processing errors:")
            for video_path, reason in failed_videos:
                print(f"  - {video_path}")
                print(f"    {reason}")
        print("Tracking was not started. Review at: http://127.0.0.1:8888/review")
        return

    if args.stage == "refine":
        target_count = 0
        for video_path in videos:
            manifest_path = initial_mask_dir(video_path) / "manifest.json"
            if not manifest_path.is_file():
                raise SystemExit(
                    f"No initial masks for {video_path}; run --stage discover first."
                )
            manifest = json.loads(manifest_path.read_text())
            target_count += sum(
                item.get("source") in {"manual_box", "sam3_box_refined"}
                for item in manifest.get("objects", [])
            )
        if not target_count:
            print("No manual boxes need SAM3 refinement. Return to the review window.")
            return
        print(f"Loading SAM3 to refine {target_count} manual box(es)…", flush=True)
        emit_progress(5, f"加载 SAM3，准备细化 {target_count} 个人工框")
        detector, detector_processor = load_sam3_detector(
            args.checkpoint,
            args.gpu,
            enable_inst_interactivity=True,
        )
        refined_count = 0
        try:
            for video_path in videos:
                manifest = json.loads(
                    (initial_mask_dir(video_path) / "manifest.json").read_text()
                )
                default_frame_id = manifest_discovery_frames(manifest)[0]["frame_id"]
                target_frame_ids = sorted({
                    str(item.get("discovery_frame_id") or default_frame_id)
                    for item in manifest.get("objects", [])
                    if item.get("source") in {"manual_box", "sam3_box_refined"}
                })
                for target_frame_id in target_frame_ids:
                    completed_before_frame = refined_count

                    def report_refinement(current: int, _total: int) -> None:
                        completed = completed_before_frame + current
                        emit_progress(
                            10 + 90 * completed / max(1, target_count),
                            f"细化人工框 {completed}/{target_count}",
                        )

                    refined_count += refine_manual_boxes(
                        video_path,
                        detector,
                        detector_processor,
                        semantic_prompt=args.prompt,
                        progress_callback=report_refinement,
                        discovery_frame_id_filter=target_frame_id,
                    )
        finally:
            del detector_processor
            del detector
            gc.collect()
            torch.cuda.empty_cache()
        print(f"SAM3 refined {refined_count} manual box(es).")
        emit_progress(100, f"SAM3 细化完成：{refined_count} 个人工框")
        print("Review the refined masks again: http://127.0.0.1:8888/review")
        return

    summaries = []
    detections_by_video = {}
    review_revisions_by_video: dict[Path, int] = {}
    for video_path in videos:
        manifest_path = initial_mask_dir(video_path) / "manifest.json"
        if not manifest_path.is_file():
            reason = "no initial masks; run --stage discover first"
            if args.continue_on_error:
                tracking_failures.append((video_path, reason))
                print(f"No initial masks; skipped: {video_path}")
                continue
            raise SystemExit(f"No initial masks for {video_path}; run --stage discover first.")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            if args.continue_on_error:
                reason = f"invalid initial-mask manifest: {type(exc).__name__}: {exc}"
                tracking_failures.append((video_path, reason))
                print(f"Invalid initial-mask manifest; skipped: {video_path}")
                continue
            raise SystemExit(f"Invalid initial-mask manifest for {video_path}") from exc
        try:
            discovery_frame_path = resolve_project_path(manifest["frame"])
        except (KeyError, TypeError):
            if args.continue_on_error:
                tracking_failures.append((video_path, "manifest has no valid discovery frame"))
                print(f"Invalid discovery-frame path; skipped: {video_path}")
                continue
            raise SystemExit(f"Invalid initial-mask manifest for {video_path}")
        detections = load_cached_initial_masks(
            video_path,
            discovery_frame_path,
            args.prompt,
            args.threshold,
            validate_discovery_settings=False,
        )
        if detections is None:
            reason = "saved masks do not match the current discovery settings"
            if args.continue_on_error:
                tracking_failures.append((video_path, reason))
                print(f"No compatible initial masks; skipped: {video_path}")
                continue
            raise SystemExit(f"No compatible initial masks for {video_path}")
        if not detections:
            reason = "initial-mask manifest contains no objects"
            if args.continue_on_error:
                tracking_failures.append((video_path, reason))
                print(f"No objects to track; skipped: {video_path}")
                continue
            raise SystemExit(f"No objects to track for {video_path}")
        detections_by_video[video_path] = detections
        review_revisions_by_video[video_path] = int(manifest.get("revision", 0))
        print(f"Using {len(detections)} currently saved masks: {video_path}")

    videos = list(detections_by_video)
    if not videos:
        print("\n=== Tracking report ===")
        print(f"Requested videos: {len(requested_videos)}")
        print("Tracked videos:   0")
        print(f"Failed/skipped:   {len(tracking_failures)}")
        if args.dirty_only:
            print(f"Already current:  {len(current_videos_skipped)}")
            print(f"No review state:  {len(unavailable_videos_skipped)}")
        for video_path, reason in tracking_failures:
            print(f"  - {video_path}\n    {reason}")
        emit_progress(100, "没有可跟踪的视频；请查看跳过报告")
        return

    emit_progress(10, f"已准备 {len(videos)} 个可跟踪视频")
    print("Loading official SAM 3.1 multiplex mask tracker…", flush=True)
    emit_progress(12, "正在加载 SAM 3.1 multiplex 掩码跟踪模型")
    tracker = load_sam31_mask_tracker(args.sam31_checkpoint, args.gpu)
    total_frames = sum(len(frame_paths_by_video[path]) for path in videos)
    completed_frames = 0
    for video_index, video_path in enumerate(
        tqdm(videos, desc="Videos (SAM 3.1 mask tracker)"), start=1
    ):
        frames_in_video = len(frame_paths_by_video[video_path])

        def report_tracking(current: int, _total: int) -> None:
            overall = completed_frames + min(current, frames_in_video)
            emit_progress(
                12 + 86 * overall / max(1, total_frames),
                f"跟踪视频 {video_index}/{len(videos)}，帧 {current}/{frames_in_video}",
            )

        try:
            summary = process_video_sam31_mask(
                video_path,
                tracker,
                frame_paths_by_video[video_path],
                detections_by_video[video_path],
                args.sample_fps,
                args.threshold,
                progress_callback=report_tracking,
            )
            mark_initial_masks_tracked(
                video_path, review_revisions_by_video[video_path]
            )
            summaries.append(summary)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            reason = f"tracking: {type(exc).__name__}: {exc}"
            tracking_failures.append((video_path, reason))
            print(f"Tracking failed; skipped: {video_path}")
            print(f"  {reason}")
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            completed_frames += frames_in_video

    object_count = sum(len(item["objects"]) for item in summaries)
    print(
        f"Processed {len(summaries)} video(s) with SAM 3.1 mask tracking, "
        f"exported {object_count} object track(s)"
    )
    if args.continue_on_error:
        print("\n=== Tracking report ===")
        print(f"Requested videos: {len(requested_videos)}")
        print(f"Tracked videos:   {len(summaries)}")
        print(f"Failed/skipped:   {len(tracking_failures)}")
        print(f"Already current:  {len(current_videos_skipped)}")
        if args.dirty_only:
            print(f"No review state:  {len(unavailable_videos_skipped)}")
        print(f"Object tracks:    {object_count}")
        for video_path, reason in tracking_failures:
            print(f"  - {video_path}\n    {reason}")
    print(f"Results: {OUTPUT_ROOT}")
    emit_progress(
        100,
        f"跟踪完成：{len(summaries)}/{len(requested_videos)} 个视频，"
        f"{object_count} 条物体轨迹",
    )


if __name__ == "__main__":
    main()
