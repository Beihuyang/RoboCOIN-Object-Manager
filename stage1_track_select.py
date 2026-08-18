#!/usr/bin/env python3
"""Track first-frame objects and export two representative frames per object.

The dataset contract for this stage is that every object of interest is visible
in the first frame. SAM 3 detects the objects once on frame 0 and propagates
their masks over uniformly sampled frames. For every track the script exports:

* ``largest``: the frame with the largest mask area.
* ``best_quality``: the frame with the best combined sharpness, area,
  confidence, temporal area stability, and image-boundary score.
* ``comparison.jpg``: a labelled side-by-side preview of both choices.

The two choices are intentionally kept separate so a human can decide which
selection policy works better before either is used as the canonical asset.

Usage:
    python stage1_track_select.py --limit 1 --sample-fps 5
    python stage1_track_select.py --video /path/to/video.mp4 --sample-fps 5
    python stage1_track_select.py --checkpoint /path/to/sam3.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "RoboCOIN_datasets"
OUTPUT_ROOT = BASE_DIR / "objects" / "tracks"
SAM3_REPO = BASE_DIR / "sam3"
DEFAULT_CKPT = BASE_DIR / "checkpoints" / "sam3" / "sam3.pt"
BPE_PATH = SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CAMERA_KEYWORDS = ("head", "center", "high")


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


def find_videos(video_root: Path = VIDEO_ROOT) -> list[Path]:
    videos = []
    for path in sorted(video_root.rglob("episode_000000.mp4")):
        text = str(path).lower()
        if "wrist" in text or "chunk-000" not in text:
            continue
        if any(keyword in text for keyword in CAMERA_KEYWORDS):
            videos.append(path)
    return videos


def video_key(video_path: Path) -> Path:
    """Return a stable, filesystem-safe relative key for output directories."""
    try:
        relative = video_path.resolve().relative_to(VIDEO_ROOT.resolve())
        return relative.with_suffix("")
    except ValueError:
        return Path(video_path.stem)


def extract_sampled_frames(video_path: Path, frame_dir: Path, sample_fps: float) -> list[Path]:
    """Extract sequential JPEG frames, reusing a completed extraction."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    marker = frame_dir / "extraction.json"
    existing = sorted(frame_dir.glob("*.jpg"))
    if marker.exists() and existing:
        try:
            metadata = json.loads(marker.read_text())
            if (
                metadata.get("source") == str(video_path.resolve())
                and float(metadata.get("sample_fps")) == sample_fps
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

    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-vf", f"fps={sample_fps}", "-q:v", "2",
        str(temp_dir / "%06d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr.strip()}")

    extracted = sorted(temp_dir.glob("*.jpg"))
    if not extracted:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # SAM's folder loader accepts numeric stems. Start at zero so the filename
    # and SAM frame index match exactly.
    for index, path in enumerate(extracted):
        target = temp_dir / f"{index:06d}.jpg"
        if path != target:
            path.rename(target)

    for old in frame_dir.glob("*.jpg"):
        old.unlink()
    for path in sorted(temp_dir.glob("*.jpg")):
        path.rename(frame_dir / path.name)
    temp_dir.rmdir()
    marker.write_text(json.dumps({
        "source": str(video_path.resolve()),
        "sample_fps": sample_fps,
        "frame_count": len(extracted),
    }, indent=2))
    return sorted(frame_dir.glob("*.jpg"))


def load_predictor(checkpoint: Path | None, gpu: int):
    torch.cuda.set_device(gpu)
    sys.path.insert(0, str(SAM3_REPO))
    from sam3.model_builder import build_sam3_video_predictor

    kwargs = {"bpe_path": str(BPE_PATH), "gpus_to_use": [gpu]}
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM 3 checkpoint not found: {checkpoint}. "
                "Pass --checkpoint /path/to/sam3.pt."
            )
        kwargs["checkpoint_path"] = str(checkpoint)
    # With no checkpoint path, the official builder attempts the gated HF
    # download. This is useful on machines that have already authenticated.
    return build_sam3_video_predictor(**kwargs)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
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
        item.quality_score = float(
            0.45 * sharpness_scores[index]
            + 0.20 * area_scores[index]
            + 0.15 * item.confidence
            + 0.10 * item.area_stability
            + 0.10 * boundary_score
        )


def select_candidates(candidates: list[FrameCandidate]) -> tuple[FrameCandidate, FrameCandidate]:
    if not candidates:
        raise ValueError("Cannot select from an empty track")
    score_candidates(candidates)
    largest = max(candidates, key=lambda item: (item.mask_area, item.quality_score))
    best = max(candidates, key=lambda item: (item.quality_score, item.mask_area))
    return largest, best


def tight_crop(frame: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.08):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot crop an empty mask")
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x = max(4, int((x2 - x1) * pad_ratio))
    pad_y = max(4, int((y2 - y1) * pad_ratio))
    x1, x2 = max(0, x1 - pad_x), min(frame.shape[1], x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(frame.shape[0], y2 + pad_y)
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    masked = crop.copy()
    masked[~crop_mask] = 0
    return crop, masked, crop_mask


def _preview(masked: np.ndarray, title: str, metrics: FrameCandidate, size=(480, 480)) -> Image.Image:
    image = Image.fromarray(masked).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 72), "#151515")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, size[1] + 8), title, fill="white")
    draw.text(
        (10, size[1] + 32),
        f"frame={metrics.frame_index}  area={metrics.mask_area}  "
        f"sharp={metrics.sharpness:.1f}  quality={metrics.quality_score:.3f}",
        fill="#bbbbbb",
    )
    return canvas


def export_choice(
    object_dir: Path,
    name: str,
    candidate: FrameCandidate,
    mask: np.ndarray,
) -> np.ndarray:
    frame = np.asarray(Image.open(candidate.frame_path).convert("RGB"))
    context, masked, crop_mask = tight_crop(frame, mask)
    Image.fromarray(context).save(object_dir / f"{name}_context.jpg", quality=95)
    Image.fromarray(masked).save(object_dir / f"{name}.jpg", quality=95)
    Image.fromarray((crop_mask * 255).astype(np.uint8)).save(object_dir / f"{name}_mask.png")
    return masked


def process_video(
    video_path: Path,
    predictor,
    sample_fps: float,
    prompt: str,
    output_threshold: float,
) -> dict:
    output_dir = OUTPUT_ROOT / video_key(video_path)
    frame_dir = output_dir / "frames"
    frame_paths = extract_sampled_frames(video_path, frame_dir, sample_fps)

    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(frame_dir),
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
    })
    session_id = response["session_id"]
    tracks: dict[int, list[FrameCandidate]] = {}
    masks: dict[tuple[int, int], np.ndarray] = {}

    try:
        predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": output_threshold,
        })
        stream = predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": len(frame_paths),
            "output_prob_thresh": output_threshold,
        })
        for result in tqdm(stream, total=len(frame_paths), desc=video_path.name, leave=False):
            frame_index = int(result["frame_index"])
            if not 0 <= frame_index < len(frame_paths):
                continue
            frame = np.asarray(Image.open(frame_paths[frame_index]).convert("RGB"))
            for obj_id, mask, confidence in decode_outputs(result["outputs"]):
                if not mask.any():
                    continue
                candidate = make_candidate(
                    frame_index, frame_paths[frame_index], frame, mask, confidence
                )
                tracks.setdefault(obj_id, []).append(candidate)
                masks[(obj_id, frame_index)] = mask
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    manifest = {
        "video": str(video_path),
        "sample_fps": sample_fps,
        "sampled_frame_count": len(frame_paths),
        "prompt": prompt,
        "objects": [],
    }
    for obj_id, candidates in sorted(tracks.items()):
        largest, best = select_candidates(candidates)
        object_dir = output_dir / f"object_{obj_id:04d}"
        object_dir.mkdir(parents=True, exist_ok=True)
        largest_masked = export_choice(
            object_dir, "largest", largest, masks[(obj_id, largest.frame_index)]
        )
        best_masked = export_choice(
            object_dir, "best_quality", best, masks[(obj_id, best.frame_index)]
        )
        comparison = Image.new("RGB", (960, 552), "#151515")
        comparison.paste(_preview(largest_masked, "LARGEST MASK", largest), (0, 0))
        comparison.paste(_preview(best_masked, "BEST QUALITY", best), (480, 0))
        comparison.save(object_dir / "comparison.jpg", quality=95)

        track_data = {
            "object_id": obj_id,
            "largest": asdict(largest),
            "best_quality": asdict(best),
            "candidates": [asdict(item) for item in candidates],
        }
        (object_dir / "track.json").write_text(
            json.dumps(track_data, indent=2, ensure_ascii=False)
        )
        manifest["objects"].append(track_data)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM3_CKPT", str(DEFAULT_CKPT))),
    )
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")

    videos = [args.video] if args.video else find_videos()
    videos = [path.resolve() for path in videos if path and path.is_file()]
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(
            "No input videos found. Download RoboCOIN data or pass --video /path/to/video.mp4."
        )

    predictor = load_predictor(args.checkpoint, args.gpu)
    summaries = []
    for video_path in tqdm(videos, desc="Videos"):
        summaries.append(process_video(
            video_path, predictor, args.sample_fps, args.prompt, args.threshold
        ))

    object_count = sum(len(item["objects"]) for item in summaries)
    print(f"Processed {len(summaries)} video(s), exported {object_count} object track(s)")
    print(f"Results: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
