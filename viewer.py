#!/usr/bin/env python3
"""
Object Library Viewer — browse the deduplicated object library.

Usage:
    pip install fastapi uvicorn jinja2
    python viewer.py
    # → open http://localhost:8888
"""

import json
import html as html_lib
import os
import signal
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from functools import lru_cache
from urllib.parse import quote
from copy import deepcopy
from io import BytesIO
from datetime import datetime, timezone
from typing import Literal
from pathlib import Path

# Configure CUDA allocation before any lazily imported model module loads torch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from dedup_tree import (
    backfill_layout_instance_fingerprints,
    generate_layout as generate_dedup_tree_layout,
    inherit_unchanged_adjustments,
    load_layout as load_dedup_tree_layout,
    move_instance as move_dedup_tree_instance,
    rebuild_library as rebuild_library_from_tree,
    save_layout as save_dedup_tree_layout,
    trash_instance as trash_dedup_tree_instance,
    TREE_PATH as DEDUP_TREE_PATH,
)
from project_paths import portable_path, resolve_project_path
from semantic_prompts import PROMPT_STRATEGY

BASE_DIR = Path(__file__).resolve().parent
LIBRARY_INDEX = BASE_DIR / "objects" / "new_library" / "index.json"
LIBRARY_STATE = BASE_DIR / "objects" / "new_library" / "build_state.json"
LIBRARY_DIR = BASE_DIR / "objects" / "new_library"
CROPS_DIR = BASE_DIR / "objects" / "crops"
TRACKS_DIR = BASE_DIR / "objects" / "tracks"
WORK_DIR = BASE_DIR / "objects" / "new_library_work"
DEDUP_CANDIDATES = WORK_DIR / "dedup_candidates.json"
DEDUP_DECISIONS = WORK_DIR / "dedup_decisions.json"
ATTRIBUTE_OVERRIDES = WORK_DIR / "attribute_overrides.json"
REVIEW_TEMPLATE = BASE_DIR / "templates" / "review.html"
DEDUP_TEMPLATE = BASE_DIR / "templates" / "dedup.html"
OBJECT_LINK_TEMPLATE = BASE_DIR / "templates" / "object_links.html"
OBJECT_LINK_MANIFEST = BASE_DIR / "objects" / "object_text_links" / "manifest.json"
OBJECT_LINK_DATASETS = BASE_DIR / "RoboCOIN_datasets"
OBJECT_LINK_PREVIEWS = BASE_DIR / "objects" / "object_text_links" / "previews"

# StaticFiles requires its target directories to exist even when the library is
# still empty (for example before the first successful Stage 1 run).
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Serve crop images and library (canonical) images
app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/library", StaticFiles(directory=str(LIBRARY_DIR)), name="library")
app.mount("/tracks", StaticFiles(directory=str(TRACKS_DIR)), name="tracks")

EDIT_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
DEDUP_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
PROGRESS_PREFIX = "@@PROGRESS "
SAM3_MODEL_LOCK = threading.RLock()
SAM3_DETECTOR = None
SAM3_PROCESSOR = None


@app.on_event("shutdown")
def _shutdown_models() -> None:
    _unload_resident_sam3()


def _job_return_code(job: dict) -> int | None:
    process = job.get("process")
    if process is not None:
        return process.poll()
    if job.get("thread") is not None and job["thread"].is_alive():
        return None
    return_code = job.get("return_code")
    return int(return_code) if return_code is not None else 1


def _job_is_running(job: dict) -> bool:
    return _job_return_code(job) is None


def _assert_no_model_job() -> None:
    """Prevent metadata/library mutations while a model subprocess owns them."""
    with JOB_LOCK:
        if any(_job_is_running(job) for job in JOBS.values()):
            raise HTTPException(status_code=409, detail="模型任务运行期间不能修改物体库或去重状态")


def _append_job_log(job: dict, message: str) -> None:
    with job["log_path"].open("a") as log_file:
        log_file.write(message.rstrip() + "\n")
        log_file.flush()


def _resident_sam3(gpu: int = 0, require_interactivity: bool = False):
    """Lazily load and retain the SAM3 image model used by review refinement."""
    global SAM3_DETECTOR, SAM3_PROCESSOR
    if (
        require_interactivity
        and SAM3_DETECTOR is not None
        and SAM3_DETECTOR.inst_interactive_predictor is None
    ):
        _unload_resident_sam3()
    if SAM3_DETECTOR is None or SAM3_PROCESSOR is None:
        from stage1_track_select import DEFAULT_CKPT, load_sam3_detector

        checkpoint = Path(os.environ.get("SAM3_CKPT", str(DEFAULT_CKPT)))
        SAM3_DETECTOR, SAM3_PROCESSOR = load_sam3_detector(
            checkpoint,
            gpu,
            enable_inst_interactivity=require_interactivity,
        )
    return SAM3_DETECTOR, SAM3_PROCESSOR


def _unload_resident_sam3() -> bool:
    """Release the review model before another GPU-heavy pipeline starts."""
    global SAM3_DETECTOR, SAM3_PROCESSOR
    with SAM3_MODEL_LOCK:
        if SAM3_DETECTOR is None and SAM3_PROCESSOR is None:
            return False
        SAM3_DETECTOR = None
        SAM3_PROCESSOR = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True


def _run_resident_refine_job(
    job_id: str,
    video_path: Path,
    semantic_prompt: str,
    frame_id: str | None,
) -> None:
    job = JOBS[job_id]
    cancel_event = job["cancel_event"]
    return_code = 1
    try:
        with SAM3_MODEL_LOCK:
            if cancel_event.is_set():
                raise InterruptedError("Task cancelled before SAM3 refinement")
            was_loaded = (
                SAM3_DETECTOR is not None
                and SAM3_PROCESSOR is not None
                and SAM3_DETECTOR.inst_interactive_predictor is not None
            )
            _append_job_log(job, PROGRESS_PREFIX + json.dumps({
                "percent": 5,
                "message": "复用常驻 SAM3" if was_loaded else "首次加载 SAM3",
            }, ensure_ascii=False))
            detector, processor = _resident_sam3(require_interactivity=True)

            from stage1_track_select import refine_manual_boxes

            def report_progress(current: int, total: int) -> None:
                if cancel_event.is_set():
                    raise InterruptedError("Task cancelled during SAM3 refinement")
                _append_job_log(job, PROGRESS_PREFIX + json.dumps({
                    "percent": 10 + 90 * current / max(1, total),
                    "message": f"原生框点细化 {current}/{total}",
                }, ensure_ascii=False))

            with EDIT_LOCK:
                refined = refine_manual_boxes(
                    video_path,
                    detector,
                    processor,
                    semantic_prompt=semantic_prompt,
                    progress_callback=report_progress,
                    discovery_frame_id_filter=frame_id,
                )
            _append_job_log(job, f"SAM3 refined {refined} manual box(es).")
            _append_job_log(job, PROGRESS_PREFIX + json.dumps({
                "percent": 100,
                "message": f"原生框点细化完成：{refined} 个人工框",
            }, ensure_ascii=False))
            return_code = 0
    except InterruptedError as exc:
        job["cancel_requested"] = True
        _append_job_log(job, str(exc))
        return_code = 130
    except Exception:
        _append_job_log(job, traceback.format_exc())
        return_code = 1
    finally:
        job["return_code"] = return_code


def _run_resident_keyframe_job(
    job_id: str,
    directory: Path,
    video_path: Path,
    timestamp_seconds: float,
) -> None:
    """Extract a chosen frame, discover objects, and append a review keyframe."""
    job = JOBS[job_id]
    cancel_event = job["cancel_event"]
    candidate_path = directory / f".keyframe-{job_id}.jpg"
    return_code = 1
    snapshot_created = False
    try:
        from stage1_track_select import (
            detect_first_frame,
            discovery_frame_id,
            discovery_frame_metadata,
            enhance_frame_to_2k,
            extract_source_frame_image,
            _video_stream_geometry,
        )

        manifest, state = _load_review_state(directory)
        source_fps, _, _ = _video_stream_geometry(video_path)
        duration = _video_duration_seconds(video_path)
        timestamp_seconds = min(timestamp_seconds, max(0.0, duration - 1 / source_fps))
        source_frame_index = max(0, round(timestamp_seconds * source_fps))
        frame_id = discovery_frame_id(source_frame_index)
        if any(
            item["frame_id"] == frame_id for item in _manifest_review_frames(manifest)
        ):
            raise RuntimeError(
                f"源帧 {source_frame_index} 已经是发现关键帧，请选择其他时间"
            )
        _append_job_log(job, PROGRESS_PREFIX + json.dumps({
            "percent": 5,
            "message": f"正在提取 {source_frame_index / source_fps:.1f}s 关键帧",
        }, ensure_ascii=False))
        extract_source_frame_image(video_path, source_frame_index, candidate_path)
        enhancement = enhance_frame_to_2k(candidate_path)
        if cancel_event.is_set():
            raise InterruptedError("Task cancelled before keyframe discovery")

        with SAM3_MODEL_LOCK:
            was_loaded = SAM3_DETECTOR is not None and SAM3_PROCESSOR is not None
            _append_job_log(job, PROGRESS_PREFIX + json.dumps({
                "percent": 15,
                "message": "复用常驻 SAM3" if was_loaded else "首次加载 SAM3",
            }, ensure_ascii=False))
            detector, processor = _resident_sam3()
            robot_filter = manifest.get("robot_arm_filter", {})
            prompt = str(manifest.get("prompt") or "object")
            prompts = manifest.get("prompts")
            if not isinstance(prompts, list) or not prompts:
                from stage1_track_select import semantic_discovery_prompts

                prompts = semantic_discovery_prompts(video_path, prompt)
            threshold = float(manifest.get("threshold", 0.2))
            detections, robot_report = detect_first_frame(
                candidate_path,
                detector,
                processor,
                prompts,
                threshold,
                exclude_robot_arms=bool(robot_filter.get("enabled", True)),
                robot_arm_threshold=float(robot_filter.get("threshold", 0.10)),
                robot_arm_overlap=float(robot_filter.get("overlap_threshold", 0.50)),
            )
        if cancel_event.is_set():
            raise InterruptedError("Task cancelled after keyframe discovery")

        _append_job_log(job, PROGRESS_PREFIX + json.dumps({
            "percent": 85,
            "message": f"SAM3 发现 {len(detections)} 个候选，正在保存",
        }, ensure_ascii=False))
        with EDIT_LOCK:
            _snapshot_review_state(directory)
            snapshot_created = True
            current_frame_path = resolve_project_path(manifest["frame"])
            frame_path = current_frame_path.parent / f"source_{source_frame_index:06d}.jpg"
            candidate_path.replace(frame_path)
            metadata = discovery_frame_metadata(
                video_path, source_frame_index, "manual_keyframe_append"
            )
            metadata["image_representation"] = "realesrgan_2k"
            metadata["super_resolution"] = enhancement
            frame_record = {
                **metadata,
                "frame_id": frame_id,
                "frame": portable_path(frame_path),
                "robot_arm_filter": robot_report,
                "boundary_filter": robot_report.get(
                    "boundary_filter", {"enabled": True}
                ),
            }
            appended_state = list(state)
            for detection in detections:
                appended_state.append(({
                    "score": float(detection["score"]),
                    "prompt": detection.get("prompt", prompt),
                    "matched_prompts": detection.get("matched_prompts", []),
                    "source": "sam3_appended_keyframe",
                    "discovery_frame_id": frame_id,
                    "source_frame_index": source_frame_index,
                    "source_frame": portable_path(frame_path),
                }, detection["mask"].astype(bool)))
            updated_manifest = dict(manifest)
            updated_manifest["prompts"] = prompts
            updated_manifest["prompt_strategy"] = PROMPT_STRATEGY
            updated_manifest["discovery_frames"] = [
                *_manifest_review_frames(manifest), frame_record
            ]
            updated_manifest["active_discovery_frame_id"] = frame_id
            _write_review_state(
                directory, updated_manifest, appended_state
            )
        _append_job_log(job, PROGRESS_PREFIX + json.dumps({
            "percent": 100,
            "message": f"已追加关键帧：{source_frame_index / source_fps:.1f}s，"
                       f"新增 {len(detections)} 个候选",
        }, ensure_ascii=False))
        return_code = 0
    except InterruptedError as exc:
        job["cancel_requested"] = True
        _append_job_log(job, str(exc))
        return_code = 130
    except Exception:
        if snapshot_created:
            try:
                with EDIT_LOCK:
                    _restore_latest_snapshot(directory)
            except Exception:
                _append_job_log(job, "Failed to restore keyframe snapshot:\n" + traceback.format_exc())
        _append_job_log(job, traceback.format_exc())
        return_code = 1
    finally:
        if candidate_path.is_file():
            candidate_path.unlink()
        job["return_code"] = return_code


def _parse_progress(log_text: str) -> dict | None:
    for fragment in reversed(log_text.split(PROGRESS_PREFIX)[1:]):
        candidate = fragment.splitlines()[0].strip().strip("\r")
        try:
            parsed = json.loads(candidate)
            return {
                "percent": float(parsed["percent"]),
                "message": str(parsed["message"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _stop_job_process(job: dict) -> None:
    """Stop a model job and every child process it created."""
    process = job.get("process")
    if process is None:
        if not _job_is_running(job):
            return
        job["cancel_requested"] = True
        job["cancel_event"].set()
        _append_job_log(job, "Task cancellation requested by user.")
        return
    if process.poll() is not None:
        return
    job["cancel_requested"] = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    try:
        with job["log_path"].open("a") as log_file:
            log_file.write("\nTask cancelled by user.\n")
    except OSError:
        pass


class ReviewAction(BaseModel):
    session: str
    action: Literal[
        "add_box", "add_point", "add_negative_point", "clear_points", "merge",
        "delete", "delete_keyframe", "undo",
    ]
    object_ids: list[int] = Field(default_factory=list)
    box: list[float] | None = None
    point: list[float] | None = None
    frame_id: str | None = None


class ReviewJobRequest(BaseModel):
    session: str
    kind: Literal["refine", "track", "keyframe"]
    timestamp_seconds: float | None = Field(default=None, ge=0)
    frame_id: str | None = None
    quality_sharpness: float = Field(default=0.45, ge=0)
    quality_area: float = Field(default=0.20, ge=0)
    quality_confidence: float = Field(default=0.15, ge=0)
    quality_stability: float = Field(default=0.10, ge=0)
    quality_boundary: float = Field(default=0.10, ge=0)
    quality_min_relative_area: float = Field(default=0.25, ge=0, le=1)
    quality_min_stability: float = Field(default=0.50, ge=0, le=1)
    quality_min_sharpness_quantile: float = Field(default=0.10, ge=0, le=1)
    quality_exclude_boundary: bool = True


class ReviewBatchTrackRequest(BaseModel):
    quality_sharpness: float = Field(default=0.45, ge=0)
    quality_area: float = Field(default=0.20, ge=0)
    quality_confidence: float = Field(default=0.15, ge=0)
    quality_stability: float = Field(default=0.10, ge=0)
    quality_boundary: float = Field(default=0.10, ge=0)
    quality_min_relative_area: float = Field(default=0.25, ge=0, le=1)
    quality_min_stability: float = Field(default=0.50, ge=0, le=1)
    quality_min_sharpness_quantile: float = Field(default=0.10, ge=0, le=1)
    quality_exclude_boundary: bool = True


class DedupDecision(BaseModel):
    pair_id: str
    decision: Literal["merge", "reject", "pending"]
    canonical_instance_id: str | None = None


class DedupClusterDecision(BaseModel):
    instance_ids: list[str] = Field(min_length=2)
    selected_ids: list[str] = Field(default_factory=list)
    decision: Literal["merge", "reject", "merge_selected", "exclude_selected"]


class DedupSettings(BaseModel):
    clip_recall_threshold: float = Field(ge=-1, le=1)
    attribute_threshold: float = Field(default=0.50, ge=0, le=1)


class DedupTreeMove(BaseModel):
    instance_id: str
    target_cluster_id: str | None = None
    target_node_id: str | None = None


class DedupTreeTrash(BaseModel):
    instance_id: str


class DedupTreeRegenerate(BaseModel):
    clip_threshold: float = Field(default=0.82, ge=-1, le=1)
    attribute_threshold: float = Field(default=0.50, ge=0, le=1)


class AttributeUpdate(BaseModel):
    attributes: dict


class ObjectLinkDecision(BaseModel):
    selected_ids: list[str] = Field(default_factory=list)
    ignored: bool = False


class VlmRunRequest(BaseModel):
    tracker: Literal["sam3"] = "sam3"
    force: bool = False


def _review_sessions() -> list[dict]:
    sessions = []
    for manifest_path in sorted(TRACKS_DIR.rglob("initial_sam3_sr_2k/manifest.json")):
        directory = manifest_path.parent
        try:
            key = directory.relative_to(TRACKS_DIR).as_posix()
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        sessions.append({
            "key": key,
            "label": key.removesuffix("/initial_sam3_sr_2k"),
            "object_count": len(manifest.get("objects", [])),
            "tracker_dirty": bool(manifest.get("tracker_dirty", False)),
        })
    return sessions


def _resolve_review_session(key: str) -> Path:
    try:
        directory = (TRACKS_DIR / key).resolve()
        directory.relative_to(TRACKS_DIR.resolve())
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid review session")
    if directory.name != "initial_sam3_sr_2k" or not (directory / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="Review session not found")
    return directory


def _load_review_state(directory: Path) -> tuple[dict, list[tuple[dict, np.ndarray]]]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest.pop("top_background_filter", None)
        for frame in manifest.get("discovery_frames", []):
            if isinstance(frame, dict):
                frame.pop("top_background_filter", None)
        state = []
        shape = None
        for item in manifest.get("objects", []):
            mask_path = directory / Path(item["mask"]).name
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 127
            if shape is not None and mask.shape != shape:
                raise ValueError("Mask sizes do not match")
            shape = mask.shape
            cleaned = dict(item)
            cleaned.pop("review_hidden", None)
            cleaned.pop("filter_reason", None)
            cleaned.pop("spatial_filter_metrics", None)
            state.append((cleaned, mask))
        return manifest, state
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid mask manifest: {exc}")


def _manifest_review_frames(manifest: dict) -> list[dict]:
    """Normalize legacy single-frame and current multi-keyframe manifests."""
    from stage1_track_select import manifest_discovery_frames

    return manifest_discovery_frames(manifest)


def _item_frame_id(item: dict, manifest: dict) -> str:
    frames = _manifest_review_frames(manifest)
    default_id = frames[0]["frame_id"] if frames else "source_000000"
    return str(item.get("discovery_frame_id") or default_id)


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise HTTPException(status_code=400, detail="Empty masks cannot be saved")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _snapshot_review_state(directory: Path) -> None:
    history = directory / ".history"
    snapshot = history / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]
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


def _write_review_state(
    directory: Path, manifest: dict, state: list[tuple[dict, np.ndarray]]
) -> None:
    temp_dir = directory / (".editing-" + uuid.uuid4().hex)
    temp_dir.mkdir()
    objects = []
    existing_ids = {
        int(item["object_id"])
        for item, _ in state
        if isinstance(item.get("object_id"), int) and int(item["object_id"]) >= 0
    }
    next_object_id = max(
        int(manifest.get("next_object_id", 0)),
        max(existing_ids, default=-1) + 1,
    )
    assigned_ids = set()
    try:
        for item, mask in state:
            candidate_id = item.get("object_id")
            if (
                isinstance(candidate_id, int)
                and candidate_id >= 0
                and candidate_id not in assigned_ids
            ):
                object_id = candidate_id
            else:
                while next_object_id in assigned_ids:
                    next_object_id += 1
                object_id = next_object_id
                next_object_id += 1
            assigned_ids.add(object_id)
            mask_name = f"object_{object_id:04d}.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(temp_dir / mask_name)
            saved = dict(item)
            saved.update({
                "object_id": object_id,
                "mask": mask_name,
                "bbox": _mask_bbox(mask),
            })
            objects.append(saved)

        updated = dict(manifest)
        updated["objects"] = objects
        updated["next_object_id"] = next_object_id
        updated["revision"] = int(manifest.get("revision", 0)) + 1
        updated["edited_at"] = datetime.now(timezone.utc).isoformat()
        updated["tracker_dirty"] = True
        (temp_dir / "manifest.json").write_text(
            json.dumps(updated, indent=2, ensure_ascii=False)
        )

        expected = {item["mask"] for item in objects}
        for path in temp_dir.glob("object_*.png"):
            path.replace(directory / path.name)
        for old_path in directory.glob("object_*.png"):
            if old_path.name not in expected:
                old_path.unlink()
        (temp_dir / "manifest.json").replace(directory / "manifest.json")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _restore_latest_snapshot(directory: Path) -> None:
    history = directory / ".history"
    snapshots = sorted(path for path in history.glob("*") if path.is_dir())
    if not snapshots:
        raise HTTPException(status_code=409, detail="没有可以撤销的操作")
    snapshot = snapshots[-1]
    for path in directory.glob("object_*.png"):
        path.unlink()
    for path in snapshot.glob("object_*.png"):
        shutil.copy2(path, directory / path.name)
    if (snapshot / "discovery_frame.jpg").is_file():
        restored_manifest = json.loads((snapshot / "manifest.json").read_text())
        frame_path = resolve_project_path(restored_manifest["frame"])
        shutil.copy2(snapshot / "discovery_frame.jpg", frame_path)
        if (snapshot / "extraction.json").is_file():
            shutil.copy2(snapshot / "extraction.json", frame_path.parent / "extraction.json")
    shutil.copy2(snapshot / "manifest.json", directory / "manifest.json")
    shutil.rmtree(snapshot)


def _review_payload(key: str, directory: Path, frame_id: str | None = None) -> dict:
    manifest, state = _load_review_state(directory)
    frames = _manifest_review_frames(manifest)
    if not frames:
        raise HTTPException(status_code=500, detail="Manifest has no discovery frame")
    frames_by_id = {item["frame_id"]: item for item in frames}
    requested_frame_id = str(
        frame_id or manifest.get("active_discovery_frame_id") or frames[0]["frame_id"]
    )
    if requested_frame_id not in frames_by_id:
        raise HTTPException(status_code=404, detail="Discovery frame not found")
    active_frame = frames_by_id[requested_frame_id]
    try:
        frame_path = resolve_project_path(active_frame["frame"])
        frame_relative = frame_path.relative_to(TRACKS_DIR.resolve()).as_posix()
        frame_url = f"/tracks/{frame_relative}?v={frame_path.stat().st_mtime_ns}"
    except (KeyError, ValueError, OSError):
        raise HTTPException(status_code=500, detail="Discovery frame is outside tracks directory")
    objects = []
    for item, mask in state:
        if _item_frame_id(item, manifest) != requested_frame_id:
            continue
        object_id = int(item["object_id"])
        mask_path = directory / Path(item["mask"]).name
        payload = {
            "object_id": object_id,
            "mask_url": f"/tracks/{key}/{mask_path.name}?v={mask_path.stat().st_mtime_ns}",
            "bbox": _mask_bbox(mask),
            "score": float(item.get("score", 1.0)),
            "source": item.get("source", "sam3"),
            "prompt_points": [
                [float(point[0]), float(point[1])]
                for point in item.get("prompt_points", [])
                if isinstance(point, (list, tuple)) and len(point) == 2
            ],
            "prompt_negative_points": [
                [float(point[0]), float(point[1])]
                for point in item.get("prompt_negative_points", [])
                if isinstance(point, (list, tuple)) and len(point) == 2
            ],
        }
        objects.append(payload)
    video_path, _ = _session_video_and_fps(directory)
    return {
        "key": key,
        "frame_url": frame_url,
        "active_frame_id": requested_frame_id,
        "active_discovery_frame": active_frame,
        "discovery_frames": [{
            **item,
            "object_count": sum(
                _item_frame_id(obj, manifest) == item["frame_id"]
                for obj, _ in state
            ),
        } for item in frames],
        "objects": objects,
        "total_object_count": len(state),
        "revision": int(manifest.get("revision", 0)),
        "tracker_dirty": bool(manifest.get("tracker_dirty", False)),
        "discovery_frame": active_frame,
        "video_duration_seconds": _video_duration_seconds(video_path),
        "robot_arm_filter": active_frame.get(
            "robot_arm_filter", manifest.get("robot_arm_filter", {})
        ),
        "can_undo": any((directory / ".history").glob("*")),
    }


def _session_video_and_fps(directory: Path) -> tuple[Path, float]:
    extraction_path = directory.parent / "frames" / "extraction.json"
    if not extraction_path.is_file():
        extraction_path = directory.parent / "first_frame_sr_2k" / "extraction.json"
    if not extraction_path.is_file():
        extraction_path = directory.parent / "first_frame" / "extraction.json"
    try:
        metadata = json.loads(extraction_path.read_text())
        video_path = resolve_project_path(metadata["source"])
        sample_fps = float(metadata.get("sample_fps", 1.0))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid frame extraction metadata: {exc}")
    if not video_path.is_file() or sample_fps <= 0:
        raise HTTPException(status_code=500, detail="Source video or sample FPS is invalid")
    return video_path, sample_fps


def _video_duration_seconds(video_path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ], capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 0.0
    return max(0.0, duration)


def load_library():
    if not LIBRARY_INDEX.exists():
        return {}
    from stage4_dedup import items_fingerprint, load_attributes

    try:
        state = json.loads(LIBRARY_STATE.read_text())
        current = items_fingerprint(load_attributes())
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=f"物体库状态不可验证，请重新建库：{exc}")
    if state.get("source_fingerprint") != current:
        raise HTTPException(status_code=409, detail="物体库已过期，请补全属性并重新生成类别树")
    with open(LIBRARY_INDEX) as f:
        return json.load(f)


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)


def _restore_file(path: Path, previous: bytes | None) -> None:
    """Rollback a state file after a dependent rebuild fails."""
    if previous is None:
        if path.exists():
            path.unlink()
        return
    temp = path.with_name(path.name + ".rollback-" + uuid.uuid4().hex)
    temp.write_bytes(previous)
    temp.replace(path)


def _dedup_settings() -> dict:
    defaults = {
        "clip_recall_threshold": 0.88,
        "attribute_threshold": 0.50,
    }
    if not DEDUP_CANDIDATES.is_file():
        return defaults
    try:
        stored = json.loads(DEDUP_CANDIDATES.read_text())
        return {key: float(stored.get(key, value)) for key, value in defaults.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return defaults


def _run_library_build(
    settings: dict | None = None, reuse_candidates: bool = False
) -> str:
    _unload_resident_sam3()
    settings = settings or _dedup_settings()
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable, str(BASE_DIR / "stage4_dedup.py"),
        "--clip-recall-threshold", str(settings["clip_recall_threshold"]),
        "--attribute-threshold", str(settings["attribute_threshold"]),
    ]
    if reuse_candidates:
        command.append("--reuse-candidates")
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stderr or "") or (exc.stdout or ""))
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        raise HTTPException(
            status_code=504,
            detail=(output[-1500:] + "\n完整建库运行超过1小时，已停止。")[-2000:],
        ) from exc
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(result.stderr or result.stdout or "Library rebuild failed")[-2000:],
        )
    return result.stdout


def _track_image_url(path_text: str) -> str:
    path = resolve_project_path(path_text)
    try:
        relative = path.relative_to(TRACKS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=500, detail=f"Invalid tracked image: {path_text}")
    return "/tracks/" + relative.as_posix()


@app.get("/dedup", response_class=HTMLResponse)
def dedup_page():
    if not DEDUP_TEMPLATE.is_file():
        return HTMLResponse("<h1>Dedup template not found</h1>", status_code=500)
    return HTMLResponse(DEDUP_TEMPLATE.read_text())


def _dedup_tree_items() -> dict[str, dict]:
    from stage4_dedup import load_attributes

    return {item["instance_id"]: item for item in load_attributes()}


def _dedup_tree_object(item: dict, library_name: str | None = None) -> dict:
    best_path = resolve_project_path(item["best_quality_path"])
    lossless_preview = best_path.with_suffix(".png")
    return {
        "instance_id": item["instance_id"],
        "name": library_name,
        "object_id": item["object_id"],
        "session_key": item["session_key"],
        "quality_score": item.get("representative_quality_score", 0.0),
        "attributes": item["attributes"],
        "category_path": item.get("category_path", []),
        "attribute_paths": item.get("attribute_paths", {}),
        "wordnet_warning": item.get("wordnet_warning"),
        "best_url": _track_image_url(
            portable_path(lossless_preview if lossless_preview.is_file() else best_path)
        ),
        "context_url": f"/api/dedup-tree/context/{item['instance_id']}",
    }


def _dedup_tree_payload(layout: dict) -> dict:
    items = _dedup_tree_items()
    instance_names = {}
    try:
        for library_item in load_library().values():
            for source in library_item.get("source_instances", []):
                instance_names[source["instance_id"]] = library_item.get("name")
    except HTTPException:
        pass
    clusters = []
    assigned_ids = set()
    for cluster in layout.get("clusters", []):
        members = [
            _dedup_tree_object(items[instance_id], instance_names.get(instance_id))
            for instance_id in cluster.get("member_ids", [])
            if instance_id in items
        ]
        if not members:
            continue
        assigned_ids.update(item["instance_id"] for item in members)
        clusters.append({**cluster, "members": members, "member_ids": [
            item["instance_id"] for item in members
        ]})
    trash = [
        _dedup_tree_object(items[instance_id], instance_names.get(instance_id))
        for instance_id in layout.get("trash_instance_ids", [])
        if instance_id in items
    ]
    assigned_ids.update(item["instance_id"] for item in trash)
    return {
        "version": layout.get("version", 1),
        "generated_at": layout.get("generated_at"),
        "settings": layout.get("settings", {}),
        "nodes": layout.get("nodes", []),
        "clusters": clusters,
        "trash": trash,
        "object_count": len(items),
        "assigned_count": len(assigned_ids),
        "cluster_count": len(clusters),
        "trash_count": len(trash),
    }


def _ensure_dedup_tree_layout() -> dict:
    from stage4_dedup import items_fingerprint

    items = list(_dedup_tree_items().values())
    current_fingerprint = items_fingerprint(items)
    previous = load_dedup_tree_layout() if DEDUP_TREE_PATH.is_file() else None
    if previous is not None and previous.get("source_fingerprint") == current_fingerprint:
        return previous
    if previous is not None:
        raise HTTPException(
            status_code=409,
            detail="类别树已过期，请点击“补全属性并生成树”重新计算 CLIP 和类别树",
        )
    layout = generate_dedup_tree_layout(items)
    rebuild_library_from_tree(layout)
    save_dedup_tree_layout(layout)
    return layout


@app.get("/api/dedup-tree/context/{instance_id}")
def dedup_tree_context_image(instance_id: str):
    from stage3_attribute import build_highlighted_scene_context, find_instances

    instance = next(
        (item for item in find_instances("sam3") if item["instance_id"] == instance_id),
        None,
    )
    if instance is None:
        raise HTTPException(status_code=404, detail="找不到物体的跟踪记录")
    image = build_highlighted_scene_context(instance, full_frame=True)
    if image is None:
        raise HTTPException(status_code=404, detail="这个物体没有可用的原始场景背景")
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return Response(
        content=output.getvalue(), media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/dedup-tree")
def dedup_tree_data():
    with DEDUP_LOCK:
        return _dedup_tree_payload(_ensure_dedup_tree_layout())


@app.post("/api/dedup-tree/move")
def move_dedup_tree_object(request: DedupTreeMove):
    _assert_no_model_job()
    if bool(request.target_cluster_id) == bool(request.target_node_id):
        raise HTTPException(status_code=400, detail="必须选择目标簇或目标节点之一")
    with DEDUP_LOCK:
        layout = deepcopy(_ensure_dedup_tree_layout())
        known_ids = {
            value for cluster in layout.get("clusters", [])
            for value in cluster.get("member_ids", [])
        } | set(layout.get("trash_instance_ids", []))
        if request.instance_id not in known_ids:
            raise HTTPException(status_code=404, detail="树中没有这个物体")
        try:
            move_dedup_tree_instance(
                layout, request.instance_id, request.target_cluster_id,
                request.target_node_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        rebuild_library_from_tree(layout)
        save_dedup_tree_layout(layout)
        return {"ok": True, "tree": _dedup_tree_payload(layout)}


@app.post("/api/dedup-tree/trash")
def trash_dedup_tree_object(request: DedupTreeTrash):
    _assert_no_model_job()
    with DEDUP_LOCK:
        layout = deepcopy(_ensure_dedup_tree_layout())
        known_ids = {
            value for cluster in layout.get("clusters", [])
            for value in cluster.get("member_ids", [])
        }
        if request.instance_id not in known_ids:
            raise HTTPException(status_code=404, detail="树中没有这个物体")
        trash_dedup_tree_instance(layout, request.instance_id)
        rebuild_library_from_tree(layout)
        save_dedup_tree_layout(layout)
        return {"ok": True, "tree": _dedup_tree_payload(layout)}


@app.post("/api/dedup-tree/regenerate")
def regenerate_dedup_tree(request: DedupTreeRegenerate):
    _assert_no_model_job()
    with DEDUP_LOCK:
        items = list(_dedup_tree_items().values())
        previous = load_dedup_tree_layout() if DEDUP_TREE_PATH.is_file() else None
        if previous is not None:
            previous, _ = backfill_layout_instance_fingerprints(previous, items)
        layout = generate_dedup_tree_layout(
            items,
            request.clip_threshold,
            request.attribute_threshold,
        )
        if previous is not None:
            layout = inherit_unchanged_adjustments(previous, layout)
        rebuild_library_from_tree(layout)
        save_dedup_tree_layout(layout, archive_existing=True)
        return {"ok": True, "tree": _dedup_tree_payload(layout)}


@app.get("/api/dedup")
def dedup_data():
    if not DEDUP_CANDIDATES.is_file() or not (WORK_DIR / "instances.json").is_file():
        raise HTTPException(status_code=404, detail="请先运行 stage4_dedup.py")
    candidate_data = json.loads(DEDUP_CANDIDATES.read_text())
    instances = {
        item["instance_id"]: item
        for item in json.loads((WORK_DIR / "instances.json").read_text())
    }
    pairs = []
    for candidate in candidate_data.get("candidates", []):
        left = instances[candidate["left_instance_id"]]
        right = instances[candidate["right_instance_id"]]
        pairs.append({
            **candidate,
            "left": {
                "instance_id": left["instance_id"],
                "object_id": left["object_id"],
                "session_key": left["session_key"],
                "quality_score": left.get("representative_quality_score", 0.0),
                "attributes": left["attributes"],
                "best_url": _track_image_url(left["best_quality_path"]),
            },
            "right": {
                "instance_id": right["instance_id"],
                "object_id": right["object_id"],
                "session_key": right["session_key"],
                "quality_score": right.get("representative_quality_score", 0.0),
                "attributes": right["attributes"],
                "best_url": _track_image_url(right["best_quality_path"]),
            },
        })
    clusters = _build_dedup_clusters(pairs)
    return {
        "clip_recall_threshold": candidate_data.get("clip_recall_threshold", 0.88),
        "attribute_threshold": candidate_data.get("attribute_threshold", 0.50),
        "pairs": pairs,
        "pending": sum(item["decision"] == "pending" for item in pairs),
        "automatic": sum(item["decision"] == "auto_merge" for item in pairs),
        "clusters": clusters,
        "pending_clusters": sum(item["status"] == "pending" for item in clusters),
    }


def _build_dedup_clusters(pairs: list[dict]) -> list[dict]:
    """Build conservative complete-link clusters from pairwise candidates."""
    eligible = {"pending", "auto_merge", "merge"}
    objects = {}
    active_by_key = {}
    for pair in pairs:
        left_id, right_id = pair["left_instance_id"], pair["right_instance_id"]
        objects[left_id], objects[right_id] = pair["left"], pair["right"]
        if pair["decision"] in eligible:
            active_by_key[frozenset((left_id, right_id))] = pair

    parent = {instance_id: instance_id for instance_id in objects}

    def find(instance_id):
        while parent[instance_id] != instance_id:
            parent[instance_id] = parent[parent[instance_id]]
            instance_id = parent[instance_id]
        return instance_id

    def groups():
        result = {}
        for instance_id in parent:
            result.setdefault(find(instance_id), set()).add(instance_id)
        return result

    def union(left_id, right_id):
        left_root, right_root = find(left_id), find(right_id)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    # Preserve relations that are already accepted by automatic or manual review.
    for pair in pairs:
        if pair["decision"] in {"auto_merge", "merge"}:
            union(pair["left_instance_id"], pair["right_instance_id"])

    # Add pending relations from strongest to weakest, but only when every member
    # across the two groups has an eligible edge. This prevents chain explosions.
    pending_pairs = sorted(
        (pair for pair in pairs if pair["decision"] == "pending"),
        key=lambda pair: (
            float((pair.get("vlm_verification") or {}).get(
                "same_type_confidence", 0.0
            )),
            pair["visual_similarity"],
        ),
        reverse=True,
    )
    for pair in pending_pairs:
        left_id, right_id = pair["left_instance_id"], pair["right_instance_id"]
        current_groups = groups()
        left_group, right_group = current_groups[find(left_id)], current_groups[find(right_id)]
        if left_group == right_group:
            continue
        if all(
            frozenset((left_member, right_member)) in active_by_key
            for left_member in left_group
            for right_member in right_group
        ):
            union(left_id, right_id)

    clusters = []
    for member_set in groups().values():
        active_members = {
            instance_id for instance_id in member_set
            if any(instance_id in key for key in active_by_key)
        }
        if len(active_members) < 2:
            continue
        member_ids = sorted(active_members)
        member_set = set(member_ids)
        edges = [
            pair for pair in pairs
            if pair["left_instance_id"] in member_set
            and pair["right_instance_id"] in member_set
        ]
        active_edges = [pair for pair in edges if pair["decision"] in eligible]
        status = (
            "pending" if any(pair["decision"] == "pending" for pair in active_edges)
            else "manual_merge" if any(pair["decision"] == "merge" for pair in active_edges)
            else "auto_merge"
        )
        clusters.append({
            "cluster_id": uuid.uuid5(
                uuid.NAMESPACE_URL, "robocoin-dedup:" + "|".join(member_ids)
            ).hex[:16],
            "status": status,
            "members": [objects[instance_id] for instance_id in member_ids],
            "member_ids": member_ids,
            "edge_count": len(active_edges),
            "candidate_edge_count": len(edges),
            "pending_edge_count": sum(
                pair["decision"] == "pending" for pair in active_edges
            ),
            "manual_edge_count": sum(
                pair["decision"] in {"merge", "reject"} for pair in edges
            ),
            "auto_edge_count": sum(
                pair["decision"] == "auto_merge" for pair in active_edges
            ),
            "max_visual_similarity": max(
                (pair["visual_similarity"] for pair in active_edges), default=0.0
            ),
            "max_vlm_confidence": max(
                (float((pair.get("vlm_verification") or {}).get(
                    "same_type_confidence", 0.0
                )) for pair in active_edges), default=0.0
            ),
        })
    clusters.sort(key=lambda item: (
        item["status"] != "pending", -len(item["members"]), item["cluster_id"]
    ))
    return clusters


@app.post("/api/dedup/settings")
def update_dedup_settings(request: DedupSettings):
    _assert_no_model_job()
    with DEDUP_LOCK:
        output = _run_library_build(request.dict())
    return {"ok": True, "build_output": output}


@app.post("/api/pipeline/vlm/job")
def start_vlm_pipeline_job(request: VlmRunRequest):
    """Start the full library pipeline without a fixed wall-clock timeout."""
    with JOB_LOCK:
        for job in JOBS.values():
            if _job_is_running(job):
                raise HTTPException(status_code=409, detail="已有模型任务正在运行")
        _unload_resident_sam3()
        job_id = uuid.uuid4().hex
        log_dir = WORK_DIR / ".jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"vlm-library-{job_id}.log"
        command = [
            sys.executable,
            str(BASE_DIR / "run_vlm_library_pipeline.py"),
            "--tracker",
            request.tracker,
        ]
        if request.force:
            command.append("--force")
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        JOBS[job_id] = {
            "process": process,
            "log_path": log_path,
            "kind": "vlm_library_pipeline",
            "session": "all",
            "cancel_requested": False,
        }
    return {"job_id": job_id, "kind": "vlm_library_pipeline"}


@app.post("/api/dedup/decision")
def save_dedup_decision(request: DedupDecision):
    _assert_no_model_job()
    with DEDUP_LOCK:
        data = dedup_data()
        valid_pairs = {item["pair_id"] for item in data["pairs"]}
        if request.pair_id not in valid_pairs:
            raise HTTPException(status_code=404, detail="去重候选不存在")
        try:
            decisions = json.loads(DEDUP_DECISIONS.read_text()) if DEDUP_DECISIONS.is_file() else {}
        except json.JSONDecodeError:
            decisions = {}
        pair = next(item for item in data["pairs"] if item["pair_id"] == request.pair_id)
        valid_instances = {pair["left_instance_id"], pair["right_instance_id"]}
        if request.decision == "merge" and request.canonical_instance_id not in valid_instances:
            raise HTTPException(status_code=400, detail="人工合并时必须选择左图或右图作为主图")
        if request.decision == "pending":
            decisions.pop(request.pair_id, None)
        else:
            decisions[request.pair_id] = {
                "decision": request.decision,
                "canonical_instance_id": (
                    request.canonical_instance_id if request.decision == "merge" else None
                ),
            }
        previous = DEDUP_DECISIONS.read_bytes() if DEDUP_DECISIONS.is_file() else None
        previous_candidates = (
            DEDUP_CANDIDATES.read_bytes() if DEDUP_CANDIDATES.is_file() else None
        )
        _write_json_atomic(DEDUP_DECISIONS, decisions)
        try:
            output = _run_library_build(reuse_candidates=True)
        except Exception:
            _restore_file(DEDUP_DECISIONS, previous)
            _restore_file(DEDUP_CANDIDATES, previous_candidates)
            raise
        return {
            "ok": True,
            "decision": request.decision,
            "build_output": output,
            "dedup": dedup_data(),
        }


@app.post("/api/dedup/cluster-decision")
def save_dedup_cluster_decision(request: DedupClusterDecision):
    _assert_no_model_job()
    with DEDUP_LOCK:
        return _save_dedup_cluster_decision_locked(request)


def _save_dedup_cluster_decision_locked(request: DedupClusterDecision):
    data = dedup_data()
    requested_ids = set(request.instance_ids)
    cluster = next(
        (item for item in data["clusters"] if set(item["member_ids"]) == requested_ids),
        None,
    )
    if cluster is None:
        raise HTTPException(status_code=409, detail="候选簇已经变化，请刷新页面")
    selected_ids = set(request.selected_ids)
    if not selected_ids.issubset(requested_ids):
        raise HTTPException(status_code=400, detail="所选物体不属于当前候选簇")
    if request.decision == "merge_selected" and len(selected_ids) < 2:
        raise HTTPException(status_code=400, detail="至少选择两个物体才能合并")
    if request.decision == "exclude_selected" and not selected_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个要移出的物体")
    if request.decision in {"merge_selected", "exclude_selected"} and selected_ids == requested_ids:
        raise HTTPException(status_code=400, detail="部分操作必须至少保留一个未选物体")
    try:
        decisions = json.loads(DEDUP_DECISIONS.read_text()) if DEDUP_DECISIONS.is_file() else {}
    except json.JSONDecodeError:
        decisions = {}

    objects = {item["instance_id"]: item for item in cluster["members"]}
    canonical_pool = selected_ids if request.decision == "merge_selected" else requested_ids
    canonical_id = max(
        canonical_pool,
        key=lambda instance_id: objects[instance_id].get("quality_score", 0.0),
    )
    for pair in data["pairs"]:
        left_id, right_id = pair["left_instance_id"], pair["right_instance_id"]
        if left_id not in requested_ids or right_id not in requested_ids:
            continue
        left_selected, right_selected = left_id in selected_ids, right_id in selected_ids
        is_cross_selection = left_selected != right_selected
        if request.decision == "reject" or (
            request.decision in {"merge_selected", "exclude_selected"}
            and is_cross_selection
        ):
            decisions[pair["pair_id"]] = {
                "decision": "reject", "canonical_instance_id": None,
            }
        elif (
            request.decision == "merge"
            or request.decision == "merge_selected" and left_selected and right_selected
        ) and pair["decision"] in {"pending", "auto_merge", "merge"}:
            edge_canonical = canonical_id if canonical_id in {left_id, right_id} else max(
                (left_id, right_id),
                key=lambda instance_id: objects[instance_id].get("quality_score", 0.0),
            )
            decisions[pair["pair_id"]] = {
                "decision": "merge", "canonical_instance_id": edge_canonical,
            }

    previous = DEDUP_DECISIONS.read_bytes() if DEDUP_DECISIONS.is_file() else None
    previous_candidates = (
        DEDUP_CANDIDATES.read_bytes() if DEDUP_CANDIDATES.is_file() else None
    )
    _write_json_atomic(DEDUP_DECISIONS, decisions)
    try:
        output = _run_library_build(reuse_candidates=True)
    except Exception:
        _restore_file(DEDUP_DECISIONS, previous)
        _restore_file(DEDUP_CANDIDATES, previous_candidates)
        raise
    return {"ok": True, "build_output": output, "dedup": dedup_data()}


@app.post("/api/new-library/{obj_id}/attributes")
def update_library_attributes(obj_id: str, request: AttributeUpdate):
    _assert_no_model_job()
    with DEDUP_LOCK:
        library = load_library()
        item = library.get(obj_id)
        if item is None:
            raise HTTPException(status_code=404, detail="新物体库中没有这个对象")
        allowed = {"category", "color", "material", "shape", "texture"}
        attributes = {
            key: value for key, value in request.attributes.items() if key in allowed
        }
        if not attributes:
            raise HTTPException(status_code=400, detail="没有可保存的属性")
        try:
            overrides = json.loads(ATTRIBUTE_OVERRIDES.read_text()) if ATTRIBUTE_OVERRIDES.is_file() else {}
        except json.JSONDecodeError:
            overrides = {}
        affected_ids = {
            source["instance_id"] for source in item.get("source_instances", [])
        }
        for source in item.get("source_instances", []):
            instance_id = source["instance_id"]
            overrides[instance_id] = {**overrides.get(instance_id, {}), **attributes}
        previous_overrides = (
            ATTRIBUTE_OVERRIDES.read_bytes() if ATTRIBUTE_OVERRIDES.is_file() else None
        )
        _write_json_atomic(ATTRIBUTE_OVERRIDES, overrides)
        try:
            current_items = list(_dedup_tree_items().values())
            previous_layout = (
                load_dedup_tree_layout() if DEDUP_TREE_PATH.is_file() else None
            )
            if "category" in attributes:
                settings = (previous_layout or {}).get("settings", {})
                layout = generate_dedup_tree_layout(
                    current_items,
                    float(settings.get("clip_threshold", 0.82)),
                    float(settings.get("attribute_threshold", 0.50)),
                )
                if previous_layout is not None:
                    previous_layout, _ = backfill_layout_instance_fingerprints(
                        previous_layout, current_items
                    )
                    old_fingerprints = dict(
                        previous_layout.get("instance_fingerprints", {})
                    )
                    for instance_id in affected_ids:
                        old_fingerprints.pop(instance_id, None)
                    previous_layout["instance_fingerprints"] = old_fingerprints
                    layout = inherit_unchanged_adjustments(previous_layout, layout)
                rebuilt = rebuild_library_from_tree(layout)
                save_dedup_tree_layout(
                    layout, archive_existing=previous_layout is not None
                )
            else:
                from stage4_dedup import items_fingerprint

                if previous_layout is None:
                    layout = generate_dedup_tree_layout(current_items)
                else:
                    layout = deepcopy(previous_layout)
                    layout["source_fingerprint"] = items_fingerprint(current_items)
                rebuilt = rebuild_library_from_tree(layout)
                save_dedup_tree_layout(
                    layout, archive_existing=previous_layout is not None
                )
        except Exception:
            _restore_file(ATTRIBUTE_OVERRIDES, previous_overrides)
            raise
        return {
            "ok": True,
            "build_output": f"Rebuilt {len(rebuilt)} objects from editable tree",
        }


@app.get("/review", response_class=HTMLResponse)
def review_page():
    if not REVIEW_TEMPLATE.is_file():
        return HTMLResponse("<h1>Review template not found</h1>", status_code=500)
    return HTMLResponse(REVIEW_TEMPLATE.read_text())


@app.get("/object-links", response_class=HTMLResponse)
def object_links_page():
    if not OBJECT_LINK_TEMPLATE.is_file():
        return HTMLResponse("<h1>Object-link template not found</h1>", status_code=500)
    return HTMLResponse(OBJECT_LINK_TEMPLATE.read_text())


def _object_link_manifest() -> dict:
    if not OBJECT_LINK_MANIFEST.is_file():
        raise HTTPException(
            status_code=404,
            detail="尚未生成物体名词映射，请先运行 python object_text_linker.py --generate",
        )
    try:
        return json.loads(OBJECT_LINK_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"物体名词映射损坏：{exc}") from exc


@lru_cache(maxsize=4096)
def _object_link_episode_videos(dataset: str, episode_index: int) -> tuple[Path, ...]:
    """Return every camera video for an episode, contained inside the source root."""
    dataset_dir = (OBJECT_LINK_DATASETS / dataset).resolve()
    try:
        dataset_dir.relative_to(OBJECT_LINK_DATASETS.resolve())
    except ValueError:
        return ()
    if not dataset_dir.is_dir():
        return ()
    name = f"episode_{episode_index:06d}.mp4"
    return tuple(sorted(path.resolve() for path in dataset_dir.glob(f"videos/*/*/{name}")))


@lru_cache(maxsize=2048)
def _object_link_task_episode(dataset: str, source_text: str) -> int | None:
    """Find a representative episode containing a dataset-level task string."""
    episodes = OBJECT_LINK_DATASETS / dataset / "meta" / "episodes.jsonl"
    if not episodes.is_file():
        return None
    wanted = source_text.strip().casefold().rstrip(".")
    try:
        with episodes.open() as handle:
            for fallback_index, line in enumerate(handle):
                record = json.loads(line)
                tasks = record.get("tasks", [])
                if any(str(task).strip().casefold().rstrip(".") == wanted for task in tasks):
                    return int(record.get("episode_index", fallback_index))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def _object_link_media(mention: dict) -> dict:
    source_path = str(mention.get("source_path", ""))
    line_index = int(mention.get("line_index", 0))
    dataset = str(mention.get("dataset", ""))
    is_task = source_path.endswith("/meta/tasks.jsonl")
    episode_index = (
        _object_link_task_episode(dataset, str(mention.get("source_text", "")))
        if is_task else line_index
    )
    relation = "representative" if is_task else "episode"
    if episode_index is None:
        episode_index = 0
        relation = "fallback"
    videos = _object_link_episode_videos(dataset, episode_index)
    streams = []
    for camera_index, path in enumerate(videos):
        query = (
            f"dataset={quote(dataset)}&episode={episode_index}"
            f"&camera={camera_index}"
        )
        streams.append({
            "camera": path.parent.name,
            "video_url": f"/api/object-links/media/video?{query}",
            "preview_url": f"/api/object-links/media/frame?{query}",
        })
    return {"episode_index": episode_index, "relation": relation, "streams": streams}


@app.get("/api/object-links")
def object_link_items(
    status: str = Query("pending"),
    q: str = Query(""),
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):
    manifest = _object_link_manifest()
    library = load_library()
    mentions = manifest.get("mentions", [])
    stats = {}
    for mention in mentions:
        key = str(mention.get("status", "unresolved"))
        stats[key] = stats.get(key, 0) + 1
    query = q.strip().lower()
    filtered = []
    for mention in mentions:
        if status != "all" and mention.get("status") != status:
            continue
        haystack = " ".join(map(str, (
            mention.get("dataset", ""), mention.get("source_path", ""),
            mention.get("source_text", ""), mention.get("text", ""),
            " ".join(mention.get("candidate_ids", [])),
            " ".join(mention.get("selected_ids", [])),
        ))).lower()
        if query and query not in haystack:
            continue
        filtered.append(mention)
    page = []
    for mention in filtered[offset:offset + limit]:
        candidate_ids = list(dict.fromkeys([
            *mention.get("candidate_ids", []), *mention.get("selected_ids", [])
        ]))
        candidates = []
        for object_id in candidate_ids[:20]:
            item = library.get(object_id)
            if item is None:
                continue
            attributes = item.get("attributes", {})
            canonical = str(item.get("canonical_path", ""))
            candidates.append({
                "id": object_id,
                "category": attributes.get("category", "unknown"),
                "color": attributes.get("color", "unknown"),
                "material": attributes.get("material", "unknown"),
                "image_url": "/library/" + canonical.replace(
                    "objects/new_library/", "", 1
                ),
            })
        page.append({
            **mention,
            "candidates": candidates,
            "candidate_count": len(candidate_ids),
            "media": _object_link_media(mention),
        })
    return {
        "items": page,
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "stats": stats,
    }


@app.get("/api/object-links/library/search")
def object_link_library_search(
    q: str = Query("", min_length=1),
    limit: int = Query(30, ge=1, le=100),
):
    library = load_library()
    query = q.strip().lower()
    results = []
    for object_id, item in library.items():
        attributes = item.get("attributes", {})
        haystack = " ".join([
            object_id,
            str(attributes.get("category", "")),
            str(attributes.get("color", "")),
            str(attributes.get("material", "")),
            str(attributes.get("shape", "")),
        ]).lower()
        if query not in haystack:
            continue
        canonical = str(item.get("canonical_path", ""))
        results.append({
            "id": object_id,
            "category": attributes.get("category", "unknown"),
            "color": attributes.get("color", "unknown"),
            "material": attributes.get("material", "unknown"),
            "image_url": "/library/" + canonical.replace(
                "objects/new_library/", "", 1
            ),
        })
        if len(results) >= limit:
            break
    return {"items": results}


def _resolve_object_link_video(dataset: str, episode: int, camera: int) -> Path:
    videos = _object_link_episode_videos(dataset, episode)
    if camera < 0 or camera >= len(videos):
        raise HTTPException(status_code=404, detail="没有找到对应 episode 的相机视频")
    return videos[camera]


@app.get("/api/object-links/media/video")
def object_link_video(
    dataset: str = Query(...), episode: int = Query(..., ge=0),
    camera: int = Query(0, ge=0),
):
    return FileResponse(
        _resolve_object_link_video(dataset, episode, camera), media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/object-links/media/frame")
def object_link_frame(
    dataset: str = Query(...), episode: int = Query(..., ge=0),
    camera: int = Query(0, ge=0),
):
    video = _resolve_object_link_video(dataset, episode, camera)
    preview = OBJECT_LINK_PREVIEWS / dataset / f"episode_{episode:06d}_{camera}.jpg"
    if not preview.is_file() or preview.stat().st_mtime_ns < video.stat().st_mtime_ns:
        preview.parent.mkdir(parents=True, exist_ok=True)
        temp = preview.with_name(preview.name + ".tmp.jpg")
        command = [
            "ffmpeg", "-loglevel", "error", "-y", "-ss", "0.5", "-i", str(video),
            "-frames:v", "1", "-vf", "scale='min(960,iw)':-2", str(temp),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            temp.replace(preview)
        except (OSError, subprocess.SubprocessError) as exc:
            temp.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"无法提取视频画面：{exc}") from exc
    return FileResponse(
        preview, media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/object-links/{mention_id}")
def update_object_link(mention_id: str, request: ObjectLinkDecision):
    _assert_no_model_job()
    from object_text_linker import save_decision

    with EDIT_LOCK:
        try:
            mention = save_decision(
                mention_id, request.selected_ids, ignored=request.ignored
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="没有这个名词标注") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f"保存失败：{exc}") from exc
    return {"ok": True, "mention": mention}


@app.get("/api/review/sessions")
def review_sessions():
    return {"sessions": _review_sessions()}


@app.get("/api/review/session")
def review_session(session: str = Query(...), frame_id: str | None = Query(default=None)):
    directory = _resolve_review_session(session)
    return _review_payload(session, directory, frame_id)


@app.get("/api/review/frame-preview")
def review_frame_preview(
    session: str = Query(...), timestamp_seconds: float = Query(..., ge=0)
):
    directory = _resolve_review_session(session)
    video_path, _ = _session_video_and_fps(directory)
    duration = _video_duration_seconds(video_path)
    timestamp_seconds = min(timestamp_seconds, max(0.0, duration - 0.001))
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", f"{timestamp_seconds:.6f}",
        "-i", str(video_path), "-frames:v", "1",
        "-vf", "scale='min(1280,iw)':-2", "-q:v", "3",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ], capture_output=True)
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode(errors="replace").strip()
        raise HTTPException(status_code=500, detail=f"关键帧预览失败：{detail}")
    return Response(
        content=result.stdout,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/review/action")
def review_action(request: ReviewAction):
    directory = _resolve_review_session(request.session)
    with EDIT_LOCK:
        if request.action == "undo":
            _restore_latest_snapshot(directory)
            return _review_payload(request.session, directory)

        manifest, state = _load_review_state(directory)
        frames = _manifest_review_frames(manifest)
        active_frame_id = str(
            request.frame_id
            or manifest.get("active_discovery_frame_id")
            or (frames[0]["frame_id"] if frames else "")
        )
        frames_by_id = {item["frame_id"]: item for item in frames}
        if active_frame_id not in frames_by_id:
            raise HTTPException(status_code=404, detail="当前关键帧不存在")
        by_id = {int(item["object_id"]): index for index, (item, _) in enumerate(state)}
        requested_ids = list(dict.fromkeys(request.object_ids))

        if request.action == "delete_keyframe":
            if len(frames) <= 1:
                raise HTTPException(status_code=409, detail="至少需要保留一张关键帧")
            deleted_index = next(
                index for index, frame in enumerate(frames)
                if frame["frame_id"] == active_frame_id
            )
            remaining_frames = [
                frame for frame in frames if frame["frame_id"] != active_frame_id
            ]
            next_frame = remaining_frames[
                min(deleted_index, len(remaining_frames) - 1)
            ]
            new_state = [
                pair for pair in state
                if _item_frame_id(pair[0], manifest) != active_frame_id
            ]
            manifest["discovery_frames"] = remaining_frames
            manifest["active_discovery_frame_id"] = next_frame["frame_id"]
            manifest["frame"] = next_frame["frame"]
            manifest["discovery_frame"] = {
                key: value for key, value in next_frame.items()
                if key not in {"frame_id", "frame", "object_count"}
            }

        elif request.action == "delete":
            if not requested_ids:
                raise HTTPException(status_code=400, detail="请先选择要删除的物体")
            missing = [value for value in requested_ids if value not in by_id]
            if missing:
                raise HTTPException(status_code=404, detail=f"Object IDs not found: {missing}")
            selected = set(requested_ids)
            new_state = [pair for pair in state if int(pair[0]["object_id"]) not in selected]

        elif request.action == "merge":
            if len(requested_ids) < 2:
                raise HTTPException(status_code=400, detail="合并至少需要选择两个物体")
            missing = [value for value in requested_ids if value not in by_id]
            if missing:
                raise HTTPException(status_code=404, detail=f"Object IDs not found: {missing}")
            selected = set(requested_ids)
            selected_pairs = [
                pair for pair in state if int(pair[0]["object_id"]) in selected
            ]
            selected_frame_ids = {
                _item_frame_id(item, manifest) for item, _ in selected_pairs
            }
            if len(selected_frame_ids) != 1:
                raise HTTPException(status_code=400, detail="只能合并同一关键帧的物体")
            merged_mask = np.logical_or.reduce([mask for _, mask in selected_pairs])
            merged_item = dict(selected_pairs[0][0])
            merged_item.update({
                "score": max(float(item.get("score", 1.0)) for item, _ in selected_pairs),
                "prompt": "manual_merge",
                "source": "manual_merge",
                "merged_from": requested_ids,
            })
            insert_at = min(by_id[value] for value in requested_ids)
            new_state = []
            for index, pair in enumerate(state):
                if index == insert_at:
                    new_state.append((merged_item, merged_mask))
                if int(pair[0]["object_id"]) not in selected:
                    new_state.append(pair)

        elif request.action in {"add_point", "add_negative_point"}:
            if len(requested_ids) != 1 or request.point is None or len(request.point) != 2:
                raise HTTPException(status_code=400, detail="请选择一个人工框并点击提示点")
            object_id = requested_ids[0]
            if object_id not in by_id:
                raise HTTPException(status_code=404, detail="人工框不存在")
            item, mask = state[by_id[object_id]]
            if item.get("source") not in {"manual_box", "sam3_box_refined"}:
                raise HTTPException(status_code=400, detail="关键点只能添加到人工框物体")
            if _item_frame_id(item, manifest) != active_frame_id:
                raise HTTPException(status_code=400, detail="人工框不属于当前关键帧")
            x, y = [float(value) for value in request.point]
            if not np.isfinite([x, y]).all():
                raise HTTPException(status_code=400, detail="关键点坐标无效")
            prompt_box = item.get("prompt_box", item.get("bbox", _mask_bbox(mask)))
            x1, y1, x2, y2 = [float(value) for value in prompt_box]
            if not (x1 <= x < x2 and y1 <= y < y2):
                raise HTTPException(status_code=400, detail="关键点必须位于人工框内部")
            point_field = (
                "prompt_points"
                if request.action == "add_point"
                else "prompt_negative_points"
            )
            points = [
                [float(point[0]), float(point[1])]
                for point in item.get(point_field, [])
                if isinstance(point, (list, tuple)) and len(point) == 2
            ]
            if not any((px - x) ** 2 + (py - y) ** 2 < 4 for px, py in points):
                points.append([x, y])
            new_state = []
            for candidate, candidate_mask in state:
                updated = dict(candidate)
                if int(candidate["object_id"]) == object_id:
                    updated[point_field] = points
                new_state.append((updated, candidate_mask))

        elif request.action == "clear_points":
            if len(requested_ids) != 1:
                raise HTTPException(status_code=400, detail="请选择一个人工框")
            object_id = requested_ids[0]
            if object_id not in by_id:
                raise HTTPException(status_code=404, detail="人工框不存在")
            new_state = []
            for candidate, candidate_mask in state:
                updated = dict(candidate)
                if int(candidate["object_id"]) == object_id:
                    updated.pop("prompt_points", None)
                    updated.pop("prompt_negative_points", None)
                new_state.append((updated, candidate_mask))

        else:
            if request.box is None or len(request.box) != 4:
                raise HTTPException(status_code=400, detail="新增物体需要四个框坐标")
            try:
                frame_path = resolve_project_path(frames_by_id[active_frame_id]["frame"])
                with Image.open(frame_path) as image:
                    width, height = image.size
            except (KeyError, OSError) as exc:
                raise HTTPException(status_code=500, detail=f"Cannot read discovery frame: {exc}")
            x1, y1, x2, y2 = [float(value) for value in request.box]
            x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
            y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
            ix1, iy1, ix2, iy2 = map(round, (x1, y1, x2, y2))
            if ix2 - ix1 < 3 or iy2 - iy1 < 3:
                raise HTTPException(status_code=400, detail="框选区域太小")
            mask = np.zeros((height, width), dtype=bool)
            mask[iy1:iy2, ix1:ix2] = True
            new_state = state + [({
                "score": 1.0,
                "prompt": "manual_box",
                "source": "manual_box",
                "prompt_box": [ix1, iy1, ix2, iy2],
                "prompt_points": [],
                "prompt_negative_points": [],
                "discovery_frame_id": active_frame_id,
                "source_frame_index": int(
                    frames_by_id[active_frame_id]["source_frame_index"]
                ),
                "source_frame": frames_by_id[active_frame_id]["frame"],
            }, mask)]

        _snapshot_review_state(directory)
        _write_review_state(directory, manifest, new_state)
        response_frame_id = (
            manifest["active_discovery_frame_id"]
            if request.action == "delete_keyframe"
            else active_frame_id
        )
        return _review_payload(request.session, directory, response_frame_id)


@app.post("/api/review/job")
def start_review_job(request: ReviewJobRequest):
    directory = _resolve_review_session(request.session)
    manifest, state = _load_review_state(directory)
    if request.kind == "refine" and not any(
        item.get("source") in {"manual_box", "sam3_box_refined"}
        and (
            request.frame_id is None
            or _item_frame_id(item, manifest) == request.frame_id
        )
        for item, _ in state
    ):
        raise HTTPException(status_code=400, detail="没有需要 SAM3 细化的人工框")
    if request.kind == "keyframe" and request.timestamp_seconds is None:
        raise HTTPException(status_code=400, detail="请选择关键帧时间")
    quality_weights = {
        "sharpness": request.quality_sharpness,
        "area": request.quality_area,
        "confidence": request.quality_confidence,
        "stability": request.quality_stability,
        "boundary": request.quality_boundary,
    }
    if request.kind == "track" and sum(quality_weights.values()) <= 0:
        raise HTTPException(status_code=400, detail="质量权重不能全部为 0")
    video_path, _ = _session_video_and_fps(directory)
    sample_fps = 1.0

    with JOB_LOCK:
        for job in JOBS.values():
            if _job_is_running(job):
                raise HTTPException(status_code=409, detail="已有模型任务正在运行")
        job_id = uuid.uuid4().hex
        log_dir = directory / ".jobs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{request.kind}-{job_id}.log"

        if request.kind in {"refine", "keyframe"}:
            job = {
                "process": None,
                "thread": None,
                "return_code": None,
                "cancel_event": threading.Event(),
                "log_path": log_path,
                "kind": request.kind,
                "session": request.session,
                "cancel_requested": False,
            }
            if request.kind == "refine":
                target = _run_resident_refine_job
                thread_args = (
                    job_id,
                    video_path,
                    str(manifest.get("prompt") or "object"),
                    request.frame_id,
                )
            else:
                target = _run_resident_keyframe_job
                thread_args = (
                    job_id, directory, video_path, float(request.timestamp_seconds)
                )
            thread = threading.Thread(
                target=target,
                args=thread_args,
                name=f"sam3-{request.kind}-{job_id[:8]}",
                daemon=True,
            )
            job["thread"] = thread
            JOBS[job_id] = job
            thread.start()
            return {"job_id": job_id, "kind": request.kind}

        _unload_resident_sam3()
        command = [
            sys.executable,
            str(BASE_DIR / "stage1_track_select.py"),
            "--video", str(video_path),
            "--sample-fps", str(sample_fps),
            "--stage", "track",
        ]
        command.extend([
            "--quality-sharpness", str(quality_weights["sharpness"]),
            "--quality-area", str(quality_weights["area"]),
            "--quality-confidence", str(quality_weights["confidence"]),
            "--quality-stability", str(quality_weights["stability"]),
            "--quality-boundary", str(quality_weights["boundary"]),
            "--quality-min-relative-area", str(request.quality_min_relative_area),
            "--quality-min-stability", str(request.quality_min_stability),
            "--quality-min-sharpness-quantile", str(request.quality_min_sharpness_quantile),
            "--quality-exclude-boundary" if request.quality_exclude_boundary
            else "--no-quality-exclude-boundary",
        ])
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        JOBS[job_id] = {
            "process": process,
            "log_path": log_path,
            "kind": request.kind,
            "session": request.session,
            "cancel_requested": False,
        }
    return {"job_id": job_id, "kind": request.kind}


@app.get("/api/review/job/{job_id}")
def review_job_status(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或服务已重启")
        return_code = _job_return_code(job)
        try:
            log_text = job["log_path"].read_text(errors="replace")[-8000:]
        except OSError:
            log_text = ""
        progress = _parse_progress(log_text)
    return {
        "job_id": job_id,
        "kind": job["kind"],
        "running": return_code is None,
        "return_code": return_code,
        "cancelled": bool(job.get("cancel_requested", False)),
        "log": log_text,
        "progress": progress,
    }


@app.post("/api/review/job/{job_id}/cancel")
def cancel_review_job(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或服务已重启")
        if not _job_is_running(job):
            return {
                "job_id": job_id,
                "cancelled": bool(job.get("cancel_requested", False)),
                "already_finished": True,
            }
        _stop_job_process(job)
        return {
            "job_id": job_id,
            "cancelled": True,
            "already_finished": False,
        }


@app.post("/api/review/job/all")
def start_all_tracking_job(request: ReviewBatchTrackRequest):
    """Track only review manifests changed since their last successful track."""
    quality_weights = {
        "sharpness": request.quality_sharpness,
        "area": request.quality_area,
        "confidence": request.quality_confidence,
        "stability": request.quality_stability,
        "boundary": request.quality_boundary,
    }
    if sum(quality_weights.values()) <= 0:
        raise HTTPException(status_code=400, detail="质量权重不能全部为 0")
    dirty_count = sum(item["tracker_dirty"] for item in _review_sessions())
    if not dirty_count:
        raise HTTPException(status_code=409, detail="没有需要重新跟踪的数据")
    with JOB_LOCK:
        for job in JOBS.values():
            if _job_is_running(job):
                raise HTTPException(status_code=409, detail="已有模型任务正在运行")
        _unload_resident_sam3()
        job_id = uuid.uuid4().hex
        log_dir = TRACKS_DIR / ".jobs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"track-all-{job_id}.log"
        command = [
            sys.executable,
            str(BASE_DIR / "stage1_track_select.py"),
            "--stage", "track",
            "--sample-fps", "1.0",
            "--continue-on-error",
            "--dirty-only",
            "--quality-sharpness", str(quality_weights["sharpness"]),
            "--quality-area", str(quality_weights["area"]),
            "--quality-confidence", str(quality_weights["confidence"]),
            "--quality-stability", str(quality_weights["stability"]),
            "--quality-boundary", str(quality_weights["boundary"]),
            "--quality-min-relative-area", str(request.quality_min_relative_area),
            "--quality-min-stability", str(request.quality_min_stability),
            "--quality-min-sharpness-quantile",
            str(request.quality_min_sharpness_quantile),
            "--quality-exclude-boundary" if request.quality_exclude_boundary
            else "--no-quality-exclude-boundary",
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        JOBS[job_id] = {
            "process": process,
            "log_path": log_path,
            "kind": "track_all",
            "session": "all",
            "cancel_requested": False,
        }
    return {"job_id": job_id, "kind": "track_all", "dirty_count": dirty_count}


@app.get("/", include_in_schema=False)
def root():
    """The primary UI builds a new library; never land on bundled sample data."""
    return RedirectResponse(url="/review", status_code=307)


@app.get("/new-library", response_class=HTMLResponse)
def index(
    category: str = Query("", description="Filter by category"),
    color: str = Query("", description="Filter by color"),
    material: str = Query("", description="Filter by material"),
    sort: str = Query("count", description="Sort: count | category | color"),
):
    lib = load_library()
    objects = list(lib.values())

    # Extract for filters
    all_categories = sorted(set(
        o["attributes"].get("category", "?") for o in objects
    ))
    all_colors = sorted(set(
        o["attributes"].get("color", "?") for o in objects
    ))
    all_materials = sorted(set(
        o["attributes"].get("material", "?") for o in objects
    ))

    # Apply filters
    if category:
        objects = [o for o in objects if o["attributes"].get("category") == category]
    if color:
        objects = [o for o in objects if o["attributes"].get("color") == color]
    if material:
        objects = [o for o in objects if o["attributes"].get("material") == material]

    # Sort
    if sort == "count":
        objects.sort(key=lambda o: o["instance_count"], reverse=True)
    elif sort == "category":
        objects.sort(key=lambda o: o["attributes"].get("category", ""))
    elif sort == "color":
        objects.sort(key=lambda o: o["attributes"].get("color", ""))

    filters_html = _build_filters(all_categories, all_colors, all_materials,
                                  category, color, material, sort)

    cards = ""
    for obj in objects:
        canonical = obj["canonical_path"]
        canonical_url = "/library/" + canonical.replace("objects/new_library/", "", 1)
        attr = obj["attributes"]
        cards += f"""
        <a href="/new-object/{obj['id']}" class="card">
            <img src="{canonical_url}" loading="lazy">
            <div class="card-info">
                <div class="name">{html_lib.escape(str(obj.get('name', attr.get('category', '?'))))}</div>
                <div class="sub">{attr.get('color', '?')} · {attr.get('material', '?')}</div>
                <div class="count">{obj['instance_count']} instances</div>
            </div>
        </a>"""

    return HTML_TEMPLATE.format(
        title=f"New Object Library ({len(objects)} objects)",
        filters=filters_html,
        content=cards,
        count=len(objects),
    )


@app.get("/new-object/{obj_id}", response_class=HTMLResponse)
def object_detail(obj_id: str):
    lib = load_library()
    obj = lib.get(obj_id)
    if not obj:
        return HTMLResponse(f"<h1>Object {obj_id} not found</h1>", status_code=404)

    attr = obj["attributes"]
    canonical = obj["canonical_path"]
    canonical_url = "/library/" + canonical.replace("objects/new_library/", "", 1)

    instances_html = ""
    copied_by_id = {
        Path(path).name.split("_best.jpg", 1)[0]: path
        for path in obj.get("instances", [])
    }
    for source in obj.get("source_instances", []):
        instance_id = source["instance_id"]
        inst = copied_by_id.get(instance_id)
        if not inst:
            continue
        inst_path = inst.replace("objects/new_library/", "", 1)
        session = html_lib.escape(str(source.get("session_key", "unknown")))
        object_id = html_lib.escape(str(source.get("object_id", "?")))
        quality = float(source.get("representative_quality_score", 0.0))
        primary = " · 主图" if instance_id == obj.get("canonical_instance_id") else ""
        instances_html += (
            f'<div class="instance-card"><img src="/library/{inst_path}" loading="lazy">'
            f'<div><b>物体 #{object_id}{primary}</b><br>质量 {quality:.3f}<br>'
            f'<span>{session}</span></div></div>'
        )

    def value(key):
        return html_lib.escape(str(attr.get(key, "unknown")), quote=True)

    html = f"""
    <div class="detail">
        <a href="/new-library" class="back">← 返回新物体库</a>
        <div class="detail-main">
            <img class="canonical" src="{canonical_url}">
            <div class="detail-attrs" id="attributeForm">
                <h2>{html_lib.escape(str(obj.get('name', obj['id'])))}</h2>
                <p class="muted">内部 ID：{obj['id']}</p>
                <table>
                    <tr><td>Category</td><td><input data-key="category" value="{value('category')}"></td></tr>
                    <tr><td>Color</td><td><input data-key="color" value="{value('color')}"></td></tr>
                    <tr><td>Material</td><td><input data-key="material" value="{value('material')}"></td></tr>
                    <tr><td>Shape</td><td><input data-key="shape" value="{value('shape')}"></td></tr>
                    <tr><td>Texture</td><td><input data-key="texture" value="{value('texture')}"></td></tr>
                    <tr><td>Instances</td><td><b>{obj['instance_count']}</b></td></tr>
                </table>
                <button onclick="saveAttributes()">保存人工修改</button>
                <span id="saveMessage"></span>
                <div id="attributeProgress" class="attribute-progress" hidden>
                    <div id="attributeProgressLabel">准备保存…</div>
                    <progress id="attributeProgressBar" max="100" value="0"></progress>
                    <span id="attributeProgressPercent">0%</span>
                </div>
            </div>
        </div>
        <h3>All Instances</h3>
        <div class="instances">{instances_html}</div>
    </div>
    <script>
    function setAttributeProgress(value,label) {{
      const box=document.getElementById('attributeProgress');
      const percent=Math.max(0,Math.min(100,Number(value)));
      box.hidden=false;
      document.getElementById('attributeProgressBar').value=percent;
      document.getElementById('attributeProgressPercent').textContent=percent.toFixed(0)+'%';
      document.getElementById('attributeProgressLabel').textContent=label;
    }}
    async function saveAttributes() {{
      const attributes={{}};
      document.querySelectorAll('#attributeForm [data-key]').forEach(input=>attributes[input.dataset.key]=input.value);
      const message=document.getElementById('saveMessage'),button=document.querySelector('#attributeForm button');
      button.disabled=true; message.textContent=' 保存中…'; setAttributeProgress(5,'正在提交属性修改…');
      try {{
        const response=await fetch('/api/new-library/{obj_id}/attributes',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{attributes}})}});
        setAttributeProgress(80,'物体库已重建，正在读取结果…');
        const data=await response.json();
        if(!response.ok)throw new Error(data.detail||response.status);
        setAttributeProgress(100,'属性和物体库已更新'); message.textContent=' 已保存并重建物体库';
      }} catch(error) {{
        setAttributeProgress(100,'保存失败：'+error.message); message.textContent=' 保存失败：'+error.message;
      }} finally {{ button.disabled=false; }}
    }}
    </script>"""
    return HTML_TEMPLATE.format(title=f"{obj_id} - {attr.get('category', '?')}",
                                filters="", content=html, count=0)


def _build_filters(cats, colors, materials, sel_cat, sel_color, sel_mat, sort):
    def select_opts(options, selected, name):
        html = f'<select name="{name}" onchange="this.form.submit()">'
        html += f'<option value="">All {name}s</option>'
        for o in options:
            sel = "selected" if o == selected else ""
            html += f'<option value="{o}" {sel}>{o}</option>'
        html += '</select>'
        return html

    return f"""
    <form class="filters">
        {select_opts(cats, sel_cat, 'category')}
        {select_opts(colors, sel_color, 'color')}
        {select_opts(materials, sel_mat, 'material')}
        <select name="sort" onchange="this.form.submit()">
            <option value="count" {"selected" if sort=="count" else ""}>Sort: count ↓</option>
            <option value="category" {"selected" if sort=="category" else ""}>Sort: category</option>
            <option value="color" {"selected" if sort=="color" else ""}>Sort: color</option>
        </select>
    </form>"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ box-sizing:border-box; }}
:root {{ color-scheme:dark; --bg:#090d14; --panel:#111824; --panel2:#172131; --line:#263247; --text:#eef5ff; --muted:#91a0b5; --blue:#5aa9ff; }}
body {{ margin:0; font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:radial-gradient(circle at 20% 0,#13233a 0,transparent 32%),var(--bg); color:var(--text); padding:28px; }}
h1 {{ margin:0 0 8px; font-size:1.65em; letter-spacing:-.02em; }}
.filters {{ display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; }}
.filters select {{ padding:9px 12px; border-radius:8px; border:1px solid var(--line);
                    background:var(--panel2); color:var(--text); cursor:pointer; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }}
.card {{ display:block; background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden;
        text-decoration:none; color:#c8d3e1; transition:transform .18s,border-color .18s,box-shadow .18s; }}
.card:hover {{ transform:translateY(-4px); border-color:#476486; box-shadow:0 18px 35px #0005; }}
.card img {{ width:100%; aspect-ratio:1; object-fit:cover; }}
.card-info {{ padding:12px; }}
.card-info .name {{ color:#fff; font-weight:600; text-transform:capitalize; }}
.card-info .sub {{ font-size:.8em; color:var(--muted); margin-top:3px; }}
.card-info .count {{ font-size:.75em; color:#68788e; margin-top:6px; }}
.detail {{ max-width:980px; margin:0 auto; }}
.back {{ color:var(--blue); text-decoration:none; display:inline-block; margin-bottom:18px; }}
.detail-main {{ display:flex; gap:28px; margin-bottom:34px; flex-wrap:wrap; background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; }}
.canonical {{ width:min(360px,100%); border-radius:12px; object-fit:contain; background:#05070a; }}
.detail-attrs table {{ border-collapse:collapse; }}
.detail-attrs td {{ padding:6px 16px 6px 0; color:var(--muted); }}
.detail-attrs td b {{ color:#eee; }}
.detail-attrs input {{ width:240px; max-width:48vw; padding:8px 10px; color:var(--text); background:var(--panel2); border:1px solid var(--line); border-radius:7px; }}
.detail-attrs button {{ margin-top:10px; padding:9px 14px; color:#fff; background:#176fc1; border:1px solid #318de0; border-radius:8px; cursor:pointer; }}
.detail-attrs button:disabled {{ opacity:.55; cursor:not-allowed; }}
.attribute-progress {{ width:300px; max-width:70vw; margin-top:12px; color:var(--muted); font-size:.78em; }}
.attribute-progress progress {{ width:250px; max-width:58vw; height:12px; margin:6px 7px 0 0; accent-color:#54d69a; }}
.instances {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(155px,1fr)); gap:10px; }}
.instances img {{ width:100%; aspect-ratio:1; object-fit:cover; border-radius:8px; }}
.instance-card {{ min-width:0; background:var(--panel); border:1px solid var(--line); padding:8px; border-radius:10px;
                  font-size:.72em; line-height:1.4; color:#aaa; }}
.instance-card b {{ color:#eee; }}
.instance-card span {{ display:block; overflow-wrap:anywhere; margin-top:3px; }}
.summary {{ color:var(--muted); margin-bottom:18px; font-size:.9em; }}
.top-links {{ position:fixed; right:22px; top:20px; z-index:5; display:flex; gap:8px; }}
.top-links a {{ color:#dceeff; background:#17263a; border:1px solid #304866; border-radius:8px; padding:8px 11px; text-decoration:none; font-size:.82em; }}
@media(max-width:650px) {{ body {{ padding:18px; }} .top-links {{ position:static; margin-bottom:18px; }} .detail-attrs input {{ max-width:60vw; }} }}
</style>
</head>
<body>
<nav class="top-links"><a href="/review">关键帧审核</a><a href="/dedup">去重审核</a><a href="/object-links">名词 ID 审核</a></nav>
<h1>{title}</h1>
<div class="summary">{count} objects</div>
{filters}
<div class="grid">{content}</div>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    import socket
    import urllib.error
    import urllib.request
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="if the requested port is occupied, use the next available port",
    )
    args = parser.parse_args()

    def port_is_open(host: str, port: int) -> bool:
        probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        try:
            with socket.create_connection((probe_host, port), timeout=0.4):
                return True
        except OSError:
            return False

    def existing_viewer(host: str, port: int) -> bool:
        probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        try:
            with urllib.request.urlopen(
                f"http://{probe_host}:{port}/dedup", timeout=1.5
            ) as response:
                body = response.read(32768)
                return "人工调整图形树".encode("utf-8") in body
        except (OSError, urllib.error.URLError):
            return False

    if port_is_open(args.host, args.port):
        if existing_viewer(args.host, args.port):
            print(
                f"RoboCOIN Viewer is already running at "
                f"http://127.0.0.1:{args.port}"
            )
            raise SystemExit(0)
        if not args.auto_port:
            raise SystemExit(
                f"Port {args.port} is used by another program. "
                f"Use --port 8889 or --auto-port."
            )
        requested_port = args.port
        for candidate_port in range(requested_port + 1, requested_port + 101):
            if not port_is_open(args.host, candidate_port):
                args.port = candidate_port
                print(f"Port {requested_port} is occupied; using {args.port} instead.")
                break
        else:
            raise SystemExit("Could not find an available port in the next 100 ports.")

    uvicorn.run(app, host=args.host, port=args.port)
