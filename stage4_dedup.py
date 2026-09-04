#!/usr/bin/env python3
"""CLIP similarity recall, reviewed merges, and a clean new object library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from project_paths import resolve_project_path
from hardware_profiles import PROFILES, default_profile_name, get_profile

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "objects" / "new_library_work"
ATTR_CACHE = WORK_DIR / "attributes.jsonl"
DECISIONS_PATH = WORK_DIR / "dedup_decisions.json"
OVERRIDES_PATH = WORK_DIR / "attribute_overrides.json"
CANDIDATES_PATH = WORK_DIR / "dedup_candidates.json"
INSTANCES_PATH = WORK_DIR / "instances.json"
EMBEDDINGS_PATH = WORK_DIR / "clip_embeddings.npz"
LIBRARY_DIR = BASE_DIR / "objects" / "new_library"
LIBRARY_ARCHIVE_DIR = BASE_DIR / "objects" / "new_library_archive"
ID_REGISTRY_PATH = BASE_DIR / "objects" / "object_id_registry.json"
DEFAULT_CLIP_DIR = BASE_DIR / "models" / "clip"
CLIP_MODEL_ID = "ViT-L/14"
CLIP_MODEL_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/"
    "ViT-L-14.pt"
)
CLIP_MODEL_SIZE = 932_768_134
CLIP_MODEL_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
PROGRESS_PREFIX = "@@PROGRESS "
ATTRIBUTE_KEYS = ("category", "color", "material", "shape", "texture")
CONTROLLED_ATTRIBUTE_KEYS = ("color", "material", "shape", "texture")


def items_fingerprint(items: list[dict]) -> str:
    """Fingerprint the exact instance/image generation consumed by dedup."""
    records = []
    for item in items:
        records.append(json.dumps({
            "instance_id": item["instance_id"],
            "image_fingerprint": item.get("fingerprint", ""),
            "attributes": item.get("attributes", {}),
            "category_path": item.get("category_path", []),
            "attribute_paths": item.get("attribute_paths", {}),
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    payload = "|".join(sorted(records))
    return hashlib.sha256(payload.encode()).hexdigest()


def emit_progress(percent: float, message: str) -> None:
    payload = {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "message": message,
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_attributes() -> list[dict]:
    if not ATTR_CACHE.is_file():
        raise FileNotFoundError(f"{ATTR_CACHE} not found; run stage3_attribute.py")
    latest = {}
    for line in ATTR_CACHE.read_text().splitlines():
        try:
            item = json.loads(line)
            image_path = resolve_project_path(item.get("best_quality_path", ""))
            if (
                item.get("tracker_backend") == "sam3"
                and "attributes" in item
                and "error" not in item
                and item.get("library_eligible", True)
                and image_path.is_file()
            ):
                latest[item["instance_id"]] = item
        except (KeyError, json.JSONDecodeError):
            continue
    if not latest:
        raise RuntimeError("No valid VLM attribute entries found")
    overrides = {}
    if OVERRIDES_PATH.is_file():
        try:
            overrides = json.loads(OVERRIDES_PATH.read_text())
        except json.JSONDecodeError:
            overrides = {}
    items = [latest[key] for key in sorted(latest)]
    for item in items:
        override = overrides.get(item["instance_id"])
        if isinstance(override, dict):
            item = dict(item)
            original_category = str(item.get("attributes", {}).get("category", "unknown"))
            item["attributes"] = {
                key: override.get(key, item["attributes"].get(key, "unknown"))
                for key in ATTRIBUTE_KEYS
            }
            if (
                "category" in override
                and str(item["attributes"]["category"]) != original_category
            ):
                # A free-form human category no longer has the VLM's WordNet path.
                # Let dedup_tree.category_path build a new free-category node.
                item.pop("category_path", None)
                item.pop("category_synset", None)
                item.pop("wordnet_warning", None)
            latest[item["instance_id"]] = item
    return [latest[key] for key in sorted(latest)]


def ensure_clip_checkpoint(download_root: Path) -> Path:
    """Download ViT-L/14 with resume support before OpenAI CLIP loads it."""
    download_root.mkdir(parents=True, exist_ok=True)
    checkpoint = download_root / "ViT-L-14.pt"
    current_size = checkpoint.stat().st_size if checkpoint.is_file() else 0
    if current_size == CLIP_MODEL_SIZE:
        digest = file_sha256(checkpoint)
        if digest == CLIP_MODEL_SHA256:
            return checkpoint
        current_size = 0
    elif current_size > CLIP_MODEL_SIZE:
        current_size = 0

    headers = {"Range": f"bytes={current_size}-"} if current_size else {}
    request = urllib.request.Request(CLIP_MODEL_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = current_size > 0 and response.status == 206
        if not resumed:
            current_size = 0
        mode = "ab" if resumed else "wb"
        print(
            f"Downloading ViT-L/14 from {current_size / 1_000_000:.1f} MB "
            f"of {CLIP_MODEL_SIZE / 1_000_000:.1f} MB…",
            flush=True,
        )
        downloaded = current_size
        last_percent = downloaded * 100 // CLIP_MODEL_SIZE
        with checkpoint.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                percent = downloaded * 100 // CLIP_MODEL_SIZE
                if percent >= last_percent + 5:
                    print(f"  ViT-L/14 download: {percent}%", flush=True)
                    last_percent = percent

    if checkpoint.stat().st_size != CLIP_MODEL_SIZE:
        raise RuntimeError(
            f"ViT-L/14 download incomplete: {checkpoint.stat().st_size} / "
            f"{CLIP_MODEL_SIZE} bytes; run again to resume"
        )
    digest = file_sha256(checkpoint)
    if digest != CLIP_MODEL_SHA256:
        raise RuntimeError(
            f"ViT-L/14 checksum mismatch at {checkpoint}; remove this damaged file "
            "and run again"
        )
    return checkpoint


def load_clip_model(download_root: Path):
    import clip

    ensure_clip_checkpoint(download_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(
        CLIP_MODEL_ID, device=device, download_root=str(download_root)
    )
    model.eval()
    return model, preprocess, device


def attribute_text(item: dict) -> str:
    attrs = item.get("attributes", {})
    values = []
    for key in ATTRIBUTE_KEYS:
        value = str(attrs.get(key, "")).strip()
        if value and value.lower() != "unknown":
            values.append(value)
    return ", ".join(values) or "unknown object"


def _attribute_family(item: dict, key: str) -> tuple[str, str]:
    value = str(item.get("attributes", {}).get(key, "unknown")).strip().lower()
    paths = item.get("attribute_paths", {})
    path = paths.get(key) if isinstance(paths, dict) else None
    if isinstance(path, list) and len(path) >= 2 and isinstance(path[1], dict):
        return str(path[1].get("id", value)), value
    return value, value


def controlled_attribute_similarity(left: dict, right: dict) -> float:
    scores = []
    for key in CONTROLLED_ATTRIBUTE_KEYS:
        left_family, left_value = _attribute_family(left, key)
        right_family, right_value = _attribute_family(right, key)
        if "unknown" in {left_value, right_value}:
            scores.append(0.5)
        elif left_value == right_value:
            scores.append(1.0)
        elif left_family == right_family:
            scores.append(0.65)
        else:
            scores.append(0.0)
    return float(sum(scores) / len(scores))


def compute_embeddings(items: list[dict], clip_dir: Path, batch_size: int = 16):
    import clip

    emit_progress(2, "正在加载 CLIP ViT-L/14")
    model, preprocess, device = load_clip_model(clip_dir)
    visual, text = [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        images = []
        for item in chunk:
            with Image.open(resolve_project_path(item["best_quality_path"])) as source:
                images.append(preprocess(source.convert("RGB")))
        image_batch = torch.stack(images).to(device, non_blocking=True)
        tokens = clip.tokenize([attribute_text(item) for item in chunk]).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            image_features = model.encode_image(image_batch).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        visual.extend(image_features.cpu().numpy())
        text.extend(text_features.cpu().numpy())
        completed = min(start + len(chunk), len(items))
        emit_progress(
            5 + 50 * completed / max(1, len(items)),
            f"CLIP 特征计算 {completed}/{len(items)}",
        )
    return np.asarray(visual, np.float32), np.asarray(text, np.float32)


def pair_key(left: str, right: str) -> str:
    return "__".join(sorted((left, right)))


def load_decisions() -> dict[str, dict]:
    if not DECISIONS_PATH.is_file():
        return {}
    try:
        data = json.loads(DECISIONS_PATH.read_text())
        normalized = {}
        for key, value in data.items():
            if isinstance(value, str) and value in {"merge", "reject"}:
                normalized[key] = {"decision": value, "canonical_instance_id": None}
            elif isinstance(value, dict) and value.get("decision") in {"merge", "reject"}:
                normalized[key] = value
        return normalized
    except json.JSONDecodeError:
        return {}


def find_candidates(
    items: list[dict],
    visual: np.ndarray,
    text: np.ndarray,
    clip_recall_threshold: float,
    attribute_threshold: float = 0.50,
    progress_callback=None,
) -> list[dict]:
    decisions = load_decisions()
    candidates = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            visual_similarity = float(visual[left] @ visual[right])
            text_similarity = float(text[left] @ text[right])
            left_category = str(items[left]["attributes"].get("category", "")).lower()
            right_category = str(items[right]["attributes"].get("category", "")).lower()
            same_category = bool(left_category and left_category == right_category)
            attribute_similarity = controlled_attribute_similarity(
                items[left], items[right]
            )
            key = pair_key(items[left]["instance_id"], items[right]["instance_id"])
            manual = decisions.get(key)
            if manual is None and (
                visual_similarity < clip_recall_threshold
                or not same_category
                or attribute_similarity < attribute_threshold
            ):
                continue
            candidates.append({
                "pair_id": key,
                "left_instance_id": items[left]["instance_id"],
                "right_instance_id": items[right]["instance_id"],
                "visual_similarity": round(visual_similarity, 6),
                "text_similarity": round(text_similarity, 6),
                "attribute_similarity": round(attribute_similarity, 6),
                "same_category": same_category,
                "left_category": left_category,
                "right_category": right_category,
                "decision": manual["decision"] if manual else "pending",
                "canonical_instance_id": manual.get("canonical_instance_id") if manual else None,
            })
        if progress_callback is not None:
            progress_callback(left + 1, len(items))
    candidates.sort(key=lambda item: item["visual_similarity"], reverse=True)
    return candidates


def apply_clip_decisions(candidates: list[dict]) -> list[dict]:
    """Keep manual decisions; leave attribute+CLIP matches for human review."""
    manual = load_decisions()
    decided = []
    for source in candidates:
        item = dict(source)
        saved = manual.get(item["pair_id"])
        if saved:
            item["decision"] = saved["decision"]
            item["canonical_instance_id"] = saved.get("canonical_instance_id")
        else:
            item["decision"] = "pending"
            item["canonical_instance_id"] = None
        for key in (
            "vlm_verification", "vlm_error", "vlm_prefiltered",
            "vlm_verification_mode", "auto_merge_eligible",
        ):
            item.pop(key, None)
        decided.append(item)
    decided.sort(key=lambda item: item.get("visual_similarity", -1.0), reverse=True)
    return decided


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def merge_attributes(items: list[dict]) -> dict:
    merged = {}
    for key in ATTRIBUTE_KEYS:
        values = [str(item["attributes"].get(key, "unknown")) for item in items]
        merged[key] = Counter(values).most_common(1)[0][0]
    return merged


def _name_slug(category: str) -> str:
    """Convert a category label to a filesystem/UI-safe readable name stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(category).strip().lower()).strip("_")
    return slug or "object"


def load_id_registry(path: Path = ID_REGISTRY_PATH) -> dict:
    """Load the permanent object-ID registry kept outside replaceable builds."""
    if not path.is_file():
        return {"version": 1, "next_indices": {}, "objects": {}}
    registry = json.loads(path.read_text())
    if registry.get("version") != 1:
        raise RuntimeError(f"Unsupported object ID registry version: {registry.get('version')}")
    registry.setdefault("next_indices", {})
    registry.setdefault("objects", {})
    return registry


def assign_persistent_library_ids(
    groups: list[list[dict]], registry: dict
) -> tuple[dict[str, str], dict]:
    """Inherit permanent ``category_N`` IDs by instance overlap.

    Existing IDs are never renamed or reused.  A changed group inherits the old
    ID with the strongest overlap; unmatched groups consume a fresh per-category
    sequence number.
    """
    group_records = []
    for group in groups:
        instance_ids = frozenset(item["instance_id"] for item in group)
        signature = "|".join(sorted(instance_ids))
        category = merge_attributes(group).get("category", "object")
        group_records.append((signature, instance_ids, _name_slug(category), category))
    group_records.sort(key=lambda record: record[0])

    previous = registry.get("objects", {})
    candidates = []
    for group_index, (signature, instance_ids, _stem, _category) in enumerate(group_records):
        for object_id, record in previous.items():
            old_ids = set(record.get("instance_ids", []))
            overlap = len(instance_ids & old_ids)
            if overlap:
                union_size = len(instance_ids | old_ids)
                candidates.append((
                    -overlap,
                    -(overlap / max(1, union_size)),
                    object_id,
                    signature,
                    group_index,
                ))

    assigned_groups = set()
    assigned_ids = set()
    assignments = {}
    for _neg_overlap, _neg_jaccard, object_id, signature, group_index in sorted(candidates):
        if group_index in assigned_groups or object_id in assigned_ids:
            continue
        assignments[signature] = object_id
        assigned_groups.add(group_index)
        assigned_ids.add(object_id)

    next_indices = {
        str(stem): int(index)
        for stem, index in registry.get("next_indices", {}).items()
    }
    reserved_ids = set(previous)
    for _signature, _instance_ids, stem, _category in group_records:
        next_indices.setdefault(stem, 0)
        while f"{stem}_{next_indices[stem]}" in reserved_ids:
            next_indices[stem] += 1

    for signature, _instance_ids, stem, _category in group_records:
        if signature in assignments:
            continue
        index = next_indices[stem]
        object_id = f"{stem}_{index}"
        while object_id in reserved_ids:
            index += 1
            object_id = f"{stem}_{index}"
        assignments[signature] = object_id
        reserved_ids.add(object_id)
        next_indices[stem] = index + 1

    now = datetime.now(timezone.utc).isoformat()
    updated_objects = {
        object_id: {**record, "active": False}
        for object_id, record in previous.items()
    }
    for signature, instance_ids, _stem, category in group_records:
        object_id = assignments[signature]
        old_record = updated_objects.get(object_id, {})
        updated_objects[object_id] = {
            **old_record,
            "id": object_id,
            "category_at_creation": old_record.get("category_at_creation", category),
            "current_category": category,
            "instance_ids": sorted(instance_ids),
            "active": True,
            "created_at": old_record.get("created_at", now),
            "updated_at": now,
        }
    updated_registry = {
        "version": 1,
        "next_indices": next_indices,
        "objects": updated_objects,
        "updated_at": now,
    }
    return assignments, updated_registry


def build_library(items: list[dict], candidates: list[dict], progress_callback=None) -> dict:
    by_id = {item["instance_id"]: item for item in items}
    union = UnionFind(by_id)
    for candidate in candidates:
        if candidate["decision"] in {"merge", "auto_merge"}:
            union.union(candidate["left_instance_id"], candidate["right_instance_id"])
    groups = {}
    for instance_id in by_id:
        groups.setdefault(union.find(instance_id), []).append(by_id[instance_id])
    grouped_items = list(groups.values())
    id_registry = load_id_registry()
    library_ids, updated_registry = assign_persistent_library_ids(
        grouped_items, id_registry
    )

    temp_dir = LIBRARY_DIR.parent / f".new_library_build_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    library = {}
    try:
        for completed, group in enumerate(grouped_items, start=1):
            group.sort(key=lambda item: item["instance_id"])
            signature = "|".join(item["instance_id"] for item in group)
            uid = library_ids[signature]
            object_dir = temp_dir / uid
            instance_dir = object_dir / "instances"
            instance_dir.mkdir(parents=True)
            group_ids = {item["instance_id"] for item in group}
            nominated = {
                item.get("canonical_instance_id") for item in candidates
                if item["decision"] == "merge"
                and item["left_instance_id"] in group_ids
                and item["right_instance_id"] in group_ids
                and item.get("canonical_instance_id") in group_ids
            }
            canonical_pool = [
                item for item in group if item["instance_id"] in nominated
            ] or group
            canonical_item = max(
                canonical_pool,
                key=lambda item: (
                    float(item.get("representative_quality_score", 0.0)),
                    item["instance_id"],
                ),
            )
            canonical_source = resolve_project_path(canonical_item["canonical_source"])
            shutil.copy2(canonical_source, object_dir / "canonical.jpg")
            source_instances = []
            copied_instances = []
            for item in group:
                best_target = instance_dir / f"{item['instance_id']}_best.jpg"
                shutil.copy2(resolve_project_path(item["best_quality_path"]), best_target)
                copied_instances.append(
                    f"objects/new_library/{uid}/instances/{best_target.name}"
                )
                source_instances.append({
                    "instance_id": item["instance_id"],
                    "tracker_backend": item["tracker_backend"],
                    "session_key": item["session_key"],
                    "object_id": item["object_id"],
                    "representative_quality_score": float(
                        item.get("representative_quality_score", 0.0)
                    ),
                    "best_quality_source": item["best_quality_path"],
                    "attributes": item["attributes"],
                })
            entry = {
                "id": uid,
                "name": uid,
                "canonical_path": f"objects/new_library/{uid}/canonical.jpg",
                "canonical_instance_id": canonical_item["instance_id"],
                "canonical_selection": "manual" if nominated else "highest_quality",
                "attributes": merge_attributes(group),
                "instance_count": len(group),
                "instances": copied_instances,
                "source_instances": source_instances,
            }
            library[uid] = entry
            (object_dir / "metadata.json").write_text(
                json.dumps(entry, indent=2, ensure_ascii=False)
            )
            if progress_callback is not None:
                progress_callback(completed, len(groups))
        (temp_dir / "index.json").write_text(
            json.dumps(library, indent=2, ensure_ascii=False)
        )
        (temp_dir / "build_state.json").write_text(json.dumps({
            "source_fingerprint": items_fingerprint(items),
            "instance_count": len(items),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False))
        if LIBRARY_DIR.exists():
            LIBRARY_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            archive = LIBRARY_ARCHIVE_DIR / datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            if archive.exists():
                archive = archive.with_name(archive.name + "_" + uuid.uuid4().hex[:6])
            LIBRARY_DIR.replace(archive)
        temp_dir.replace(LIBRARY_DIR)
        registry_temp = ID_REGISTRY_PATH.with_suffix(".json.tmp")
        registry_temp.write_text(
            json.dumps(updated_registry, indent=2, ensure_ascii=False)
        )
        registry_temp.replace(ID_REGISTRY_PATH)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-recall-threshold", type=float, default=0.88)
    parser.add_argument("--attribute-threshold", type=float, default=0.50)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument(
        "--hardware-profile", choices=tuple(PROFILES), default=default_profile_name()
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--embeddings-only", action="store_true",
        help="Run only CLIP GPU inference; do pair search/library building locally",
    )
    parser.add_argument(
        "--reuse-embeddings", action="store_true",
        help="Reuse server-produced CLIP features and run CPU post-processing",
    )
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="Reuse existing attribute+CLIP candidates and apply manual decisions",
    )
    args = parser.parse_args()
    if args.embeddings_only and args.reuse_embeddings:
        parser.error("--embeddings-only and --reuse-embeddings are mutually exclusive")
    profile = get_profile(args.hardware_profile)
    if args.batch_size is None:
        args.batch_size = profile.clip_batch_size
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not -1 <= args.clip_recall_threshold <= 1:
        parser.error("--clip-recall-threshold must be between -1 and 1")
    if not 0 <= args.attribute_threshold <= 1:
        parser.error("--attribute-threshold must be between 0 and 1")

    items = load_attributes()
    print(f"Loaded {len(items)} latest VLM annotations")
    emit_progress(0, f"读取到 {len(items)} 个有效属性标注")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse_candidates:
        if not CANDIDATES_PATH.is_file():
            raise FileNotFoundError(
                f"{CANDIDATES_PATH} not found; run a full dedup build first"
            )
        candidate_data = json.loads(CANDIDATES_PATH.read_text())
        current_fingerprint = items_fingerprint(items)
        if candidate_data.get("source_fingerprint") != current_fingerprint:
            raise RuntimeError(
                "Cached dedup candidates are stale; run stage4_dedup.py without "
                "--reuse-candidates before applying review decisions"
            )
        valid_ids = {item["instance_id"] for item in items}
        candidates = [
            candidate for candidate in candidate_data.get("candidates", [])
            if candidate.get("left_instance_id") in valid_ids
            and candidate.get("right_instance_id") in valid_ids
        ]
        candidates = apply_clip_decisions(candidates)
        INSTANCES_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        CANDIDATES_PATH.write_text(json.dumps({
            "source_fingerprint": current_fingerprint,
            "clip_recall_threshold": args.clip_recall_threshold,
            "attribute_threshold": args.attribute_threshold,
            "dedup_method": "attributes_clip_human_review",
            "candidates": candidates,
        }, indent=2, ensure_ascii=False))
        emit_progress(10, f"已读取 {len(candidates)} 个已缓存候选对")

        def report_reuse_build(completed: int, total: int) -> None:
            emit_progress(
                10 + 90 * completed / max(1, total),
                f"重建物体库 {completed}/{total}",
            )

        library = build_library(
            items, candidates, progress_callback=report_reuse_build
        )
        emit_progress(100, f"物体库重建完成：{len(library)} 个物体")
        pending = sum(item["decision"] == "pending" for item in candidates)
        automatic = sum(item["decision"] == "auto_merge" for item in candidates)
        merged = sum(item["decision"] == "merge" for item in candidates)
        print(
            f"Reused {len(candidates)} candidate pairs ({automatic} automatic, "
            f"{pending} pending, {merged} approved merges)"
        )
        print(f"New library: {len(library)} object(s) -> {LIBRARY_DIR / 'index.json'}")
        return

    print(f"Hardware profile: {profile.name}; CLIP batch size: {args.batch_size}")
    instance_ids = np.asarray([item["instance_id"] for item in items])
    if args.reuse_embeddings:
        if not EMBEDDINGS_PATH.is_file():
            raise FileNotFoundError(f"Missing server output: {EMBEDDINGS_PATH}")
        saved = np.load(EMBEDDINGS_PATH)
        if saved["instance_ids"].tolist() != instance_ids.tolist():
            raise RuntimeError(
                "CLIP embeddings do not match current attributes; rerun server clip"
            )
        visual = np.asarray(saved["visual"], np.float32)
        text = np.asarray(saved["text"], np.float32)
        emit_progress(55, f"Reused CLIP embeddings for {len(items)} objects")
    else:
        visual, text = compute_embeddings(items, args.clip_dir, args.batch_size)
        np.savez_compressed(
            EMBEDDINGS_PATH,
            instance_ids=instance_ids,
            visual=visual,
            text=text,
        )
    INSTANCES_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    if args.embeddings_only:
        emit_progress(100, f"CLIP embeddings complete: {len(items)} objects")
        print(f"Embeddings only: {EMBEDDINGS_PATH}")
        return
    def report_candidate_search(completed: int, total: int) -> None:
        emit_progress(
            55 + 7 * completed / max(1, total),
            f"CLIP两两候选计算 {completed}/{total}",
        )

    candidates = find_candidates(
        items,
        visual,
        text,
        args.clip_recall_threshold,
        args.attribute_threshold,
        progress_callback=report_candidate_search,
    )
    emit_progress(62, f"CLIP召回完成：{len(candidates)} 个候选对")
    candidates = apply_clip_decisions(candidates)
    emit_progress(70, "已按类别、属性和 CLIP 生成待人工审核候选")
    CANDIDATES_PATH.write_text(json.dumps({
        "source_fingerprint": items_fingerprint(items),
        "clip_recall_threshold": args.clip_recall_threshold,
        "attribute_threshold": args.attribute_threshold,
        "dedup_method": "attributes_clip_human_review",
        "candidates": candidates,
    }, indent=2, ensure_ascii=False))
    def report_library_build(completed: int, total: int) -> None:
        emit_progress(
            70 + 30 * completed / max(1, total),
            f"重建物体库 {completed}/{total}",
        )

    library = build_library(
        items, candidates, progress_callback=report_library_build
    )
    emit_progress(100, f"CLIP与去重计算完成：{len(library)} 个物体")
    pending = sum(item["decision"] == "pending" for item in candidates)
    automatic = sum(item["decision"] == "auto_merge" for item in candidates)
    merged = sum(item["decision"] == "merge" for item in candidates)
    print(
        f"Attribute+CLIP candidate pairs: {len(candidates)} ({automatic} automatic, "
        f"{pending} pending, {merged} approved merges)"
    )
    print(f"New library: {len(library)} object(s) -> {LIBRARY_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
