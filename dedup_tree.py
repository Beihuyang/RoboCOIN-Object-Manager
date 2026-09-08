#!/usr/bin/env python3
"""Attribute-first, CLIP-second clustering with a persistent editable tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "objects" / "new_library_work"
EMBEDDINGS_PATH = WORK_DIR / "clip_embeddings.npz"
TREE_PATH = WORK_DIR / "dedup_tree_layout.json"
TREE_ARCHIVE_DIR = WORK_DIR / "dedup_tree_archive"
DEFAULT_CLIP_THRESHOLD = 0.82
DEFAULT_ATTRIBUTE_THRESHOLD = 0.50
ATTRIBUTE_KEYS = ("color", "size", "material", "shape", "texture")
PROGRESS_PREFIX = "@@PROGRESS "


def emit_progress(percent: float, message: str) -> None:
    payload = {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "message": message,
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _node_summary(node: dict) -> dict:
    return {
        key: node[key]
        for key in ("id", "label", "definition", "label_zh", "definition_zh")
        if key in node
    }


def _free_category_node(category: str) -> dict:
    label = category.strip() or "unknown"
    digest = hashlib.sha1(label.lower().encode()).hexdigest()[:12]
    return {
        "id": f"category.free.{digest}",
        "label": label,
        "definition": "legacy or manually supplied category",
    }


def category_path(item: dict) -> list[dict]:
    stored = item.get("category_path")
    if isinstance(stored, list) and stored:
        return [_node_summary(node) for node in stored if isinstance(node, dict)]
    return [
        {"id": "entity.n.01", "label": "entity", "definition": "tree root"},
        _free_category_node(str(item.get("attributes", {}).get("category", "unknown"))),
    ]


def _attribute_family(item: dict, key: str) -> tuple[str, str]:
    value = str(item.get("attributes", {}).get(key, "unknown")).strip().lower()
    paths = item.get("attribute_paths", {})
    path = paths.get(key) if isinstance(paths, dict) else None
    if isinstance(path, list) and len(path) >= 2 and isinstance(path[1], dict):
        return str(path[1].get("id", value)), value
    return value, value


def attribute_similarity(left: dict, right: dict) -> float:
    scores = []
    for key in ATTRIBUTE_KEYS:
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


def _embedding_map() -> dict[str, np.ndarray]:
    if not EMBEDDINGS_PATH.is_file():
        return {}
    data = np.load(EMBEDDINGS_PATH)
    return {
        str(instance_id): vector
        for instance_id, vector in zip(data["instance_ids"], data["visual"])
    }


def _visual_similarity(left_id: str, right_id: str,
                       embeddings: dict[str, np.ndarray]) -> float:
    if left_id not in embeddings or right_id not in embeddings:
        return -1.0
    return float(embeddings[left_id] @ embeddings[right_id])


def _cluster_node(items: list[dict], embeddings: dict[str, np.ndarray],
                  clip_threshold: float, attribute_threshold: float) -> list[list[str]]:
    by_id = {item["instance_id"]: item for item in items}
    groups = [{instance_id} for instance_id in sorted(by_id)]

    def compatible(left_group: set[str], right_group: set[str]) -> tuple[bool, float]:
        similarities = []
        for left_id in left_group:
            for right_id in right_group:
                attr = attribute_similarity(by_id[left_id], by_id[right_id])
                clip = _visual_similarity(left_id, right_id, embeddings)
                if attr < attribute_threshold or clip < clip_threshold:
                    return False, -1.0
                similarities.append(clip)
        return True, min(similarities, default=-1.0)

    while True:
        best = None
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                allowed, score = compatible(groups[left_index], groups[right_index])
                if allowed and (best is None or score > best[0]):
                    best = (score, left_index, right_index)
        if best is None:
            break
        _, left_index, right_index = best
        groups[left_index] |= groups[right_index]
        groups.pop(right_index)
    return [sorted(group) for group in groups]


def _cluster_id(node_id: str, members: list[str]) -> str:
    digest = hashlib.sha1((node_id + "|" + "|".join(sorted(members))).encode()).hexdigest()[:14]
    return f"cluster_{digest}"


def _instance_fingerprints(items: list[dict]) -> dict[str, str]:
    return {
        str(item["instance_id"]): str(item.get("fingerprint", ""))
        for item in items
        if item.get("instance_id") and item.get("fingerprint")
    }


def backfill_layout_instance_fingerprints(
    layout: dict, items: list[dict]
) -> tuple[dict, int]:
    """Add a per-instance baseline to legacy layouts without changing placement."""
    fingerprints = dict(layout.get("instance_fingerprints", {}))
    current = _instance_fingerprints(items)
    known_ids = {
        value
        for cluster in layout.get("clusters", [])
        for value in cluster.get("member_ids", [])
    } | set(layout.get("trash_instance_ids", []))
    added = 0
    for instance_id in known_ids:
        if instance_id not in fingerprints and current.get(instance_id):
            fingerprints[instance_id] = current[instance_id]
            added += 1
    layout["instance_fingerprints"] = fingerprints
    return layout, added


def inherit_unchanged_adjustments(previous: dict, generated: dict) -> dict:
    """Keep old tree placement only for instances whose image fingerprint is unchanged."""
    old_fingerprints = previous.get("instance_fingerprints", {})
    new_fingerprints = generated.get("instance_fingerprints", {})
    unchanged = {
        instance_id for instance_id, fingerprint in new_fingerprints.items()
        if fingerprint and old_fingerprints.get(instance_id) == fingerprint
    }
    old_assigned = {
        value
        for cluster in previous.get("clusters", [])
        for value in cluster.get("member_ids", [])
    } | set(previous.get("trash_instance_ids", []))
    preserved_ids = unchanged & old_assigned
    if not preserved_ids:
        generated["inheritance"] = {
            "unchanged_instances": 0,
            "changed_or_new_instances": len(new_fingerprints),
        }
        return generated

    auto_clusters = []
    for cluster in generated.get("clusters", []):
        members = [
            value for value in cluster.get("member_ids", [])
            if value not in preserved_ids
        ]
        if not members:
            continue
        updated = {**cluster, "member_ids": sorted(members)}
        updated["id"] = _cluster_id(updated["node_id"], updated["member_ids"])
        auto_clusters.append(updated)

    preserved_clusters = []
    required_node_ids = set()
    for cluster in previous.get("clusters", []):
        members = [
            value for value in cluster.get("member_ids", [])
            if value in preserved_ids
        ]
        if not members:
            continue
        preserved_clusters.append({**cluster, "member_ids": sorted(members)})
        required_node_ids.add(cluster["node_id"])

    old_nodes = {node["id"]: node for node in previous.get("nodes", [])}
    pending = list(required_node_ids)
    while pending:
        node = old_nodes.get(pending.pop())
        if not node:
            continue
        parent_id = node.get("parent_id")
        if parent_id and parent_id not in required_node_ids:
            required_node_ids.add(parent_id)
            pending.append(parent_id)
    nodes_by_id = {node["id"]: node for node in generated.get("nodes", [])}
    for node_id in required_node_ids:
        if node_id in old_nodes:
            nodes_by_id.setdefault(node_id, old_nodes[node_id])

    generated["nodes"] = list(nodes_by_id.values())
    generated["clusters"] = preserved_clusters + auto_clusters
    generated["trash_instance_ids"] = sorted(
        preserved_ids & set(previous.get("trash_instance_ids", []))
    )
    generated["inheritance"] = {
        "unchanged_instances": len(preserved_ids),
        "changed_or_new_instances": len(new_fingerprints) - len(preserved_ids),
    }
    return generated


def generate_layout(items: list[dict], clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
                    attribute_threshold: float = DEFAULT_ATTRIBUTE_THRESHOLD,
                    progress_callback=None) -> dict:
    embeddings = _embedding_map()
    nodes = {}
    items_by_leaf = {}
    for item in items:
        path = category_path(item)
        parent_id = None
        for node in path:
            node_id = node["id"]
            nodes.setdefault(node_id, {**node, "parent_id": parent_id})
            parent_id = node_id
        items_by_leaf.setdefault(path[-1]["id"], []).append(item)
    clusters = []
    leaf_groups = sorted(items_by_leaf.items())
    for completed, (node_id, node_items) in enumerate(leaf_groups, start=1):
        for members in _cluster_node(
            node_items, embeddings, clip_threshold, attribute_threshold
        ):
            clusters.append({
                "id": _cluster_id(node_id, members),
                "node_id": node_id,
                "member_ids": members,
                "origin": "attribute_clip",
            })
        if progress_callback is not None:
            progress_callback(completed, len(leaf_groups))
    from stage4_dedup import items_fingerprint

    fingerprint = items_fingerprint(items)
    return {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": fingerprint,
        "instance_fingerprints": _instance_fingerprints(items),
        "settings": {
            "clip_threshold": clip_threshold,
            "attribute_threshold": attribute_threshold,
        },
        "nodes": list(nodes.values()),
        "clusters": clusters,
        "trash_instance_ids": [],
    }


def load_layout() -> dict:
    if not TREE_PATH.is_file():
        raise FileNotFoundError(f"Tree layout not found: {TREE_PATH}")
    return json.loads(TREE_PATH.read_text())


def save_layout(layout: dict, archive_existing: bool = False) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if archive_existing and TREE_PATH.is_file():
        TREE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        target = TREE_ARCHIVE_DIR / f"layout_{stamp}_{uuid.uuid4().hex[:6]}.json"
        shutil.copy2(TREE_PATH, target)
    temp = TREE_PATH.with_name(TREE_PATH.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(layout, indent=2, ensure_ascii=False))
    temp.replace(TREE_PATH)


def _remove_from_clusters(layout: dict, instance_id: str) -> None:
    for cluster in layout["clusters"]:
        cluster["member_ids"] = [
            value for value in cluster["member_ids"] if value != instance_id
        ]
    layout["clusters"] = [cluster for cluster in layout["clusters"] if cluster["member_ids"]]


def move_instance(layout: dict, instance_id: str, target_cluster_id: str | None,
                  target_node_id: str | None = None) -> dict:
    if target_cluster_id and any(
        cluster["id"] == target_cluster_id
        and instance_id in cluster.get("member_ids", [])
        for cluster in layout["clusters"]
    ):
        return layout
    _remove_from_clusters(layout, instance_id)
    layout["trash_instance_ids"] = [
        value for value in layout.get("trash_instance_ids", []) if value != instance_id
    ]
    if target_cluster_id:
        cluster = next(
            (item for item in layout["clusters"] if item["id"] == target_cluster_id), None
        )
        if cluster is None:
            raise KeyError("Target cluster does not exist")
        cluster["member_ids"].append(instance_id)
        cluster["member_ids"].sort()
        cluster["origin"] = "manual"
    else:
        node_ids = {node["id"] for node in layout["nodes"]}
        if target_node_id not in node_ids:
            raise KeyError("Target category node does not exist")
        cluster_id = f"cluster_manual_{uuid.uuid4().hex[:12]}"
        layout["clusters"].append({
            "id": cluster_id,
            "node_id": target_node_id,
            "member_ids": [instance_id],
            "origin": "manual",
        })
    return layout


def trash_instance(layout: dict, instance_id: str) -> dict:
    _remove_from_clusters(layout, instance_id)
    trash = set(layout.get("trash_instance_ids", []))
    trash.add(instance_id)
    layout["trash_instance_ids"] = sorted(trash)
    return layout


def rebuild_library(layout: dict, progress_callback=None) -> dict:
    from stage4_dedup import build_library, load_attributes

    items = load_attributes()
    trashed = set(layout.get("trash_instance_ids", []))
    items = [item for item in items if item["instance_id"] not in trashed]
    valid_ids = {item["instance_id"] for item in items}
    candidates = []
    for cluster in layout.get("clusters", []):
        members = [value for value in cluster["member_ids"] if value in valid_ids]
        for left_index, left_id in enumerate(members):
            for right_id in members[left_index + 1:]:
                candidates.append({
                    "pair_id": "__".join(sorted((left_id, right_id))),
                    "left_instance_id": left_id,
                    "right_instance_id": right_id,
                    "decision": "merge",
                    "canonical_instance_id": None,
                })
    return build_library(items, candidates, progress_callback=progress_callback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--clip-threshold", type=float, default=DEFAULT_CLIP_THRESHOLD)
    parser.add_argument("--attribute-threshold", type=float, default=DEFAULT_ATTRIBUTE_THRESHOLD)
    args = parser.parse_args()
    from stage4_dedup import load_attributes

    if args.regenerate or not TREE_PATH.is_file():
        emit_progress(0, "正在按属性和 CLIP 生成类别簇")

        previous_layout = load_layout() if TREE_PATH.is_file() else None
        current_items = load_attributes()
        if previous_layout is not None:
            previous_layout, _ = backfill_layout_instance_fingerprints(
                previous_layout, current_items
            )

        def report_layout(completed: int, total: int) -> None:
            emit_progress(
                55 * completed / max(1, total),
                f"类别节点聚类 {completed}/{total}",
            )

        layout = generate_layout(
            current_items, args.clip_threshold, args.attribute_threshold,
            progress_callback=report_layout,
        )
        if previous_layout is not None:
            layout = inherit_unchanged_adjustments(previous_layout, layout)
            inheritance = layout.get("inheritance", {})
            print(
                "Tree inheritance: "
                f"{inheritance.get('unchanged_instances', 0)} unchanged object(s) "
                "kept their manual placement; "
                f"{inheritance.get('changed_or_new_instances', 0)} changed/new "
                "object(s) regenerated.",
                flush=True,
            )
        save_layout(layout, archive_existing=args.regenerate)
    else:
        layout = load_layout()
        emit_progress(55, "已读取现有类别树")

    def report_library(completed: int, total: int) -> None:
        emit_progress(
            55 + 45 * completed / max(1, total),
            f"同步物体库 {completed}/{total}",
        )

    library = rebuild_library(layout, progress_callback=report_library)
    emit_progress(100, "类别树和物体库已更新")
    print(
        f"Dedup tree: {len(layout['nodes'])} nodes, {len(layout['clusters'])} clusters, "
        f"{len(layout.get('trash_instance_ids', []))} trashed"
    )
    print(f"New library: {len(library)} object(s)")


if __name__ == "__main__":
    main()
