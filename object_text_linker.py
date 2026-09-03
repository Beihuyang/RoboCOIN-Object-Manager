#!/usr/bin/env python3
"""Link RoboCOIN object mentions to persistent library IDs in mirrored JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from semantic_prompts import GENERIC_WORDS, KNOWN_COMPOUNDS, NON_TARGET_WORDS, _singular

BASE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = BASE_DIR / "RoboCOIN_datasets"
OUTPUT_ROOT = BASE_DIR / "RoboCOIN_object_linked"
LIBRARY_INDEX = BASE_DIR / "objects" / "new_library" / "index.json"
WORK_DIR = BASE_DIR / "objects" / "object_text_links"
MANIFEST_PATH = WORK_DIR / "manifest.json"

TEXT_FILES = {
    "meta/tasks.jsonl": "task",
    "meta/episodes.jsonl": "tasks",
    "annotations/scene_annotations.jsonl": "scene",
    "annotations/subtask_annotations.jsonl": "subtask",
    "annotations/subtasks.jsonl": "subtask",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
COLOR_WORDS = {
    "black", "blue", "brown", "gray", "green", "grey", "orange", "pink",
    "purple", "red", "white", "yellow",
}


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)


def _normalized_words(text: str) -> tuple[str, ...]:
    return tuple(_singular(match.group(0).lower()) for match in TOKEN_RE.finditer(text))


def _dataset_from_session(session_key: str) -> str:
    return str(session_key).replace("\\", "/").split("/", 1)[0]


def load_catalog(path: Path | None = None) -> dict:
    path = path or LIBRARY_INDEX
    if not path.is_file():
        raise FileNotFoundError(f"Object library not found: {path}")
    catalog = json.loads(path.read_text())
    invalid = [object_id for object_id in catalog if object_id.startswith("obj_")]
    if invalid:
        raise RuntimeError(
            "Object library still uses legacy obj_<hash> IDs; rebuild the library "
            "with the persistent ID registry first"
        )
    return catalog


def build_indexes(catalog: dict) -> tuple[dict, dict, dict]:
    aliases: dict[tuple[str, ...], set[str]] = defaultdict(set)
    global_by_category: dict[str, set[str]] = defaultdict(set)
    dataset_by_category: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    category_by_id = {}
    categories = set()
    for object_id, item in catalog.items():
        category = str(item.get("attributes", {}).get("category", "object")).strip()
        normalized = " ".join(_normalized_words(category)) or "object"
        category_by_id[object_id] = normalized
        if normalized in GENERIC_WORDS or normalized in NON_TARGET_WORDS:
            continue
        categories.add(normalized)
        global_by_category[normalized].add(object_id)
        for source in item.get("source_instances", []):
            dataset = _dataset_from_session(source.get("session_key", ""))
            if dataset:
                dataset_by_category[dataset][normalized].add(object_id)
    for category in categories:
        aliases[_normalized_words(category)].add(category)
    for compound in KNOWN_COMPOUNDS:
        words = _normalized_words(compound)
        head = words[-1] if words else ""
        for category in categories:
            category_words = _normalized_words(category)
            if category_words and category_words[-1] == head:
                aliases[words].add(category)
    return aliases, global_by_category, dataset_by_category


def _candidate_ids(
    dataset: str,
    categories: set[str],
    color: str | None,
    catalog: dict,
    global_by_category: dict,
    dataset_by_category: dict,
) -> tuple[list[str], str]:
    scoped = set()
    for category in categories:
        scoped.update(dataset_by_category.get(dataset, {}).get(category, set()))
    scope = "dataset"
    if not scoped:
        scope = "global"
        for category in categories:
            scoped.update(global_by_category.get(category, set()))
    if color:
        matching_color = {
            object_id for object_id in scoped
            if str(catalog[object_id].get("attributes", {}).get("color", "")).lower()
            in {color, "gray" if color == "grey" else color, "grey" if color == "gray" else color}
        }
        if matching_color:
            scoped = matching_color
    return sorted(scoped), scope


def extract_mentions(
    text: str,
    dataset: str,
    catalog: dict,
    aliases: dict,
    global_by_category: dict,
    dataset_by_category: dict,
) -> list[dict]:
    tokens = list(TOKEN_RE.finditer(text))
    normalized = [_singular(token.group(0).lower()) for token in tokens]
    max_alias = max((len(words) for words in aliases), default=1)
    mentions = []
    occupied = set()
    for size in range(max_alias, 0, -1):
        for start_index in range(len(tokens) - size + 1):
            indexes = set(range(start_index, start_index + size))
            if indexes & occupied:
                continue
            words = tuple(normalized[start_index:start_index + size])
            categories = aliases.get(words)
            if not categories:
                continue
            span_start = tokens[start_index].start()
            color = None
            if start_index > 0 and normalized[start_index - 1] in COLOR_WORDS:
                color = normalized[start_index - 1]
                span_start = tokens[start_index - 1].start()
                indexes.add(start_index - 1)
            candidates, candidate_scope = _candidate_ids(
                dataset, categories, color, catalog,
                global_by_category, dataset_by_category,
            )
            selected = candidates if len(candidates) == 1 and candidate_scope == "dataset" else []
            mentions.append({
                "text": text[span_start:tokens[start_index + size - 1].end()],
                "start": span_start,
                "end": tokens[start_index + size - 1].end(),
                "normalized_categories": sorted(categories),
                "color": color,
                "candidate_ids": candidates,
                "candidate_scope": candidate_scope,
                "selected_ids": selected,
                "status": "auto" if selected else ("pending" if candidates else "unresolved"),
            })
            occupied.update(indexes)
    return sorted(mentions, key=lambda item: item["start"])


def linked_text(source_text: str, mentions: list[dict]) -> str:
    result = source_text
    for mention in sorted(mentions, key=lambda item: item["start"], reverse=True):
        selected = mention.get("selected_ids", [])
        if not selected or mention.get("status") == "ignored":
            continue
        replacement = "[" + ",".join(selected) + "]"
        result = result[:mention["start"]] + replacement + result[mention["end"]:]
    return result


def _record_status(mentions: list[dict]) -> str:
    statuses = {mention.get("status") for mention in mentions}
    if not mentions:
        return "no_mentions"
    if statuses <= {"auto", "reviewed", "ignored"}:
        return "complete"
    if "pending" in statuses:
        return "needs_review"
    return "unresolved"


def _mention_id(dataset: str, mention: dict) -> str:
    """Identify one reusable dataset-level mention mapping, not one occurrence."""
    payload = "|".join((
        dataset,
        mention["text"].strip().lower(),
        ",".join(mention["normalized_categories"]),
        str(mention.get("color") or ""),
        ",".join(mention.get("candidate_ids", [])),
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def write_output_file(manifest: dict, relative_source: str) -> None:
    source = SOURCE_ROOT / relative_source
    target = OUTPUT_ROOT / relative_source
    field = TEXT_FILES["/".join(Path(relative_source).parts[1:])]
    dataset = Path(relative_source).parts[0]
    decisions = {mention["id"]: mention for mention in manifest.get("mentions", [])}
    catalog = load_catalog()
    aliases, global_by_category, dataset_by_category = build_indexes(catalog)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with source.open() as input_file, temp.open("w") as output_file:
        for line_index, line in enumerate(input_file):
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get(field)
            values = value if isinstance(value, list) else [value]
            linked_values = []
            all_mentions = []
            for value_index, text in enumerate(values):
                text = text if isinstance(text, str) else ""
                mentions = extract_mentions(
                    text, dataset, catalog, aliases,
                    global_by_category, dataset_by_category,
                )
                for mention in mentions:
                    mention_id = _mention_id(dataset, mention)
                    decision = decisions.get(mention_id)
                    if decision:
                        mention["selected_ids"] = decision.get("selected_ids", [])
                        mention["status"] = decision.get("status", mention["status"])
                linked_values.append(linked_text(text, mentions))
                all_mentions.extend(mentions)
            source_key = "source_" + field
            record[source_key] = value
            record[field] = linked_values if isinstance(value, list) else linked_values[0]
            record["object_ids"] = sorted({
                object_id for mention in all_mentions
                for object_id in mention.get("selected_ids", [])
            })
            record["object_mapping_status"] = _record_status(all_mentions)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(target)


def generate() -> dict:
    catalog = load_catalog()
    aliases, global_by_category, dataset_by_category = build_indexes(catalog)
    previous = {}
    if MANIFEST_PATH.is_file() and MANIFEST_PATH.stat().st_size < 200 * 1024 * 1024:
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text())
            if old_manifest.get("version") == 2:
                previous = {
                    item["id"]: item for item in old_manifest.get("mentions", [])
                }
        except (OSError, json.JSONDecodeError):
            previous = {}
    mention_groups = {}
    source_paths = []
    for dataset_dir in sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir()):
        for relative, field in TEXT_FILES.items():
            source = dataset_dir / relative
            if not source.is_file():
                continue
            relative_source = source.relative_to(SOURCE_ROOT).as_posix()
            source_paths.append(relative_source)
            target = OUTPUT_ROOT / relative_source
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            with source.open() as input_file, temp.open("w") as output_file:
                for line_index, line in enumerate(input_file):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    value = record.get(field)
                    values = value if isinstance(value, list) else [value]
                    linked_values = []
                    record_mentions = []
                    for value_index, text in enumerate(values):
                        if not isinstance(text, str) or not text.strip():
                            linked_values.append(text)
                            continue
                        found = extract_mentions(
                            text, dataset_dir.name, catalog, aliases,
                            global_by_category, dataset_by_category,
                        )
                        for mention in found:
                            mention_id = _mention_id(dataset_dir.name, mention)
                            prior = previous.get(mention_id)
                            if prior and prior.get("status") in {"reviewed", "ignored"}:
                                mention["selected_ids"] = prior.get("selected_ids", [])
                                mention["status"] = prior.get("status", mention["status"])
                            group = mention_groups.setdefault(mention_id, {
                                **mention,
                                "id": mention_id,
                                "dataset": dataset_dir.name,
                                "source_path": relative_source,
                                "line_index": line_index,
                                "field": field,
                                "value_index": value_index if isinstance(value, list) else None,
                                "source_text": text,
                                "occurrence_count": 0,
                            })
                            group["occurrence_count"] += 1
                            record_mentions.append(mention)
                        linked_values.append(linked_text(text, found))
                    record["source_" + field] = value
                    record[field] = linked_values if isinstance(value, list) else linked_values[0]
                    record["object_ids"] = sorted({
                        object_id for mention in record_mentions
                        for object_id in mention.get("selected_ids", [])
                    })
                    record["object_mapping_status"] = _record_status(record_mentions)
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            temp.replace(target)
    manifest = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "library_ids": sorted(catalog),
        "source_paths": sorted(source_paths),
        "mentions": sorted(mention_groups.values(), key=lambda item: (
            item["status"], item["dataset"], item["text"].lower()
        )),
    }
    _write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def save_decision(mention_id: str, selected_ids: list[str], ignored: bool = False) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    catalog = load_catalog()
    mention = next(
        (item for item in manifest.get("mentions", []) if item["id"] == mention_id),
        None,
    )
    if mention is None:
        raise KeyError(mention_id)
    selected_ids = list(dict.fromkeys(selected_ids))
    unknown = [object_id for object_id in selected_ids if object_id not in catalog]
    if unknown:
        raise ValueError(f"Unknown object IDs: {', '.join(unknown)}")
    mention["selected_ids"] = [] if ignored else selected_ids
    mention["status"] = "ignored" if ignored else ("reviewed" if selected_ids else "pending")
    mention["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(MANIFEST_PATH, manifest)
    dataset_prefix = mention["dataset"] + "/"
    for source_path in manifest.get("source_paths", []):
        if source_path.startswith(dataset_prefix):
            write_output_file(manifest, source_path)
    return mention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        manifest = generate()
        counts = defaultdict(int)
        for mention in manifest["mentions"]:
            counts[mention["status"]] += 1
        print(json.dumps({
            "files": len(manifest["source_paths"]),
            "mentions": len(manifest["mentions"]),
            "status": dict(counts),
            "output": str(OUTPUT_ROOT),
        }, ensure_ascii=False, indent=2))
    else:
        parser.error("pass --generate")


if __name__ == "__main__":
    main()
