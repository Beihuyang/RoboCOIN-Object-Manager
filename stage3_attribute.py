#!/usr/bin/env python3
"""Annotate the latest tracked objects with a VLM using the best-quality view.

Inference is pluggable: ``--vlm-backend local`` (default) uses a local
Qwen3-VL model; ``--vlm-backend api`` talks to an OpenAI-compatible
vision-language endpoint (``VLM_API_BASE``/``VLM_API_KEY``/``VLM_MODEL``, for
example ``glm5.3flash``).  All WordNet/ontology decisions stay local in both
modes, and the ``attributes.jsonl`` cache behaves identically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from hardware_profiles import PROFILES, default_profile_name, get_profile
from project_paths import portable_path, resolve_project_path
from vlm_backend import DEFAULT_MODEL as DEFAULT_API_VLM_MODEL, openai_backend_from_env

BASE_DIR = Path(__file__).resolve().parent
TRACKS_DIR = BASE_DIR / "objects" / "tracks"
WORK_DIR = BASE_DIR / "objects" / "new_library_work"
ATTR_CACHE = WORK_DIR / "attributes.jsonl"
ONTOLOGY_PATH = BASE_DIR / "ontology" / "attribute_taxonomies.json"
LOCAL_QWEN_DIR = BASE_DIR / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_QWEN_MODEL = (
    str(LOCAL_QWEN_DIR)
    if (LOCAL_QWEN_DIR / "model.safetensors").is_file()
    else "Qwen/Qwen3-VL-2B-Instruct"
)
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", DEFAULT_QWEN_MODEL)
PROMPT_VERSION = "tracked-object-chinese-review-display-v18"
WORDNET_SELECTION_VERSION = "name-lookup-numbered-hierarchy-v3"
PROGRESS_PREFIX = "@@PROGRESS "
WORDNET_ROOT = "entity.n.01"
WORDNET_PHYSICAL_ROOT = "physical_entity.n.01"
WORDNET_CONCRETE_ROOT = WORDNET_PHYSICAL_ROOT
WORDNET_MAX_DEPTH = 24
WORDNET_MAX_CANDIDATE_PATHS = 24
WORDNET_EXPANDED_CANDIDATE_PATHS = 64

# Active network VLM backend.  When set (api mode), every *_generate_json /
# *_generate_raw_batch call below is delegated to this backend instead of the
# local Qwen3-VL model; the local leaf functions remain the default path.
_VLM_API_BACKEND = None


def set_vlm_api_backend(backend) -> None:
    global _VLM_API_BACKEND
    _VLM_API_BACKEND = backend


def _api_backend():
    return _VLM_API_BACKEND

PROMPT = """The first image shows one tracked physical object isolated on black. The second image, when present, shows the same target highlighted in its original scene. Use the scene only to understand identity, attachment, and function; identify the highlighted target rather than nearby background objects. Produce a short common English object name for navigating controlled taxonomies. This hint is not the final category. Output ONLY:
{
  "object_name_hint": "short common English noun or noun phrase"
}
Do not mention the black background. No markdown and no explanation."""


def emit_progress(percent: float, message: str) -> None:
    payload = {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "message": message,
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def load_qwen():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {QWEN_MODEL_ID}…", flush=True)
    local = Path(QWEN_MODEL_ID).exists()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=local,
    )
    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID, local_files_only=local
    )
    # Decoder-only batched generation must continue from the last real token of
    # every sample rather than from right-padding tokens.
    processor.tokenizer.padding_side = "left"
    return model, processor, device


def _latest_manifests(backend: str) -> list[Path]:
    manifests = sorted(TRACKS_DIR.rglob(f"tracker_{backend}/manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No tracker_{backend} manifests found under {TRACKS_DIR}"
        )
    return manifests


def find_instances(backend: str) -> list[dict]:
    instances = []
    for manifest_path in _latest_manifests(backend):
        manifest = json.loads(manifest_path.read_text())
        tracker_dir = manifest_path.parent
        session_key = tracker_dir.parent.relative_to(TRACKS_DIR).as_posix()
        for item in manifest.get("objects", []):
            object_id = int(item["object_id"])
            object_dir = tracker_dir / f"object_{object_id:04d}"
            best = object_dir / "best_quality.jpg"
            if not best.is_file():
                print(f"Skipping incomplete tracked object: {object_dir}")
                continue
            instance_id = hashlib.sha1(
                f"{backend}:{session_key}:{object_id}".encode()
            ).hexdigest()[:16]
            instance = {
                "instance_id": instance_id,
                "tracker_backend": backend,
                "session_key": session_key,
                "object_id": object_id,
                "representative_quality_score": float(
                    item.get("best_quality", {}).get("quality_score", 0.0)
                ),
                "track_manifest": str(manifest_path.relative_to(BASE_DIR)),
                "best_quality_path": str(best.relative_to(BASE_DIR)),
                "canonical_source": str(best.relative_to(BASE_DIR)),
                "mask_prompt": str(item.get("prompt", "object")).strip() or "object",
                "matched_prompts": item.get("matched_prompts", []),
            }
            preview = object_dir / "best_quality.png"
            if preview.is_file():
                instance["preview_path"] = str(preview.relative_to(BASE_DIR))
            # New tracking results keep only the rule-selected representative and
            # its context. Fall back to legacy Top-K metadata when reading old runs.
            best_metadata = item.get("best_quality", {})
            context_path = resolve_project_path(
                best_metadata.get("context_path", object_dir / "best_quality_context.jpg")
            )
            mask_path = resolve_project_path(
                best_metadata.get("context_mask_path", object_dir / "best_quality_mask.png")
            )
            frame_path = resolve_project_path(
                best_metadata.get("scene_path") or best_metadata.get("frame_path", "")
            )
            if not context_path.is_file():
                selected_rank = item.get("quality_selection", {}).get("selected_rank")
                selected_candidate = next(
                    (
                        candidate for candidate in item.get("quality_candidates", [])
                        if candidate.get("rank") == selected_rank
                    ),
                    None,
                )
                if selected_candidate:
                    context_path = object_dir / selected_candidate.get("context", "")
                    mask_path = object_dir / selected_candidate.get("mask", "")
                    frame_path = resolve_project_path(
                        selected_candidate.get("frame_path", "")
                    )
            if context_path.is_file():
                instance["context_path"] = str(context_path.relative_to(BASE_DIR))
            if mask_path.is_file():
                instance["context_mask_path"] = str(mask_path.relative_to(BASE_DIR))
            if frame_path.is_file():
                instance["scene_frame_path"] = portable_path(frame_path)
            for key in ("context_bbox", "mask_bbox", "scene_image_representation"):
                if best_metadata.get(key) is not None:
                    instance[key] = best_metadata[key]
            instance["representative_image_representation"] = manifest.get(
                "representative_image_representation", "original_video_frames"
            )
            instances.append(instance)
    return instances


def vlm_cache_identity(vlm_backend: str) -> str:
    """Return the inference identity that owns an attribute-cache entry."""
    if vlm_backend == "api":
        api_base = os.environ.get("VLM_API_BASE", "").strip().rstrip("/")
        model = os.environ.get("VLM_MODEL", DEFAULT_API_VLM_MODEL).strip()
        return f"api:{api_base}:{model or DEFAULT_API_VLM_MODEL}"
    return f"local:{QWEN_MODEL_ID}"


def _fingerprint(instance: dict, vlm_identity: str) -> str:
    digest = hashlib.sha256(PROMPT_VERSION.encode())
    digest.update(vlm_identity.encode())
    digest.update(ONTOLOGY_PATH.read_bytes())
    digest.update(resolve_project_path(instance["best_quality_path"]).read_bytes())
    digest.update(str(instance.get("mask_prompt", "object")).encode())
    for key in ("context_path", "context_mask_path", "scene_frame_path"):
        if instance.get(key):
            path = resolve_project_path(instance[key])
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def _generate_json(images: Image.Image | list[Image.Image], prompt: str,
                   model, processor, device: str,
                   max_new_tokens: int) -> tuple[dict, str]:
    backend = _api_backend()
    if backend is not None:
        return backend.generate_json(images, prompt, max_new_tokens=max_new_tokens)
    image_list = images if isinstance(images, list) else [images]
    messages = [{
        "role": "user",
        "content": [
            *({"type": "image", "image": image} for image in image_list),
            {"type": "text", "text": prompt},
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=image_list, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    raw = processor.decode(
        generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return _extract_json(raw), raw


def _generate_raw_batch(
    image_groups: list[list[Image.Image]],
    prompt: str | list[str],
    model,
    processor,
    device: str,
    max_new_tokens: int,
) -> list[str]:
    """Generate one response per object while preserving every image's pixels."""
    backend = _api_backend()
    if backend is not None:
        return backend.generate_raw_batch(
            image_groups, prompt, max_new_tokens=max_new_tokens
        )
    texts = []
    flat_images = []
    prompts = [prompt] * len(image_groups) if isinstance(prompt, str) else prompt
    if len(prompts) != len(image_groups):
        raise ValueError("Batch prompts and image groups must have equal length")
    for images, item_prompt in zip(image_groups, prompts):
        messages = [{
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": item_prompt},
            ],
        }]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        flat_images.extend(images)

    inputs = processor(
        text=texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_only = generated[:, inputs.input_ids.shape[1]:]
    return [text.strip() for text in processor.batch_decode(
        generated_only, skip_special_tokens=True
    )]


def _wordnet():
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        return wordnet
    except LookupError as exc:
        raise RuntimeError(
            "NLTK WordNet data is not installed. Run: "
            "python -m nltk.downloader wordnet"
        ) from exc


def _synset_option(synset) -> dict:
    return {
        "id": synset.name(),
        "label": synset.lemmas()[0].name().replace("_", " "),
        "definition": synset.definition(),
    }


def _translated_synset_option(synset, translations: dict) -> dict:
    result = _synset_option(synset)
    translated = translations.get(synset.name(), {}) if isinstance(translations, dict) else {}
    if isinstance(translated, dict):
        label = str(translated.get("label", "")).strip()
        definition = str(translated.get("definition", "")).strip()
        if label:
            result["label_zh"] = label
        if definition:
            result["definition_zh"] = definition
    return result


@lru_cache(maxsize=1)
def _load_ontology() -> dict:
    if not ONTOLOGY_PATH.is_file():
        raise FileNotFoundError(f"Controlled ontology not found: {ONTOLOGY_PATH}")
    data = json.loads(ONTOLOGY_PATH.read_text())
    if not isinstance(data.get("attributes"), dict):
        raise ValueError(f"Invalid controlled ontology: {ONTOLOGY_PATH}")
    seen_ids = set()

    def validate_node(node: dict) -> None:
        if not all(isinstance(node.get(key), str) for key in ("id", "label", "definition")):
            raise ValueError(f"Invalid ontology node: {node}")
        if node["id"] in seen_ids:
            raise ValueError(f"Duplicate ontology node id: {node['id']}")
        seen_ids.add(node["id"])
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ValueError(f"Invalid children for ontology node: {node['id']}")
        for child in children:
            validate_node(child)

    for root in data["attributes"].values():
        validate_node(root)
    return data


def _taxonomy_option(node: dict) -> dict:
    return {
        key: node[key] for key in ("id", "label", "definition") if key in node
    }


def _choose_taxonomy_child(
    images: list[Image.Image], taxonomy_name: str, current: dict,
    visual_context: dict, model, processor, device: str,
) -> tuple[dict | None, dict]:
    children = current.get("children", [])
    if not children:
        return None, {}
    options = [_taxonomy_option(child) for child in children]
    prompt = f"""Classify the tracked object within the controlled {taxonomy_name} taxonomy.
The first image is the isolated tracked object. If a second image is present, it shows the object highlighted in scene context.
Current node:
{json.dumps(_taxonomy_option(current), ensure_ascii=False)}
Navigation hint:
{json.dumps(visual_context, ensure_ascii=False)}

Choose exactly one DIRECT child ID from the list. Choose STOP only if the current node is already the most specific visually supportable description. At the taxonomy root, prefer the explicit unknown child rather than STOP when evidence is insufficient. Never invent an ID or skip a level.
Direct children:
{json.dumps(options, ensure_ascii=False)}

Output ONLY JSON:
{{"decision": "a listed child id or STOP", "reason": "short visual reason"}}"""
    try:
        answer, raw = _generate_json(
            images, prompt, model, processor, device, max_new_tokens=96
        )
        decision = str(answer.get("decision", "STOP")).strip()
    except Exception as exc:
        raw = f"{type(exc).__name__}: {exc}"
        decision = "STOP"
    child_by_id = {child["id"]: child for child in children}
    step = {
        "node": current["id"],
        "decision": decision,
        "raw_output": raw,
    }
    return child_by_id.get(decision), step


def _hint_synsets(wordnet, object_name_hint: str) -> list:
    hint_words = object_name_hint.strip().lower().replace("-", " ").split()
    hint_keys = []
    if hint_words:
        hint_keys.append("_".join(hint_words))
        hint_keys.extend(reversed(hint_words))
    hint_synsets = []
    for hint_key in dict.fromkeys(hint_keys):
        for synset in wordnet.synsets(hint_key, pos=wordnet.NOUN):
            if synset not in hint_synsets:
                hint_synsets.append(synset)
    return hint_synsets


def _wordnet_candidate_paths(
    wordnet, object_name_hint: str, limit: int = WORDNET_MAX_CANDIDATE_PATHS,
) -> list[list]:
    """Build a compact set of complete WordNet paths related to the name hint."""
    root = wordnet.synset(WORDNET_ROOT)
    concrete_root = wordnet.synset(WORDNET_CONCRETE_ROOT)
    paths = []
    seen = set()
    for sense in _hint_synsets(wordnet, object_name_hint):
        for source_path in sense.hypernym_paths():
            if root not in source_path or concrete_root not in source_path:
                continue
            path = source_path[source_path.index(root):][:WORDNET_MAX_DEPTH]
            key = tuple(node.name() for node in path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
            if len(paths) >= limit:
                return paths
    return paths


def _accept_wordnet_path(candidate_paths: list[list], proposed_ids) -> list:
    """Keep only the valid offered prefix returned by the single VLM call."""
    if not candidate_paths:
        return []
    if not isinstance(proposed_ids, list):
        proposed_ids = []
    proposed_ids = [str(value).strip() for value in proposed_ids]
    root = candidate_paths[0][0]
    accepted = [root]
    offset = 1 if proposed_ids and proposed_ids[0] == root.name() else 0
    for proposed_id in proposed_ids[offset:]:
        prefix = [node.name() for node in accepted]
        allowed = {
            path[len(accepted)].name(): path[len(accepted)]
            for path in candidate_paths
            if len(path) > len(accepted)
            and [node.name() for node in path[:len(accepted)]] == prefix
        }
        chosen = allowed.get(proposed_id)
        if chosen is None:
            break
        accepted.append(chosen)
    return accepted


def _resolve_single_pass_wordnet_path(wordnet, answer: dict) -> list:
    """Normalize the VLM leaf/path decision to one official WordNet path."""
    from nltk.corpus.reader.wordnet import WordNetError

    proposed = answer.get("category_path", [])
    proposed = [str(value).strip() for value in proposed] if isinstance(proposed, list) else []
    leaf_id = str(answer.get("category_synset", "")).strip()
    if not leaf_id and proposed:
        leaf_id = proposed[-1]
    leaf = None
    if leaf_id:
        try:
            leaf = wordnet.synset(leaf_id)
        except (ValueError, LookupError, WordNetError):
            leaf = None
    if leaf is None:
        hint = str(answer.get("object_name_hint", ""))
        senses = _hint_synsets(wordnet, hint)
        leaf = senses[0] if senses else wordnet.synset(WORDNET_ROOT)
    root = wordnet.synset(WORDNET_ROOT)
    concrete_root = wordnet.synset(WORDNET_CONCRETE_ROOT)
    paths = []
    for source_path in leaf.hypernym_paths():
        if root in source_path and concrete_root in source_path:
            paths.append(source_path[source_path.index(root):][:WORDNET_MAX_DEPTH])
    if not paths:
        return _concrete_fallback_path(wordnet)

    def agreement(path: list) -> tuple[int, int]:
        names = [node.name() for node in path]
        matched = 0
        for left, right in zip(names, proposed):
            if left != right:
                break
            matched += 1
        return matched, -len(path)

    return max(paths, key=agreement)


def _concrete_fallback_path(wordnet) -> list:
    """Return the canonical entity > physical entity path."""
    root = wordnet.synset(WORDNET_ROOT)
    concrete_root = wordnet.synset(WORDNET_CONCRETE_ROOT)
    paths = [
        path[path.index(root):path.index(concrete_root) + 1]
        for path in concrete_root.hypernym_paths()
        if root in path
    ]
    return min(paths, key=len) if paths else [concrete_root]


def _is_physical_synset(wordnet, synset) -> bool:
    physical = wordnet.synset(WORDNET_PHYSICAL_ROOT)
    return synset == physical or any(
        physical in path for path in synset.hypernym_paths()
    )


def _canonical_wordnet_path(wordnet, synset) -> list:
    root = wordnet.synset(WORDNET_ROOT)
    physical = wordnet.synset(WORDNET_PHYSICAL_ROOT)
    paths = [
        path[path.index(root):]
        for path in synset.hypernym_paths()
        if root in path and physical in path
    ]
    return min(paths, key=lambda path: (len(path), [node.name() for node in path]))


def _physical_name_synsets(wordnet, object_name: str) -> tuple[list, list[str]]:
    """Look up the complete phrase first, then its head noun if necessary."""
    words = object_name.strip().lower().replace("-", " ").split()
    lookup_terms = ["_".join(words)] if words else []
    if len(words) > 1:
        lookup_terms.append(words[-1])
    for term in lookup_terms:
        matches = []
        for synset in wordnet.synsets(term, pos=wordnet.NOUN):
            if _is_physical_synset(wordnet, synset) and synset not in matches:
                matches.append(synset)
        if matches:
            return matches, lookup_terms[:lookup_terms.index(term) + 1]
    return [], lookup_terms


def _number_choice(value, option_count: int, *, allow_stop: bool) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        choice = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= choice <= option_count:
        return choice - 1
    if allow_stop and choice == option_count + 1:
        return -1
    return None


def _choose_wordnet_sense(
    images: list[Image.Image], object_name: str, senses: list,
    model, processor, device: str,
) -> tuple[object | None, dict]:
    options = [
        {"number": index, "id": synset.name(), "name": synset.lemmas()[0].name().replace("_", " "),
         "definition": synset.definition()}
        for index, synset in enumerate(senses, 1)
    ]
    prompt = f"""Choose the WordNet meaning that matches the pictured physical object named {json.dumps(object_name)}.
Candidates:
{json.dumps(options, ensure_ascii=False)}
Return ONLY JSON: {{"choice": 1}}
The choice must be one listed number. Do not default to the first sense."""
    try:
        answer, raw = _generate_json(images, prompt, model, processor, device, 32)
        selected = _number_choice(answer.get("choice"), len(senses), allow_stop=False)
    except Exception as exc:
        raw, selected = f"{type(exc).__name__}: {exc}", None
    return (senses[selected] if selected is not None else None), {
        "method": "numbered_sense_choice", "options": options,
        "choice": None if selected is None else selected + 1, "raw_output": raw,
    }


def _hierarchical_wordnet_choice(
    images: list[Image.Image], model, processor, device: str,
) -> tuple[list, list[dict], int]:
    """Navigate direct physical descendants using numbered choices only."""
    wordnet = _wordnet()
    current = wordnet.synset(WORDNET_ROOT)
    path = [current]
    steps = []
    calls = 0
    for _depth in range(WORDNET_MAX_DEPTH - 1):
        children = sorted(
            (child for child in current.hyponyms() if _is_physical_synset(wordnet, child)),
            key=lambda node: node.name(),
        )
        if not children:
            break
        options = [
            {"number": index, "id": child.name(), "name": child.lemmas()[0].name().replace("_", " "),
             "definition": child.definition()}
            for index, child in enumerate(children, 1)
        ]
        stop_number = len(options) + 1
        prompt = f"""Classify the pictured physical object one WordNet level at a time.
Current node: {current.name()} — {current.definition()}
Direct-child candidates:
{json.dumps(options, ensure_ascii=False)}
{stop_number}. STOP — the image does not support a more specific choice
Return ONLY JSON: {{"choice": 1}}
Return one listed number only. Choose STOP when visual evidence is insufficient."""
        calls += 1
        try:
            answer, raw = _generate_json(images, prompt, model, processor, device, 32)
            selected = _number_choice(answer.get("choice"), len(children), allow_stop=True)
        except Exception as exc:
            raw, selected = f"{type(exc).__name__}: {exc}", None
        steps.append({
            "method": "direct_child_number", "current": current.name(),
            "options": options, "stop_number": stop_number,
            "choice": stop_number if selected == -1 else (
                None if selected is None else selected + 1
            ), "raw_output": raw,
        })
        if selected is None or selected == -1:
            break
        current = children[selected]
        path.append(current)
    return path, steps, calls


def _resolve_name_to_wordnet(
    images: list[Image.Image], object_name: str, model, processor, device: str,
) -> tuple[list, list[dict], int, dict | None]:
    wordnet = _wordnet()
    senses, lookup_terms = _physical_name_synsets(wordnet, object_name)
    if len(senses) == 1:
        path = _canonical_wordnet_path(wordnet, senses[0])
        return path, [{
            "method": "unique_local_name_match", "lookup_terms": lookup_terms,
            "selected_synset": senses[0].name(),
            "accepted_path": [node.name() for node in path],
        }], 0, None
    if len(senses) > 1:
        selected, step = _choose_wordnet_sense(
            images, object_name, senses, model, processor, device
        )
        if selected is not None:
            path = _canonical_wordnet_path(wordnet, selected)
            step["accepted_path"] = [node.name() for node in path]
            return path, [step], 1, None
    path, steps, calls = _hierarchical_wordnet_choice(
        images, model, processor, device
    )
    warning = None if len(path) > 1 else {
        "type": "hierarchy_stopped_at_root", "object_name_hint": object_name,
        "fallback_synset": path[-1].name(),
    }
    return path, [{"method": "name_lookup_failed", "lookup_terms": lookup_terms}, *steps], calls, warning


def _validated_requested_wordnet_path(wordnet, answer: dict) -> list | None:
    """Return an official local path only when the requested synset is valid."""
    from nltk.corpus.reader.wordnet import WordNetError

    requested = str(answer.get("category_synset", "")).strip()
    if not requested:
        return None
    try:
        wordnet.synset(requested)
    except (ValueError, LookupError, WordNetError):
        return None
    path = _resolve_single_pass_wordnet_path(wordnet, answer)
    return path if path and path[-1].name() == requested else None


def _wordnet_candidate_prompt(answer: dict, candidate_paths: list[list]) -> str:
    offered = []
    for candidate_id, path in enumerate(candidate_paths):
        leaf = path[-1]
        offered.append({
            "candidate_id": candidate_id,
            "synset": leaf.name(),
            "label": leaf.lemmas()[0].name().replace("_", " "),
            "definition": leaf.definition(),
            "path": " > ".join(
                node.lemmas()[0].name().replace("_", " ") for node in path
            ),
        })
    return f"""Select the best WordNet category for the highlighted tracked object.
The first image is the isolated object. The second image, when present, shows the same target highlighted in scene context. Identify only the highlighted object.

The earlier object name is only a retrieval hint:
{json.dumps(str(answer.get('object_name_hint', '')).strip(), ensure_ascii=False)}

Choose exactly one candidate_id from this locally validated list. Prefer a broader candidate when the image cannot support a narrow sense. Do not output a synset name or a path.
Every offered candidate is under object.n.01; abstract concepts are intentionally unavailable.
Candidates:
{json.dumps(offered, ensure_ascii=False)}

Output ONLY JSON:
{{"candidate_id": 0}}
No markdown or explanation."""


def _parse_candidate_id(answer: dict, candidate_count: int) -> int | None:
    value = answer.get("candidate_id")
    if isinstance(value, bool):
        return None
    try:
        candidate_id = int(value)
    except (TypeError, ValueError):
        return None
    return candidate_id if 0 <= candidate_id < candidate_count else None


def _resolve_wordnet_category(
    images: list[Image.Image], answer: dict, model, processor, device: str,
) -> tuple[list, list[dict], int, dict | None]:
    """Validate the fast result, then use candidate IDs only for failures."""
    wordnet = _wordnet()
    requested_synset = str(answer.get("category_synset", "")).strip()
    object_name_hint = str(answer.get("object_name_hint", "")).strip()
    validated = _validated_requested_wordnet_path(wordnet, answer)
    if validated is not None:
        return validated, [{
            "method": "validated_synset",
            "requested_synset": requested_synset,
            "accepted_path": [node.name() for node in validated],
        }], 0, None

    retrieval_hint = object_name_hint
    if not retrieval_hint and requested_synset:
        retrieval_hint = requested_synset.split(".", 1)[0].replace("_", " ")
    candidate_limits = [
        WORDNET_MAX_CANDIDATE_PATHS,
        WORDNET_EXPANDED_CANDIDATE_PATHS,
    ]
    steps = []
    vlm_calls = 0
    for attempt, limit in enumerate(candidate_limits, start=1):
        candidate_paths = _wordnet_candidate_paths(
            wordnet, retrieval_hint, limit=limit
        )
        if not candidate_paths:
            continue
        vlm_calls += 1
        try:
            selected, raw = _generate_json(
                images,
                _wordnet_candidate_prompt(answer, candidate_paths),
                model,
                processor,
                device,
                max_new_tokens=48,
            )
            candidate_id = _parse_candidate_id(selected, len(candidate_paths))
        except Exception as exc:
            raw = f"{type(exc).__name__}: {exc}"
            candidate_id = None
        step = {
            "method": "candidate_id",
            "attempt": attempt,
            "candidate_limit": limit,
            "candidate_count": len(candidate_paths),
            "candidate_id": candidate_id,
            "raw_output": raw,
        }
        steps.append(step)
        if candidate_id is not None:
            accepted = candidate_paths[candidate_id]
            step["accepted_path"] = [node.name() for node in accepted]
            return accepted, steps, vlm_calls, None

    root = wordnet.synset(WORDNET_CONCRETE_ROOT)
    warning = {
        "type": "candidate_selection_failed",
        "requested_synset": requested_synset or "(missing)",
        "object_name_hint": object_name_hint,
        "fallback_synset": root.name(),
    }
    steps.append({
        "method": "safe_root_fallback",
        "accepted_path": [node.name() for node in _concrete_fallback_path(wordnet)],
    })
    return _concrete_fallback_path(wordnet), steps, vlm_calls, warning


def _accept_controlled_attribute_path(root: dict, proposed_ids) -> list[dict]:
    """Keep a valid direct-child prefix from one controlled attribute tree."""
    if not isinstance(proposed_ids, list):
        proposed_ids = []
    proposed_ids = [str(value).strip() for value in proposed_ids]
    accepted = [root]
    offset = 1 if proposed_ids and proposed_ids[0] == root["id"] else 0
    current = root
    for proposed_id in proposed_ids[offset:]:
        child = next(
            (item for item in current.get("children", []) if item["id"] == proposed_id),
            None,
        )
        if child is None:
            break
        accepted.append(child)
        current = child
    if len(accepted) == 1:
        unknown = next(
            (item for item in root.get("children", [])
             if item["id"] == f"{root['id'].split('.', 1)[0]}.unknown"),
            None,
        )
        if unknown is not None:
            accepted.append(unknown)
    return accepted


def _controlled_attribute_path(root: dict, proposed_id) -> list[dict]:
    """Resolve one controlled node ID to its complete local taxonomy path."""
    target_id = str(proposed_id or "").strip()

    def find(node: dict, path: list[dict]) -> list[dict] | None:
        current_path = [*path, node]
        if node["id"] == target_id:
            return current_path
        for child in node.get("children", []):
            found = find(child, current_path)
            if found is not None:
                return found
        return None

    found = find(root, []) if target_id else None
    if found is not None:
        return found
    return _accept_controlled_attribute_path(root, [])


KNOWN_COLOR_IDS = {
    "black": "color.black", "blue": "color.blue", "brown": "color.brown",
    "gray": "color.gray", "grey": "color.gray", "green": "color.green",
    "orange": "color.orange", "pink": "color.pink", "purple": "color.purple",
    "red": "color.red", "white": "color.white", "yellow": "color.yellow",
}
KNOWN_SIZE_IDS = {
    "tiny": "size.tiny", "small": "size.small", "medium": "size.medium",
    "large": "size.large", "huge": "size.huge",
}


def _known_mask_label(mask_prompt: str | None) -> dict:
    """Split an authoritative semantic mask label into identity and attributes."""
    label = " ".join(str(mask_prompt or "object").strip().lower().split())
    if not label or label in {"object", "interactive_box_points"}:
        return {}
    words = label.split()
    attribute_ids = {}
    identity_words = []
    for word in words:
        if word in KNOWN_COLOR_IDS and "color" not in attribute_ids:
            attribute_ids["color"] = KNOWN_COLOR_IDS[word]
        elif word in KNOWN_SIZE_IDS and "size" not in attribute_ids:
            attribute_ids["size"] = KNOWN_SIZE_IDS[word]
        else:
            identity_words.append(word)
    object_name = " ".join(identity_words).strip() or label
    return {
        "mask_prompt": label,
        "object_name": object_name,
        "attribute_ids": attribute_ids,
    }


def _apply_known_mask_label(answer: dict, known: dict) -> dict:
    if not known:
        return answer
    result = dict(answer)
    result["object_name_hint"] = known["object_name"]
    attribute_ids = result.get("attribute_ids", {})
    attribute_ids = dict(attribute_ids) if isinstance(attribute_ids, dict) else {}
    attribute_ids.update(known["attribute_ids"])
    result["attribute_ids"] = attribute_ids
    result["known_mask_label"] = known
    return result


def _object_name_prompt() -> str:
    return """Identify only the highlighted physical object. Return its ordinary English object name, using a short singular noun or noun phrase such as cup or water bottle. Do not return a WordNet ID, sense number, attributes, explanation, or markdown.
Output ONLY JSON:
{"object_name": "cup"}"""


def _all_attributes_prompt(
    mask_prompt: str | None = None,
    object_name: str = "object",
    category_path: list | None = None,
) -> str:
    taxonomy_roots = _load_ontology()["attributes"]
    prompt_trees = {
        name: root for name, root in taxonomy_roots.items()
    }
    known = _known_mask_label(mask_prompt)
    if known:
        identity_instruction = f"""The detector supplied this authoritative mask label:
{json.dumps(known, ensure_ascii=False)}
Attribute IDs supplied above are authoritative and must be copied unchanged. Infer only the attributes not supplied by the label."""
    else:
        identity_instruction = "No detector-provided attributes are available; infer all attributes from the images."
    category_nodes = [
        {"id": node.name(), "name": node.lemmas()[0].name().replace("_", " "),
         "definition": node.definition()}
        for node in (category_path or [])
    ]
    return f"""Classify the attributes of the highlighted tracked object in ONE response.
The first image is the object isolated on black. The second image, when present, is the original scene with the same target highlighted. Identify only the highlighted object. Use the scene to understand identity, attachment, and function, but never copy properties from the robot, table, container, or other background objects.

{identity_instruction}

Internal English object name: {json.dumps(object_name, ensure_ascii=False)}
Selected WordNet path to translate for the human interface:
{json.dumps(category_nodes, ensure_ascii=False)}

Return only one final-node ID for each controlled attribute. Every ID must come from its corresponding controlled tree below. The selected ID may be an intermediate node when that is the most specific visually supportable choice. Use the explicit unknown child when evidence is insufficient. The application reconstructs all complete controlled paths locally.

Controlled attribute trees:
{json.dumps(prompt_trees, ensure_ascii=False)}

Output ONLY JSON:
{{
  "attribute_ids": {{
    "color": "one controlled final-node id",
    "size": "one controlled final-node id",
    "material": "one controlled final-node id",
    "shape": "one controlled final-node id",
    "texture": "one controlled final-node id"
  }},
  "chinese_display": {{
    "object_name": "简短准确的中文物体名",
    "category_nodes": {{
      "each supplied WordNet id": {{"label": "中文节点名", "definition": "中文定义"}}
    }},
    "attribute_values": {{
      "color": "中文属性值",
      "size": "中文属性值",
      "material": "中文属性值",
      "shape": "中文属性值",
      "texture": "中文属性值"
    }}
  }}
}}
Translate every supplied WordNet node and every selected attribute value. No markdown, reasons, or explanation."""


def _interpret_all_attributes(
    answer: dict,
    raw: str,
    category_resolution: tuple[list, list[dict], int, dict | None] | None = None,
) -> tuple[dict, dict]:
    from nltk.corpus.reader.wordnet import WordNetError

    taxonomy_roots = _load_ontology()["attributes"]
    wordnet = _wordnet()
    requested_synset = str(answer.get("category_synset", "")).strip()
    if category_resolution is None:
        # Backward-compatible local interpretation for old direct callers. The
        # production path supplies the candidate-selection resolution below.
        wordnet_warning = None
        if requested_synset:
            try:
                wordnet.synset(requested_synset)
            except (ValueError, LookupError, WordNetError):
                wordnet_warning = {
                    "type": "invalid_vlm_synset",
                    "requested_synset": requested_synset,
                    "object_name_hint": str(
                        answer.get("object_name_hint", "")
                    ).strip(),
                }
        category_nodes = _resolve_single_pass_wordnet_path(wordnet, answer)
        category_steps = [{
            "method": "legacy_local_resolution",
            "decision_path": answer.get("category_path", []),
            "accepted_path": [node.name() for node in category_nodes],
            "raw_output": raw,
        }]
        extra_vlm_calls = 0
    else:
        category_nodes, category_steps, extra_vlm_calls, wordnet_warning = (
            category_resolution
        )
    proposed_attribute_ids = answer.get("attribute_ids", {})
    if not isinstance(proposed_attribute_ids, dict):
        proposed_attribute_ids = {}
    # Accept the previous verbose response shape so old callers/tests and any
    # in-flight model response remain valid during this cache-compatible change.
    proposed_attribute_paths = answer.get("attribute_paths", {})
    if not isinstance(proposed_attribute_paths, dict):
        proposed_attribute_paths = {}
    accepted_attribute_nodes = {}
    for name, root in taxonomy_roots.items():
        if name in proposed_attribute_ids:
            accepted_attribute_nodes[name] = _controlled_attribute_path(
                root, proposed_attribute_ids.get(name)
            )
        else:
            accepted_attribute_nodes[name] = _accept_controlled_attribute_path(
                root, proposed_attribute_paths.get(name, [])
            )
    category_node = category_nodes[-1]
    attributes = {
        "category": category_node.lemmas()[0].name().replace("_", " "),
        **{
            name: nodes[-1]["label"]
            for name, nodes in accepted_attribute_nodes.items()
        },
    }
    known_mask_label = answer.get("known_mask_label")
    if isinstance(known_mask_label, dict) and known_mask_label.get("object_name"):
        attributes["category"] = str(known_mask_label["object_name"])
    chinese_display = answer.get("chinese_display", {})
    chinese_display = chinese_display if isinstance(chinese_display, dict) else {}
    category_translations = chinese_display.get("category_nodes", {})
    attribute_translations = chinese_display.get("attribute_values", {})
    attribute_translations = (
        attribute_translations if isinstance(attribute_translations, dict) else {}
    )
    attributes_zh = {
        name: str(attribute_translations.get(name, "")).strip()
        for name in taxonomy_roots
    }
    metadata = {
        "raw_output": raw,
        "object_name_hint": str(answer.get("object_name_hint", "")).strip(),
        "known_mask_label": known_mask_label,
        "category_synset": category_node.name(),
        "category_path": [
            _translated_synset_option(node, category_translations)
            for node in category_nodes
        ],
        "display_name_zh": str(chinese_display.get("object_name", "")).strip(),
        "attributes_zh": attributes_zh,
        "category_steps": category_steps,
        "attribute_paths": {
            name: [_taxonomy_option(node) for node in nodes]
            for name, nodes in accepted_attribute_nodes.items()
        },
        "attribute_steps": [{
            "decision_ids": proposed_attribute_ids,
            "decision_paths": proposed_attribute_paths,
            "accepted_paths": {
                name: [node["id"] for node in nodes]
                for name, nodes in accepted_attribute_nodes.items()
            },
            "raw_output": raw,
        }],
        "vlm_calls": 1 + extra_vlm_calls,
        "wordnet_selection_version": WORDNET_SELECTION_VERSION,
        "library_eligible": True,
        "exclusion_reason": None,
    }
    if wordnet_warning is not None:
        wordnet_warning["fallback_synset"] = category_node.name()
        metadata["wordnet_warning"] = wordnet_warning
    return attributes, metadata


def classify_all_attributes_single_pass(
    images: list[Image.Image], model, processor, device: str,
    max_new_tokens: int = 384,
    mask_prompt: str = "object",
) -> tuple[dict, dict]:
    """Name the object, resolve WordNet locally, then classify attributes."""
    known = _known_mask_label(mask_prompt)
    name_raw = "detector-provided name"
    name_calls = 0
    if known:
        object_name = known["object_name"]
    else:
        name_answer, name_raw = _generate_json(
            images, _object_name_prompt(), model, processor, device, 48
        )
        object_name = str(name_answer.get("object_name", "object")).strip() or "object"
        name_calls = 1
    category_resolution = _resolve_name_to_wordnet(
        images, object_name, model, processor, device
    )
    attribute_answer, attribute_raw = _generate_json(
        images,
        _all_attributes_prompt(mask_prompt, object_name, category_resolution[0]),
        model,
        processor,
        device,
        max_new_tokens=max_new_tokens,
    )
    answer = _apply_known_mask_label({
        **attribute_answer, "object_name_hint": object_name,
    }, known)
    raw = json.dumps({"name": name_raw, "attributes": attribute_raw}, ensure_ascii=False)
    attributes, metadata = _interpret_all_attributes(answer, raw, category_resolution)
    metadata["vlm_calls"] += name_calls
    return attributes, metadata


def classify_all_attributes_batch(
    image_groups: list[list[Image.Image]], model, processor, device: str,
    mask_prompts: list[str] | None = None,
) -> list[tuple[dict, dict] | Exception]:
    """Classify several objects in one generation call."""
    mask_prompts = mask_prompts or ["object"] * len(image_groups)
    known_labels = [_known_mask_label(mask_prompt) for mask_prompt in mask_prompts]
    object_names: list[str | None] = [
        known.get("object_name") if known else None for known in known_labels
    ]
    name_raws = ["detector-provided name"] * len(image_groups)
    unknown_indexes = [index for index, name in enumerate(object_names) if name is None]
    if unknown_indexes:
        generated_names = _generate_raw_batch(
            [image_groups[index] for index in unknown_indexes],
            _object_name_prompt(), model, processor, device, max_new_tokens=48,
        )
        for index, raw in zip(unknown_indexes, generated_names):
            name_raws[index] = raw
            try:
                object_names[index] = str(
                    _extract_json(raw).get("object_name", "object")
                ).strip() or "object"
            except Exception:
                object_names[index] = "object"
    category_resolutions = [
        _resolve_name_to_wordnet(images, str(object_name), model, processor, device)
        for images, object_name in zip(image_groups, object_names)
    ]
    raws = _generate_raw_batch(
        image_groups,
        [
            _all_attributes_prompt(mask_prompt, str(object_names[index]), category_resolutions[index][0])
            for index, mask_prompt in enumerate(mask_prompts)
        ],
        model,
        processor,
        device,
        max_new_tokens=384,
    )
    results = []
    for index, (images, raw, mask_prompt) in enumerate(zip(image_groups, raws, mask_prompts)):
        try:
            answer = _apply_known_mask_label(
                {**_extract_json(raw), "object_name_hint": object_names[index]},
                known_labels[index],
            )
            category_resolution = category_resolutions[index]
            combined_raw = json.dumps(
                {"name": name_raws[index], "attributes": raw}, ensure_ascii=False
            )
            attributes, metadata = _interpret_all_attributes(
                answer, combined_raw, category_resolution
            )
            if index in unknown_indexes:
                metadata["vlm_calls"] += 1
            results.append((attributes, metadata))
        except Exception as exc:
            results.append(exc)
    return results


def classify_wordnet_category(
    images: list[Image.Image], visual_context: dict, model, processor, device: str
) -> tuple[str, str, list[dict], list[dict]]:
    """Ask the VLM once for a complete path, then retain its valid prefix."""
    wordnet = _wordnet()
    object_name_hint = str(visual_context.get("object_name_hint", ""))
    candidate_paths = _wordnet_candidate_paths(wordnet, object_name_hint)
    if not candidate_paths:
        fallback = _concrete_fallback_path(wordnet)
        root = fallback[-1]
        return (
            root.lemmas()[0].name().replace("_", " "),
            root.name(),
            [_synset_option(node) for node in fallback],
            [{"decision_path": [], "accepted_path": [node.name() for node in fallback],
              "raw_output": "No WordNet path matched the object name hint"}],
        )
    offered_paths = [
        {
            "path": [_synset_option(node) for node in path],
            "leaf_sense": path[-1].name(),
        }
        for path in candidate_paths
    ]
    prompt = f"""Classify the isolated physical object with ONE complete WordNet category path decision.
The first image is the isolated tracked object. If a second image is present, it shows the object highlighted in scene context. Use that context to distinguish a robot component attached to the robot from an independent task object.
An earlier visual description is only a navigation hint:
{json.dumps(visual_context, ensure_ascii=False)}

Choose one offered path and output its node IDs from entity.n.01 downward. You may stop at any node when the image cannot support a more specific descendant, so the answer may be a prefix of an offered path. Keep every intermediate node in order. Never combine different paths, skip a level, or invent an ID.

Offered complete paths:
{json.dumps(offered_paths, ensure_ascii=False)}

Output ONLY JSON in this form:
{{"path": ["entity.n.01", "next direct child id", "..."], "reason": "short visual reason"}}"""
    try:
        answer, raw = _generate_json(
            images, prompt, model, processor, device, max_new_tokens=512
        )
        proposed_ids = answer.get("path", [])
    except Exception as exc:
        raw = f"{type(exc).__name__}: {exc}"
        proposed_ids = []
    accepted = _accept_wordnet_path(candidate_paths, proposed_ids)
    if not accepted:
        accepted = [wordnet.synset(WORDNET_ROOT)]
    current = accepted[-1]
    path = [_synset_option(node) for node in accepted]
    steps = [{
        "decision_path": proposed_ids if isinstance(proposed_ids, list) else [],
        "accepted_path": [node.name() for node in accepted],
        "raw_output": raw,
    }]
    category = current.lemmas()[0].name().replace("_", " ")
    return category, current.name(), path, steps


def classify_controlled_attributes(
    images: list[Image.Image], visual_context: dict, model, processor, device: str,
) -> tuple[dict, dict, list[dict]]:
    taxonomy_roots = _load_ontology()["attributes"]
    current = {name: root for name, root in taxonomy_roots.items()}
    paths = {
        name: [_taxonomy_option(root)] for name, root in taxonomy_roots.items()
    }
    active = set(taxonomy_roots)
    rounds = []
    for _ in range(12):
        offered = {}
        for name in sorted(active):
            children = current[name].get("children", [])
            if children:
                offered[name] = {
                    "current": _taxonomy_option(current[name]),
                    "direct_children": [_taxonomy_option(child) for child in children],
                }
            else:
                active.discard(name)
        if not offered:
            break
        prompt = f"""Classify the highlighted tracked object in the controlled attribute trees.
The first image is the object isolated on black. The second image, when present, is the original scene with the same target highlighted. Use scene context to resolve ambiguity, but assign attributes ONLY from the highlighted target; never copy properties from the table, robot, container, or other background objects.
For EACH active attribute, choose exactly one listed DIRECT child ID. Choose STOP only when the current node is already the most specific visually supportable description. At a root, choose its explicit unknown child instead of STOP when evidence is insufficient. Never output free text, invent IDs, or skip levels.
For material, do not infer composition merely from the object name or typical use. Choose material.unknown when the image lacks distinctive visual evidence. For all attributes, stop at a broad parent instead of guessing an unsupported finer child.

Object navigation hint:
{json.dumps(visual_context, ensure_ascii=False)}

Active trees:
{json.dumps(offered, ensure_ascii=False)}

Output ONLY JSON:
{{"decisions": {{"color": "listed id or STOP", "size": "listed id or STOP", "material": "listed id or STOP", "shape": "listed id or STOP", "texture": "listed id or STOP"}}, "reasons": {{"attribute": "short visual reason"}}}}
Only include currently active attributes."""
        try:
            answer, raw = _generate_json(
                images, prompt, model, processor, device, max_new_tokens=256
            )
            decisions = answer.get("decisions", answer)
            if not isinstance(decisions, dict):
                decisions = {}
        except Exception as exc:
            raw = f"{type(exc).__name__}: {exc}"
            decisions = {}
        round_record = {"raw_output": raw, "decisions": {}}
        for name in list(active):
            if name not in offered:
                continue
            decision = str(decisions.get(name, "STOP")).strip()
            children = current[name].get("children", [])
            child_by_id = {child["id"]: child for child in children}
            chosen = child_by_id.get(decision)
            if chosen is None and len(paths[name]) == 1:
                chosen = child_by_id.get(f"{name}.unknown")
            round_record["decisions"][name] = decision
            if chosen is None:
                active.discard(name)
                continue
            current[name] = chosen
            paths[name].append(_taxonomy_option(chosen))
            if not chosen.get("children"):
                active.discard(name)
        rounds.append(round_record)

    values = {name: node["label"] for name, node in current.items()}
    return values, paths, rounds


def build_highlighted_scene_context(
    instance: dict, *, full_frame: bool = False,
) -> Image.Image | None:
    required = ("scene_frame_path", "context_path", "context_mask_path")
    if not all(instance.get(key) for key in required):
        return None
    paths = {}
    for key in required:
        paths[key] = resolve_project_path(instance[key])
    try:
        with Image.open(paths["scene_frame_path"]) as source:
            frame = np.asarray(source.convert("RGB")).copy()
        with Image.open(paths["context_path"]) as source:
            crop = np.asarray(source.convert("RGB")).copy()
        with Image.open(paths["context_mask_path"]) as source:
            crop_mask = np.asarray(source.convert("L")) > 127
    except OSError:
        return None
    if crop.shape[:2] != crop_mask.shape:
        return None
    exact_bbox = instance.get("context_bbox")
    if isinstance(exact_bbox, list) and len(exact_bbox) == 4:
        x1, y1, x2, y2 = (int(value) for value in exact_bbox)
        if (
            x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]
            or x2 <= x1 or y2 <= y1
            or (x2 - x1, y2 - y1) != (crop.shape[1], crop.shape[0])
        ):
            return None
    else:
        # Legacy runs saved the context/mask at the 2K representative scale but
        # pointed scene_frame_path at the original tracking frame. Match only after
        # bringing the crop back to that original scale. Matching the unscaled 2K
        # crop caused missing backgrounds and plausible-looking wrong placements.
        if instance.get("representative_image_representation") == "realesrgan_2k":
            from super_resolution import TARGET_LONG_EDGE

            scale = max(1.0, TARGET_LONG_EDGE / max(frame.shape[:2]))
            if scale > 1.0:
                legacy_size = (
                    max(1, round(crop.shape[1] / scale)),
                    max(1, round(crop.shape[0] / scale)),
                )
                crop = cv2.resize(crop, legacy_size, interpolation=cv2.INTER_AREA)
                crop_mask = cv2.resize(
                    crop_mask.astype(np.uint8),
                    legacy_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        if crop.shape[0] > frame.shape[0] or crop.shape[1] > frame.shape[1]:
            return None
        match = cv2.matchTemplate(frame, crop, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x1, y1) = cv2.minMaxLoc(match)
        if score < 0.30:
            return None
        x2, y2 = x1 + crop.shape[1], y1 + crop.shape[0]
    if full_frame:
        # Human review needs the complete scene. Keep the same target highlight,
        # but do not reuse the token-saving VLM crop for the review modal.
        roi_x1, roi_y1 = 0, 0
        roi_x2, roi_y2 = frame.shape[1], frame.shape[0]
    else:
        # Keep the target plus one half-target of scene context on every side. This
        # removes unrelated full-frame visual tokens without resizing the object or
        # imposing a maximum image resolution; a large target still yields a large ROI.
        pad_x = crop.shape[1] // 2
        pad_y = crop.shape[0] // 2
        roi_x1 = max(0, x1 - pad_x)
        roi_y1 = max(0, y1 - pad_y)
        roi_x2 = min(frame.shape[1], x2 + pad_x)
        roi_y2 = min(frame.shape[0], y2 + pad_y)
    highlighted = (
        frame[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32) * 0.35
    ).astype(np.uint8)
    local_x1, local_y1 = x1 - roi_x1, y1 - roi_y1
    local_x2, local_y2 = x2 - roi_x1, y2 - roi_y1
    region = highlighted[local_y1:local_y2, local_x1:local_x2]
    source_region = frame[y1:y2, x1:x2]
    region[crop_mask] = source_region[crop_mask]
    result = Image.fromarray(highlighted)
    draw = ImageDraw.Draw(result)
    line_width = max(3, min(highlighted.shape[:2]) // 120)
    mask_ys, mask_xs = np.where(crop_mask)
    if not len(mask_xs):
        return None
    mask_x1 = local_x1 + int(mask_xs.min())
    mask_y1 = local_y1 + int(mask_ys.min())
    mask_x2 = local_x1 + int(mask_xs.max() + 1)
    mask_y2 = local_y1 + int(mask_ys.max() + 1)
    draw.rectangle(
        (mask_x1, mask_y1, mask_x2 - 1, mask_y2 - 1),
        outline="#ff3030",
        width=line_width,
    )
    draw.text(
        (mask_x1, max(0, mask_y1 - 16)),
        "TRACKED REGION",
        fill="#ff3030",
    )
    return result


def _judgment_images(instance: dict) -> list[Image.Image]:
    with Image.open(resolve_project_path(instance["best_quality_path"])) as source:
        image = source.convert("RGB").copy()
    context_image = build_highlighted_scene_context(instance)
    return [image] + ([context_image] if context_image is not None else [])


def recognize(
    instance: dict,
    model,
    processor,
    device: str,
    max_new_tokens: int = 384,
) -> tuple[dict, dict]:
    return classify_all_attributes_single_pass(
        _judgment_images(instance),
        model,
        processor,
        device,
        max_new_tokens=max_new_tokens,
        mask_prompt=instance.get("mask_prompt", "object"),
    )


def recognize_batch(
    instances: list[dict], model, processor, device: str,
) -> list[tuple[dict, dict] | Exception]:
    return classify_all_attributes_batch(
        [_judgment_images(instance) for instance in instances],
        model,
        processor,
        device,
        [instance.get("mask_prompt", "object") for instance in instances],
    )


def _load_cache() -> dict[str, dict]:
    cached = {}
    if not ATTR_CACHE.is_file():
        return cached
    for line in ATTR_CACHE.read_text().splitlines():
        try:
            item = json.loads(line)
            cached[item["instance_id"]] = item
        except (KeyError, json.JSONDecodeError):
            continue
    return cached


def _write_current_cache(entries: dict[str, dict], instance_ids: set[str]) -> None:
    """Atomically discard annotations belonging to obsolete tracker outputs."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = ATTR_CACHE.with_suffix(".jsonl.tmp")
    with temp_path.open("w") as output:
        for instance_id in sorted(instance_ids):
            entry = entries.get(instance_id)
            if entry is not None:
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")
    temp_path.replace(ATTR_CACHE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("sam3",), default="sam3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hardware-profile", choices=tuple(PROFILES), default=default_profile_name()
    )
    parser.add_argument(
        "--vlm-backend",
        choices=("local", "api"),
        default=os.environ.get("ROBOCOIN_VLM_BACKEND", "local"),
        help="local = Qwen3-VL on this machine; api = OpenAI-compatible VLM "
             "(VLM_API_BASE/VLM_API_KEY/VLM_MODEL)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    profile = get_profile(args.hardware_profile)
    if args.batch_size is None:
        args.batch_size = profile.qwen_batch_size
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    all_instances = find_instances(args.tracker)
    current_ids = {instance["instance_id"] for instance in all_instances}
    instances = all_instances
    if args.limit:
        instances = instances[:args.limit]
    cache = _load_cache()
    vlm_identity = vlm_cache_identity(args.vlm_backend)
    pending = []
    for instance in instances:
        instance["fingerprint"] = _fingerprint(instance, vlm_identity)
        instance["vlm_backend"] = args.vlm_backend
        instance["vlm_identity"] = vlm_identity
        old = cache.get(instance["instance_id"])
        if (
            args.force
            or old is None
            or "error" in old
            or old.get("fingerprint") != instance["fingerprint"]
            or (
                old.get("wordnet_warning")
                and old.get("wordnet_selection_version")
                != WORDNET_SELECTION_VERSION
            )
        ):
            pending.append(instance)
    print(
        f"Tracked objects: {len(instances)}; cached: {len(instances) - len(pending)}; "
        f"pending: {len(pending)}",
        flush=True,
    )
    print(f"Hardware profile: {profile.name}; VLM backend: {args.vlm_backend}")
    emit_progress(
        0,
        f"属性标注准备完成：缓存 {len(instances) - len(pending)}，待处理 {len(pending)}",
    )
    if not pending:
        _write_current_cache(cache, current_ids)
        emit_progress(100, f"属性缓存已是最新：{len(instances)} 个物体")
        print(f"Attributes are current: {ATTR_CACHE}")
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    emit_progress(1, "正在加载 WordNet 和属性词库")
    _wordnet()
    if args.vlm_backend == "api":
        backend = openai_backend_from_env()
        set_vlm_api_backend(backend)
        model = processor = None
        device = "api"
        backend_label = f"VLM API（{backend.model}）"
    else:
        emit_progress(2, "正在加载本地 Qwen3-VL 属性标注模型")
        model, processor, device = load_qwen()
        backend_label = "Qwen3-VL"
    emit_progress(
        5,
        f"{backend_label} 已就绪，按批次 {args.batch_size} 标注 {len(pending)} 个物体",
    )
    started = time.time()
    progress = tqdm(total=len(pending), desc=f"VLM attributes ({args.vlm_backend})")
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        try:
            if len(batch) == 1:
                results: list[tuple[dict, dict] | Exception] = [
                    recognize(batch[0], model, processor, device)
                ]
            else:
                results = recognize_batch(batch, model, processor, device)
        except torch.cuda.OutOfMemoryError:
            # Large source images are intentionally not resized. If two of them
            # do not fit together, retain their original pixels and retry them
            # sequentially instead of failing the whole annotation run.
            torch.cuda.empty_cache()
            results = []
            for instance in batch:
                try:
                    results.append(recognize(instance, model, processor, device))
                except Exception as exc:
                    results.append(exc)
        except Exception:
            # Preserve per-object error isolation if a processor/model version
            # cannot batch a particular mixture of one- and two-image samples.
            results = []
            for instance in batch:
                try:
                    results.append(recognize(instance, model, processor, device))
                except Exception as exc:
                    results.append(exc)

        for instance, result in zip(batch, results):
            entry = dict(instance)
            if isinstance(result, json.JSONDecodeError):
                # The compact schema should normally finish well below the
                # 384-token batch limit. Give only malformed/truncated samples
                # a larger individual retry budget.
                try:
                    result = recognize(
                        instance,
                        model,
                        processor,
                        device,
                        max_new_tokens=768,
                    )
                except Exception as exc:
                    result = exc
            if isinstance(result, Exception):
                entry["error"] = f"{type(result).__name__}: {result}"
            else:
                attributes, metadata = result
                entry.update({"attributes": attributes, **metadata})
            cache[instance["instance_id"]] = entry
            progress.update(1)
        # Persist once per batch rather than rewriting the full JSONL per object.
        _write_current_cache(cache, current_ids)
        completed = min(batch_start + len(batch), len(pending))
        emit_progress(
            5 + 95 * completed / len(pending),
            f"VLM 批量联合标注 {completed}/{len(pending)}",
        )
    progress.close()
    _write_current_cache(cache, current_ids)
    print(f"Annotated {len(pending)} object(s) in {time.time() - started:.1f}s")
    print(f"Output: {ATTR_CACHE}")


if __name__ == "__main__":
    main()
