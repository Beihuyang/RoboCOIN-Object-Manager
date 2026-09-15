"""VLM pre-review of semantic discovery nouns before SAM3 segmentation.

Nouns extracted from dataset names and annotation prose occasionally contain
verbs, abstract words or scene descriptions.  ``filter_prompts`` sends the
candidate noun list to the configured VLM (OpenAI-compatible API) and keeps
only nouns the model accepts as physical, visually detectable objects.  Every
verdict is cached in ``objects/prompt_noun_review_cache.json`` keyed by the
noun plus a fingerprint of the backend and review version, so each noun is
reviewed once across all datasets.

The review fails open: when the API is unavailable or a reply cannot be
parsed, the original prompt list is returned unchanged so SAM3 discovery is
never blocked on the collector workstation.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path

from vlm_backend import DEFAULT_MODEL, openai_backend_from_env

NOUN_REVIEW_VERSION = "vlm_noun_review_v1"
CACHE_PATH = Path(__file__).resolve().parent / "objects" / "prompt_noun_review_cache.json"
MAX_NOUNS_PER_CALL = 40

_REVIEW_PROMPT = """You are filtering noun prompts for an open-vocabulary image segmentation model (SAM3). The prompts come from robot manipulation dataset names and annotations, and some are verbs, actions, abstract concepts, scene descriptions, robot parts, or attributes rather than physical objects.

Dataset name: {dataset_name}
Relevant task/scene annotation text:
{source_context}

For each numbered noun below, decide whether it can denote a distinct physical object or physical furnishing that can be grounded by an image segmentation model from its name.

Rules:
- Keep concrete physical objects and furnishings: cup, test tube, storage box, trash bag, sponge, marker pen, mixing bowl, table, shelf, cabinet.
- Drop words that are unambiguously only verbs/actions (move, pour), abstract/relational concepts (arrangement, task, section), robot anatomy (robot arm, gripper, hand), spatial regions or structural surfaces (floor, wall, corner), and pure attributes with no distinct object form (relative, closer, beauty).
- Use the dataset name and annotation text to disambiguate a noun such as can, iron, press, scale, or orange.
- Be conservative: if a noun can plausibly denote a visible physical object in this context, keep it. Never drop it merely because it may be large, fixed, or not manipulated.

Nouns:
{list}

Reply with JSON only: {{"review": [{{"noun": "...", "verdict": "keep" or "drop", "reason": "short English reason"}}]}} with exactly one entry per input noun, in the same order."""


def _cache_fingerprint() -> str:
    backend = os.environ.get("ROBOCOIN_VLM_BACKEND", "api")
    api_base = os.environ.get("VLM_API_BASE", "").strip().rstrip("/")
    model = os.environ.get("VLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return f"{NOUN_REVIEW_VERSION}|{backend}|{api_base}|{model}"


def _context_fingerprint(dataset_name: str, source_context: list[str]) -> str:
    payload = json.dumps(
        {"dataset": dataset_name.strip(), "context": source_context},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _cache_key(noun: str, context_fingerprint: str) -> str:
    return f"{noun}|{context_fingerprint}"


def _load_cache() -> dict:
    try:
        with CACHE_PATH.open() as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=CACHE_PATH.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(CACHE_PATH)


def _review_chunk(
    backend, nouns: list[str], dataset_name: str, source_context: list[str],
) -> dict[str, dict]:
    listing = "\n".join(
        f"{index}. {noun}" for index, noun in enumerate(nouns, start=1)
    )
    context_text = "\n".join(f"- {text}" for text in source_context) or "- No annotation text available"
    prompt = _REVIEW_PROMPT.format(
        dataset_name=dataset_name or "unknown dataset",
        source_context=context_text,
        list=listing,
    )
    payload, _raw = backend.generate_json(
        None, prompt, max_new_tokens=128 * len(nouns)
    )
    entries = payload.get("review")
    if not isinstance(entries, list):
        raise ValueError("review field missing")
    verdicts: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        noun = str(entry.get("noun", "")).strip().lower()
        verdict = str(entry.get("verdict", "")).strip().lower()
        if noun in nouns and verdict in {"keep", "drop"}:
            verdicts[noun] = {
                "verdict": verdict,
                "reason": str(entry.get("reason", "")).strip()[:200],
            }
    missing = [noun for noun in nouns if noun not in verdicts]
    if missing:
        raise ValueError(f"missing verdicts for: {', '.join(missing[:5])}")
    return verdicts


def review_nouns(
    nouns: list[str], dataset_name: str = "", source_context: list[str] | None = None,
) -> tuple[dict[str, dict], str | None]:
    """Return ``{noun: {"verdict", "reason"}}`` for every noun, or fail open.

    The second return value is ``None`` on success or an error note when the
    review could not run and the caller should keep the original nouns.
    """
    unique = list(dict.fromkeys(
        noun.strip().lower() for noun in nouns if noun and noun.strip()
    ))
    if not unique:
        return {}, None
    if os.environ.get("ROBOCOIN_NOUN_REVIEW", "1").strip() == "0":
        return {}, "disabled by ROBOCOIN_NOUN_REVIEW=0"

    source_context = [" ".join(str(text).split())[:500] for text in (source_context or []) if str(text).strip()][:8]
    fingerprint = _cache_fingerprint()
    context_fingerprint = _context_fingerprint(dataset_name, source_context)
    cache = _load_cache()
    cached = {}
    pending = []
    for noun in unique:
        entry = cache.get(_cache_key(noun, context_fingerprint))
        if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint:
            cached[noun] = {
                "verdict": entry.get("verdict"),
                "reason": entry.get("reason", ""),
            }
        else:
            pending.append(noun)
    if not pending:
        return cached, None

    try:
        backend = openai_backend_from_env()
    except RuntimeError as exc:
        return cached, f"VLM unavailable, keeping original nouns ({exc})"
    verdicts = {}
    for start in range(0, len(pending), MAX_NOUNS_PER_CALL):
        chunk = pending[start:start + MAX_NOUNS_PER_CALL]
        try:
            verdicts.update(
                _review_chunk(backend, chunk, dataset_name, source_context)
            )
        except Exception as exc:  # fail open for this chunk
            return cached, f"VLM review failed, keeping original nouns ({exc})"
    for noun, verdict in verdicts.items():
        cache[_cache_key(noun, context_fingerprint)] = {
            "fingerprint": fingerprint,
            "noun": noun,
            "dataset": dataset_name,
            "context_fingerprint": context_fingerprint,
            **verdict,
        }
    _save_cache(cache)
    return {**cached, **verdicts}, None


def filter_prompts(
    prompts: list[str], dataset_name: str = "", source_context: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Apply the VLM review to a SAM3 discovery prompt list.

    The first entry (the generic ``object`` base prompt) is always kept.
    Returns the filtered prompt list and a report dict for logging.
    """
    prompts = list(prompts)
    if not prompts:
        return prompts, {"enabled": False, "reason": "empty prompt list"}
    base, nouns = prompts[0], prompts[1:]
    if not nouns:
        return prompts, {"enabled": False, "reason": "no semantic nouns"}
    verdicts, error = review_nouns(nouns, dataset_name, source_context)
    if error is not None:
        return prompts, {
            "enabled": False,
            "dataset": dataset_name,
            "reason": error,
        }
    kept = [base]
    dropped = []
    for noun in nouns:
        verdict = verdicts.get(noun.strip().lower(), {})
        if verdict.get("verdict") == "drop":
            dropped.append({"noun": noun, "reason": verdict.get("reason", "")})
        else:
            kept.append(noun)
    return kept, {
        "enabled": True,
        "dataset": dataset_name,
        "version": NOUN_REVIEW_VERSION,
        "kept": len(kept) - 1,
        "dropped": dropped,
    }
