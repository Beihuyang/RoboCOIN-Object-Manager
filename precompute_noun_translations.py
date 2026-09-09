#!/usr/bin/env python3
"""Pre-translate English SAM3 review prompts into Chinese with a lightweight API.

``/review`` shows each English discovery prompt as a Chinese label.  The
built-in ``noun_translations.NOUN_ZH`` dictionary covers the common terms; this
script scans every review manifest for the *complete* set of English prompts,
sends the unknown ones to a lightweight OpenAI-compatible text model (default
``glm-4.7-flashx``, paid but measured ~1s per batch; ``glm-4.7-flash`` is a free
fallback) and writes ``noun_translations_cache.json``.

``prompt_zh`` consults that cache after ``NOUN_ZH``, so annotation pages and
offline review packages stay fully offline at runtime — only this precompute
step touches the network.

Usage:

    export VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
    export VLM_API_KEY=<key>
    python3 precompute_noun_translations.py          # 默认模型 glm-4.5-flash

Optional: --model, --source, --output, --batch-size, --limit.  The model id can
also be set with ``VLM_TRANSLATE_MODEL``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from noun_translations import NOUN_ZH, normalize_prompt
from vlm_backend import OpenAIChatBackend

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_TRACKS = BASE_DIR / "objects" / "tracks"
DEFAULT_OUTPUT = BASE_DIR / "noun_translations_cache.json"
# glm-4.5-flash (free) was measured at minutes per batch on the Zhipu platform
# and can hit free-tier limits.  Paid small models are ~50x faster with
# thinking disabled: glm-4.7-flashx (~1.1s/8 terms), glm-4.6 (~1.3s),
# glm-4.5-airx (~1.7s).  glm-4.7-flash is a free fallback (~3.4s/8 terms).
DEFAULT_MODEL = "glm-4.7-flashx"
TRANSLATION_PROMPT = (
    "Translate each English object noun phrase into one concise simplified "
    "Chinese noun. Keep every key exactly as given; do not add, remove, or "
    "reorder keys. Return ONLY a JSON object that maps each key to its "
    "Chinese translation. No markdown, no explanations.\n\nKeys:\n"
)


def collect_terms(tracks_root: Path) -> set[str]:
    """Return the normalized set of English prompts shown anywhere in review."""
    terms: set[str] = set()
    for manifest_path in sorted(tracks_root.rglob("initial_sam3_sr_2k/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        candidates = list(manifest.get("prompts") or [])
        for item in manifest.get("objects") or []:
            if isinstance(item, dict):
                if item.get("prompt"):
                    candidates.append(item["prompt"])
                for match in item.get("matched_prompts") or []:
                    if isinstance(match, dict) and match.get("prompt"):
                        candidates.append(match["prompt"])
        for value in candidates:
            term = normalize_prompt(value)
            if term and term not in {"object", "manual box", "sam3 box refined"}:
                terms.add(term)
    return terms


def read_existing(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {
        normalize_prompt(key): str(value).strip()
        for key, value in (data.items() if isinstance(data, dict) else [])
        if normalize_prompt(key) and str(value).strip()
    }


def merge_translations(existing: dict[str, str], additions: dict[str, str]) -> dict[str, str]:
    merged = {normalize_prompt(key): value.strip() for key, value in existing.items()}
    for key, value in additions.items():
        key = normalize_prompt(key)
        if key and isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    return dict(sorted(merged.items()))


def _translate_batch_once(backend, batch: list[str]) -> dict[str, str]:
    """Translate one batch; a single retry covers dropped/misspelled keys."""
    prompt = TRANSLATION_PROMPT + json.dumps(batch, ensure_ascii=False)
    answer, _ = backend.generate_json(
        [], prompt, max_new_tokens=max(256, len(batch) * 40),
    )
    if not isinstance(answer, dict):
        answer = {}
    translated: dict[str, str] = {}
    for key, value in answer.items():
        key = normalize_prompt(key)
        if key in batch and isinstance(value, str) and value.strip():
            translated[key] = value.strip()
    missing = [term for term in batch if term not in translated]
    if missing:
        retry_prompt = TRANSLATION_PROMPT + json.dumps(missing, ensure_ascii=False)
        try:
            retry_answer, _ = backend.generate_json(
                [], retry_prompt, max_new_tokens=max(256, len(missing) * 40),
            )
        except Exception:
            retry_answer = {}
        if isinstance(retry_answer, dict):
            for key, value in retry_answer.items():
                key = normalize_prompt(key)
                if key in missing and isinstance(value, str) and value.strip():
                    translated[key] = value.strip()
    print(
        f"  batch {len(batch)} 词 -> 新增 {len(translated)}，"
        f"仍缺 {sum(1 for t in batch if t not in translated)}",
        flush=True,
    )
    return translated


def translate_batch(backend, terms: list[str], batch_size: int,
                    concurrency: int = 1) -> dict[str, str]:
    """Translate every term, running batches in parallel up to concurrency."""
    batches = [
        terms[index:index + batch_size]
        for index in range(0, len(terms), batch_size)
    ]
    translated: dict[str, str] = {}
    if not batches:
        return translated
    if concurrency <= 1 or len(batches) == 1:
        for batch in batches:
            translated.update(_translate_batch_once(backend, batch))
        return translated
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as executor:
        futures = [executor.submit(_translate_batch_once, backend, batch)
                   for batch in batches]
        for future in futures:
            translated.update(future.result())
    return translated


def write_output(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_TRACKS,
                        help="objects/tracks 目录（默认本项目）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="输出缓存 json（默认 noun_translations_cache.json）")
    parser.add_argument("--model", default=os.environ.get(
        "VLM_TRANSLATE_MODEL", DEFAULT_MODEL), help="翻译模型 code")
    parser.add_argument("--api-base", default=None,
                        help="OpenAI 兼容端点；默认读 VLM_API_BASE")
    parser.add_argument("--api-key", default=None,
                        help="API key；默认读 VLM_API_KEY")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="每次请求翻译的词数（默认 50）")
    parser.add_argument("--concurrency", type=int,
                        default=int(os.environ.get("VLM_TRANSLATE_CONCURRENCY") or "4"),
                        help="并行翻译的批数（默认 4，付费档可调高）")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅处理前 N 个未知词（联调用）")
    args = parser.parse_args()

    api_base = (args.api_base or os.environ.get("VLM_API_BASE", "")).strip().rstrip("/")
    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
    if not api_base or not api_key:
        parser.error("需要 VLM_API_BASE 与 VLM_API_KEY（或 --api-base/--api-key）")

    backend = OpenAIChatBackend(
        api_base,
        api_key,
        model=args.model,
        temperature=0.0,
        reasoning_effort=None,
        thinking=os.environ.get("VLM_TRANSLATE_THINKING", "disabled") or None,
        token_scale=2.0,
    )
    terms = collect_terms(args.source)
    existing = read_existing(args.output)
    builtin = {normalize_prompt(key) for key in NOUN_ZH}
    unknown = sorted(
        term for term in terms
        if term not in builtin and term not in existing
    )
    print(f"扫描到 {len(terms)} 个去重词；内置词表+已有缓存已覆盖 "
          f"{len(terms) - len(unknown)}，待翻译 {len(unknown)}", flush=True)
    if args.limit:
        unknown = unknown[:args.limit]
    if not unknown:
        if existing:
            write_output(args.output, existing)
            print(f"无需翻译，已更新缓存：{args.output}")
        else:
            print("无需翻译且当前没有缓存文件，未写入。")
        return

    print(f"使用模型 {args.model} 翻译 {len(unknown)} 个词"
          f"（batch={args.batch_size}, concurrency={args.concurrency}）…", flush=True)
    try:
        additions = translate_batch(
            backend, unknown, args.batch_size, concurrency=args.concurrency,
        )
    except Exception as exc:
        print(f"翻译失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    merged = merge_translations(existing, additions)
    write_output(args.output, merged)
    print(f"已写入 {len(merged)} 条翻译缓存：{args.output}")
    if len(additions) < len(unknown):
        print(f"警告：{len(unknown) - len(additions)} 个词未获得翻译，请稍后重试或手工补入词表。", file=sys.stderr)


if __name__ == "__main__":
    main()
