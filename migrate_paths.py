#!/usr/bin/env python3
"""Convert absolute project paths in generated metadata to portable paths."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from project_paths import PROJECT_ROOT, portable_saved_string


DEFAULT_SCAN_ROOT = PROJECT_ROOT / "objects"


def convert(value):
    if isinstance(value, dict):
        return {key: convert(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert(item) for item in value]
    if isinstance(value, str):
        return portable_saved_string(value)
    return value


def write_atomic(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".migrating", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(text)
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def migrate_json(path: Path, apply: bool) -> tuple[bool, int]:
    original = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        lines = []
        changes = 0
        for line in original.splitlines():
            if not line.strip():
                lines.append(line)
                continue
            before = json.loads(line)
            after = convert(before)
            changes += before != after
            lines.append(json.dumps(after, ensure_ascii=False))
        updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    else:
        before = json.loads(original)
        after = convert(before)
        changes = int(before != after)
        updated = json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    changed = updated != original and changes > 0
    if changed and apply:
        write_atomic(path, updated)
    return changed, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument(
        "--apply", action="store_true", help="write changes (default: report only)"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted({*root.rglob("*.json"), *root.rglob("*.jsonl")})
    changed_files = 0
    invalid_files = []
    for path in paths:
        try:
            changed, _ = migrate_json(path, args.apply)
        except (OSError, json.JSONDecodeError) as exc:
            invalid_files.append((path, str(exc)))
            continue
        if changed:
            changed_files += 1
            print(("Updated" if args.apply else "Would update") + f": {path}")
    mode = "updated" if args.apply else "would be updated"
    print(f"Path migration: {changed_files} file(s) {mode}; {len(paths)} scanned.")
    if invalid_files:
        print(f"Skipped {len(invalid_files)} invalid JSON file(s).")


if __name__ == "__main__":
    main()
