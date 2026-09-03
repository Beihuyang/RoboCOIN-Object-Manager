#!/usr/bin/env python3
"""Download only RoboCOIN text files that can contain object mentions."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

ORG = "RoboCOIN"
BASE_DIR = Path(__file__).resolve().parent / "RoboCOIN_datasets"
OBJECT_TEXT_PATTERNS = [
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
    "annotations/scene_annotations.jsonl",
    "annotations/subtask_annotations.jsonl",
    "annotations/subtasks.jsonl",
]


def list_datasets() -> list[str]:
    api = HubApi()
    dataset_ids = []
    page = 1
    while True:
        result = api.list_repos(
            "dataset", owner=ORG, page_number=page, page_size=50
        )
        dataset_ids.extend(f"{ORG}/{repo.name}" for repo in result)
        if not result.has_next:
            return dataset_ids
        page += 1


def download_object_text(repo_id: str) -> tuple[str, str]:
    target = BASE_DIR / repo_id.split("/", 1)[1]
    last_error = ""
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=OBJECT_TEXT_PATTERNS,
                local_dir=str(target),
            )
            return repo_id, "done"
        except Exception as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(2 * attempt)
    return repo_id, f"failed: {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RoboCOIN text containing possible object mentions"
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = list_datasets()
    if args.limit:
        datasets = datasets[: args.limit]
    print(f"Found {len(datasets)} datasets", flush=True)
    print("Patterns:", *OBJECT_TEXT_PATTERNS, sep="\n  ", flush=True)
    if args.dry_run:
        return

    done = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_object_text, repo_id): repo_id
            for repo_id in datasets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            repo_id, status = future.result()
            if status == "done":
                done += 1
            else:
                failures.append((repo_id, status))
            if completed % 50 == 0 or status != "done":
                print(
                    f"progress {completed}/{len(datasets)} "
                    f"done={done} failed={len(failures)}",
                    flush=True,
                )
    print(f"COMPLETE done={done} failed={len(failures)}", flush=True)
    for repo_id, status in failures:
        print(f"FAILED {repo_id}: {status}", flush=True)


if __name__ == "__main__":
    main()
