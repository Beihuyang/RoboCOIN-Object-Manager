#!/usr/bin/env python3
"""Build lossless human-review previews from existing context and mask files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
TRACKS_DIR = BASE_DIR / "objects" / "tracks"


def repair(force: bool = False) -> dict[str, int]:
    contexts = sorted(TRACKS_DIR.glob(
        "**/tracker_sam3/object_*/best_quality_context.jpg"
    ))
    counts = {"total": len(contexts), "created": 0, "reused": 0, "failed": 0}
    for context_path in tqdm(contexts, desc="Lossless tree previews"):
        object_dir = context_path.parent
        mask_path = object_dir / "best_quality_mask.png"
        output_path = object_dir / "best_quality.png"
        if output_path.is_file() and not force:
            counts["reused"] += 1
            continue
        try:
            with Image.open(context_path) as source:
                context = np.asarray(source.convert("RGB")).copy()
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L")) > 127
            if context.shape[:2] != mask.shape:
                raise ValueError("context and mask shapes differ")
            context[~mask] = 0
            Image.fromarray(context).save(output_path)
            counts["created"] += 1
        except (OSError, ValueError):
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    counts = repair(force=args.force)
    print(", ".join(f"{key}={value}" for key, value in counts.items()))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
