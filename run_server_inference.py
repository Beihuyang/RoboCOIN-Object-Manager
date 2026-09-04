#!/usr/bin/env python3
"""A800 entry point containing only model-inference stages (never inpainting)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one GPU inference stage using the A800/80GB profile."
    )
    parser.add_argument(
        "stage", choices=("discover", "refine", "track", "attributes", "clip")
    )
    args, extra = parser.parse_known_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        parser.error("set CUDA_VISIBLE_DEVICES to your assigned card first")
    if "," in visible:
        parser.error("server mode expects exactly one assigned visible GPU")

    if args.stage in {"discover", "refine", "track"}:
        run([
            str(BASE_DIR / "stage1_track_select.py"),
            "--stage", args.stage,
            "--gpu", "0",
            "--hardware-profile", "a800",
            *extra,
        ])
    elif args.stage == "attributes":
        run([
            str(BASE_DIR / "stage3_attribute.py"),
            "--hardware-profile", "a800",
            *extra,
        ])
    else:
        run([
            str(BASE_DIR / "stage4_dedup.py"),
            "--hardware-profile", "a800",
            "--embeddings-only",
            *extra,
        ])


if __name__ == "__main__":
    main()
