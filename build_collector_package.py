#!/usr/bin/env python3
"""Build the minimal complete standalone collector-workstation directory.

Large data/model/result files are hard-linked locally so this staging directory
does not consume another 30+ GB.  Rsync/SSH reads them as normal files and the
collector workstation receives an independent physical copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR.parent / "RoboCOIN-Collector"
STAGING_MARKER = ".robocoin-collector-staging"
DEPLOY_EXCLUDES = (
    ".venv",
    "__pycache__",
    "*.pyc",
    "*.log",
    ".jobs",
    ".history",
    "_tracker_archive",
)

CODE_FILES = (
    "viewer.py",
    "stage1_track_select.py",
    "stage3_attribute.py",
    "stage4_dedup.py",
    "dedup_tree.py",
    "run_vlm_library_pipeline.py",
    "hardware_profiles.py",
    "project_paths.py",
    "migrate_paths.py",
    "super_resolution.py",
    "sam3_official_batch.py",
    "semantic_prompts.py",
    "noun_translations.py",
    "noun_translations_cache.json",
    "vlm_backend.py",
    "object_text_linker.py",
    "setup.sh",
    "start_collector.sh",
    "deploy_collector_ssh.sh",
    "collector.env.example",
    "requirements-collector.txt",
)
TREE_DIRS = ("sam3", "templates", "ontology")
RUNTIME_DIRS = ("RoboCOIN_datasets", "RoboCOIN_object_linked")
OBJECT_PATHS = (
    "tracks",
    "new_library_work",
    "new_library",
    "object_text_links",
    "object_id_registry.json",
)
MODEL_FILES = (
    "sam3_weights/sam3.pt",
    "sam3_weights/sam3.1_multiplex.pt",
    "models/realesrgan/RealESRGAN_x2plus.pth",
    "models/clip/ViT-L-14.pt",
)


def copy_file(source: Path, target: Path, *, hardlink: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if hardlink:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path, *, hardlink: bool) -> None:
    if target.exists():
        shutil.rmtree(target)
    copy_function = (lambda src, dst: copy_file(Path(src), Path(dst), hardlink=hardlink))
    shutil.copytree(
        source,
        target,
        copy_function=copy_function,
        ignore=shutil.ignore_patterns(*DEPLOY_EXCLUDES),
    )


def build(output: Path) -> None:
    marker = output / STAGING_MARKER
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise RuntimeError(f"拒绝覆盖非部署暂存目录：{output}")
        for child in output.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text("由 build_collector_package.py 自动生成，请勿直接用于审核。\n")
    for name in CODE_FILES:
        source = BASE_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"缺少部署文件：{source}")
        copy_file(source, output / name, hardlink=False)
    for name in TREE_DIRS:
        copy_tree(BASE_DIR / name, output / name, hardlink=False)
    for name in RUNTIME_DIRS:
        source = BASE_DIR / name
        if source.exists():
            copy_tree(source, output / name, hardlink=True)
    for name in OBJECT_PATHS:
        source = BASE_DIR / "objects" / name
        target = output / "objects" / name
        if source.is_dir():
            copy_tree(source, target, hardlink=True)
        elif source.is_file():
            copy_file(source, target, hardlink=True)
    for name in MODEL_FILES:
        source = BASE_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"缺少部署模型：{source}")
        copy_file(source, output / name, hardlink=True)

    docs = output / "docs"
    docs.mkdir(exist_ok=True)
    copy_file(
        BASE_DIR / "docs" / "DATA_COLLECTOR_GUIDE.md",
        docs / "DATA_COLLECTOR_GUIDE.md",
        hardlink=False,
    )
    images = BASE_DIR / "docs" / "images"
    if images.is_dir():
        copy_tree(images, docs / "images", hardlink=False)
    copy_file(BASE_DIR / "README.md", output / "README_技术说明.md", hardlink=False)
    (output / "README_数采员.md").write_text(
        "# RoboCOIN 数采程序\n\n"
        "首次安装请让技术人员执行 `./setup.sh`。配置 `collector.env` 后，"
        "日常只需双击或运行 `./start_collector.sh`，程序会检查环境并自动打开浏览器。\n\n"
        "部署目录在传输前使用硬链接节省本机空间，请不要直接在该暂存目录里开展审核。\n\n"
        "详细操作见 `docs/DATA_COLLECTOR_GUIDE.md`。\n",
        encoding="utf-8",
    )
    for executable in ("setup.sh", "start_collector.sh", "deploy_collector_ssh.sh"):
        (output / executable).chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    build(output)
    print(f"数采员完整部署目录已生成：{output}")


if __name__ == "__main__":
    main()
