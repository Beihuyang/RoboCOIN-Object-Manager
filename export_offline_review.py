#!/usr/bin/env python3
"""Export a self-contained offline mask-review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from project_paths import portable_path, resolve_project_path


BASE_DIR = Path(__file__).resolve().parent
TRACKS_DIR = BASE_DIR / "objects" / "tracks"
REVIEW_DIR_NAME = "initial_sam3_sr_2k"
CODE_FILES = (
    "viewer.py", "stage1_track_select.py", "hardware_profiles.py",
    "project_paths.py", "semantic_prompts.py", "noun_translations.py",
    "super_resolution.py", "sam3_official_batch.py",
)


def available_sessions() -> dict[str, Path]:
    result = {}
    for manifest in TRACKS_DIR.glob(f"**/{REVIEW_DIR_NAME}/manifest.json"):
        directory = manifest.parent
        result[directory.relative_to(TRACKS_DIR).as_posix()] = directory
    return result


def write_baseline_file(source_manifest: Path, target_initial_dir: Path) -> Path:
    """Record the server manifest digest at export time inside the package.

    ``import_offline_review.py`` uses this file to detect whether the server
    manifest changed *after* the package was exported, avoiding silent
    overwrites of server-side edits.
    """
    manifest_bytes = source_manifest.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError:
        manifest = {}
    baseline = {
        "version": 1,
        "manifest_sha1": hashlib.sha1(manifest_bytes).hexdigest(),
        "revision": manifest.get("revision"),
        "objects": len(manifest.get("objects", [])),
        "discovery_frames": len(manifest.get("discovery_frames", [])),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "server_manifest": source_manifest.as_posix(),
    }
    baseline_path = target_initial_dir / ".export_baseline.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))
    return baseline_path


def copy_review_session(key: str, source: Path, package: Path) -> None:
    target = package / "objects" / "tracks" / key
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source, target,
        ignore=shutil.ignore_patterns(
            ".history", ".jobs", ".editing-*", ".keyframe-*"
        ),
    )
    write_baseline_file(source / "manifest.json", target)
    frame_source = source.parent / "first_frame_sr_2k"
    frame_target = target.parent / "first_frame_sr_2k"
    shutil.copytree(frame_source, frame_target, dirs_exist_ok=True)
    extraction_path = frame_target / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    video_source = resolve_project_path(extraction["source"])
    if not video_source.is_file():
        raise FileNotFoundError(f"审核视频不存在：{video_source}")
    video_target = target.parent / "review_video" / video_source.name
    video_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_source, video_target)
    extraction["source"] = portable_path(video_target, package)
    extraction_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False))


def write_launch_files(package: Path) -> None:
    (package / "requirements-review.txt").write_text(
        "fastapi\nuvicorn\npydantic\nnumpy<2\npillow\nopencv-python\ntqdm\n"
        "timm>=1.0.17\nftfy==6.1.1\nregex\niopath>=0.1.10\n"
        "typing_extensions\npycocotools\nbasicsr\nrealesrgan\nnltk\n"
    )
    launcher = package / "start_review.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -e\ncd \"$(dirname \"$0\")\"\n"
        "export ROBOCOIN_OFFLINE_REVIEW=1\n"
        "if [ -x .venv/bin/python ]; then\n"
        "  python_cmd=.venv/bin/python\n"
        "elif [ -n \"${VIRTUAL_ENV:-}\" ] && [ -x \"$VIRTUAL_ENV/bin/python\" ]; then\n"
        "  python_cmd=\"$VIRTUAL_ENV/bin/python\"\n"
        "else\n"
        "  python_cmd=python3\n"
        "fi\n"
        "exec \"$python_cmd\" viewer.py \"$@\"\n"
    )
    launcher.chmod(0o755)
    (package / "README_离线审核.md").write_text(
        "# RoboCOIN 离线审核包\n\n"
        "本包只包含人工掩码审核、视频关键帧选择和本地 SAM3 框点细化。"
        "不包含批量首帧检测、SAM3.1 跟踪、VLM 或完整原始数据集。\n\n"
        "## 首次配置\n\n```bash\npython3 -m venv .venv\n"
        ".venv/bin/pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu128\n"
        ".venv/bin/pip install -r requirements-review.txt\n"
        ".venv/bin/pip install -e ./sam3\n```\n\n"
        "还需要系统命令 `ffmpeg` 和 `ffprobe`。\n\n"
        "## 启动\n\n```bash\n./start_review.sh\n```\n\n"
        "浏览器打开 <http://127.0.0.1:8888/review>。审核完成后，把整个 "
        "`objects/tracks/` 目录打包交回（数采员不需要删除任何文件；服务器端"
        "会用导入工具做增量合并）。为节省传输，也可以只交回你实际改动过的"
        " `initial_sam3_sr_2k/` 整个目录（含隐藏文件 `.export_baseline.json`）"
        "和新增关键帧对应的 `first_frame_sr_2k/source_*.jpg`。\n"
        "无论交回多少，都要保持 `objects/tracks/` 下的相对路径不变。\n\n"
        "## 服务器端导入\n\n数采员把包目录（或其中 `objects/tracks`）放到服务器后，"
        "研发人员在本项目目录执行：\n\n```bash\n"
        "python3 import_offline_review.py --returned /服务器上解压的包目录\n"
        "```\n\n"
        "导入工具只合并审核结果：不会覆盖服务器已有的 `frames/`、`first_frame/`、"
        "`tracker_sam3/` 等跟踪目录，也不会回写 `review_video/`。导入后再在服务器"
        "打开 `/review` 对标记为“等待跟踪”的数据运行跟踪即可。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="导出全部审核数据")
    parser.add_argument(
        "--include-models", action=argparse.BooleanOptionalAction, default=True,
        help="复制 SAM3 与 RealESRGAN 权重（默认启用）",
    )
    args = parser.parse_args()
    sessions = available_sessions()
    if args.all == bool(args.session):
        parser.error("必须且只能使用 --all 或至少一个 --session")
    try:
        selected = sessions if args.all else {key: sessions[key] for key in args.session}
    except KeyError as exc:
        parser.error(f"审核数据不存在：{exc.args[0]}")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"输出目录必须为空：{output}")
    output.mkdir(parents=True, exist_ok=True)

    for name in CODE_FILES:
        shutil.copy2(BASE_DIR / name, output / name)
    cache = BASE_DIR / "noun_translations_cache.json"
    if cache.is_file():
        shutil.copy2(cache, output / cache.name)
    (output / "templates").mkdir()
    shutil.copy2(
        BASE_DIR / "templates" / "review.html",
        output / "templates" / "review.html",
    )
    for key, directory in selected.items():
        copy_review_session(key, directory, output)

    if args.include_models:
        shutil.copytree(BASE_DIR / "sam3", output / "sam3")
        (output / "sam3_weights").mkdir()
        shutil.copy2(
            BASE_DIR / "sam3_weights" / "sam3.pt",
            output / "sam3_weights" / "sam3.pt",
        )
        (output / "models" / "realesrgan").mkdir(parents=True)
        shutil.copy2(
            BASE_DIR / "models" / "realesrgan" / "RealESRGAN_x2plus.pth",
            output / "models" / "realesrgan" / "RealESRGAN_x2plus.pth",
        )
    write_launch_files(output)
    print(f"已导出 {len(selected)} 条审核数据：{output}")


if __name__ == "__main__":
    main()
