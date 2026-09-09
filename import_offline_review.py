#!/usr/bin/env python3
"""Import offline review annotations back into the server's objects/tracks.

The offline review package written by ``export_offline_review.py`` only
carries per-session ``initial_sam3_sr_2k/`` manifests + masks, its
``first_frame_sr_2k/`` keyframe images and the scrubbing ``review_video/``
copies.  The annotator hands the whole ``objects/tracks`` directory back, and
this tool performs a *safe incremental merge* on the server:

* imports ``initial_sam3_sr_2k/manifest.json`` and ``object_*.png``
  (server masks that the annotator deleted are removed as well);
* imports ``first_frame_sr_2k/source_*.jpg`` keyframes that are referenced by
  the returned manifest (e.g. frames appended offline);
* snapshots the previous server review state into ``.history/`` first, exactly
  like an in-browser review edit would, so server undo keeps working;
* never touches ``review_video/``, ``frames/``, ``first_frame/``,
  ``tracker_sam3/``, ``extraction.json`` or any ``.history``/``.jobs`` content,
  so existing tracking results and original-video references are preserved;
* validates the manifest baseline written at export time, so edits made on the
  server *after* the package was exported are detected instead of silently
  being overwritten.

Usage on the server (from the project directory):

    python3 import_offline_review.py --returned /path/to/returned-package \
        [--session <session-key> ...] [--dry-run] [--allow-conflict] \
        [--report report.json]

A short one-line summary is printed per session; ``--report`` writes the full
structured summary as JSON or CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_TRACKS = BASE_DIR / "objects" / "tracks"
REVIEW_DIR_NAME = "initial_sam3_sr_2k"
BASELINE_NAME = ".export_baseline.json"


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha1_bytes(path.read_bytes())


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]


def locate_returned_tracks(returned: Path) -> tuple[Path, Path] | None:
    """Return ``(tracks_root, project_root)`` for a returned offline package.

    Accepts either the package root (which contains ``objects/tracks``) or a
    path that already points at the ``objects/tracks`` directory itself.
    """
    candidate = returned.expanduser().resolve()
    if (candidate / "objects" / "tracks").is_dir():
        return (candidate / "objects" / "tracks"), candidate
    if candidate.name == "tracks":
        if candidate.parent.name == "objects":
            return candidate, candidate.parent.parent
        return candidate, candidate.parent
    return None


def iter_returned_manifests(tracks_root: Path) -> list[Path]:
    return sorted(tracks_root.glob(f"**/{REVIEW_DIR_NAME}/manifest.json"))


def session_key_for(manifest_path: Path, tracks_root: Path) -> str:
    return manifest_path.parent.relative_to(tracks_root).as_posix()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_baseline(initial_dir: Path) -> dict | None:
    baseline = read_json(initial_dir / BASELINE_NAME)
    return baseline if isinstance(baseline, dict) else None

def _referenced_relative_frames(manifest: dict) -> set[str]:
    """Project-relative frame image paths referenced by the manifest."""
    rels: set[str] = set()
    for frame in manifest.get("discovery_frames") or []:
        if isinstance(frame, dict) and isinstance(frame.get("frame"), str):
            rels.add(frame["frame"])
    for item in manifest.get("objects") or []:
        source_frame = item.get("source_frame")
        if isinstance(source_frame, str):
            rels.add(source_frame)
    return {rel for rel in rels if rel.startswith("objects/tracks/")}


def _referenced_frame_basenames(manifest: dict) -> set[str]:
    return {Path(rel).name for rel in _referenced_relative_frames(manifest)}


def snapshot_server_session(initial_dir: Path) -> Path:
    """Copy the pre-merge review state into .history so server undo still works."""
    history = initial_dir / ".history"
    snapshot = history / now_stamp()
    snapshot.mkdir(parents=True)
    shutil.copy2(initial_dir / "manifest.json", snapshot / "manifest.json")
    for mask_path in initial_dir.glob("object_*.png"):
        shutil.copy2(mask_path, snapshot / mask_path.name)
    return snapshot


def merge_manifest_and_masks(server_initial: Path, returned_initial: Path) -> dict:
    """Copy manifest.json + object_*.png; delete server masks the annotator removed."""
    server_before = {path.name for path in server_initial.glob("object_*.png")}
    returned_names = {path.name for path in returned_initial.glob("object_*.png")}
    added = sorted(returned_names - server_before)
    changed = sorted(
        name for name in returned_names & server_before
        if (returned_initial / name).read_bytes() != (server_initial / name).read_bytes()
    )
    removed = sorted(server_before - returned_names)

    temp_dir = server_initial / (".editing-" + uuid.uuid4().hex)
    temp_dir.mkdir()
    try:
        shutil.copy2(returned_initial / "manifest.json", temp_dir / "manifest.json")
        for name in returned_names:
            shutil.copy2(returned_initial / name, temp_dir / name)
        (temp_dir / "manifest.json").replace(server_initial / "manifest.json")
        for name in returned_names:
            (temp_dir / name).replace(server_initial / name)
        for name in removed:
            (server_initial / name).unlink()
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    return {"added": added, "changed": changed, "removed": removed}


def merge_referenced_frames(
    manifest: dict,
    returned_project_root: Path,
    server_project_root: Path,
) -> tuple[int, int]:
    """Copy returned keyframe images referenced by the manifest onto the server."""
    copied = 0
    missing = 0
    for rel in sorted(_referenced_relative_frames(manifest)):
        source = (returned_project_root / rel).resolve()
        target = (server_project_root / rel).resolve()
        if not source.is_file():
            missing += 1
            continue
        if target.is_file() and source.read_bytes() == target.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied, missing


def prune_orphan_keyframes(
    manifest: dict,
    server_initial: Path,
) -> list[str]:
    """Delete session keyframe images not referenced by the new manifest.

    Only files physically stored under the session's own ``first_frame_sr_2k``
    are considered, and files still referenced by any server-side ``.history``
    snapshot are kept so the server's undo chain never breaks.
    """
    frame_dir = server_initial.parent / "first_frame_sr_2k"
    if not frame_dir.is_dir():
        return []
    referenced = _referenced_frame_basenames(manifest)

    history_referenced: set[str] = set()
    for snapshot in (server_initial / ".history").glob("*/manifest.json"):
        history_referenced |= _referenced_frame_basenames(read_json(snapshot) or {})

    pruned = []
    for image in sorted(frame_dir.glob("source_*.jpg")):
        if image.name in referenced or image.name in history_referenced:
            continue
        image.unlink()
        pruned.append(image.name)
    return pruned


def _blank_row() -> dict:
    return {
        "session": None, "status": None, "baseline_present": None,
        "conflict": None, "revision_before": None, "revision_after": None,
        "objects_before": None, "objects_after": None,
        "keyframes_before": None, "keyframes_after": None,
        "masks_added": None, "masks_changed": None, "masks_removed": None,
        "frames_copied": None, "frames_missing": None, "orphans_pruned": None,
        "tracker_dirty": None, "edited_at": None, "note": None,
    }


def import_session(
    returned_manifest: Path,
    returned_tracks_root: Path,
    returned_project_root: Path,
    server_tracks_root: Path,
    server_project_root: Path,
    *,
    allow_conflict: bool,
    prune_orphan_frames: bool,
    dry_run: bool,
) -> dict:
    row = _blank_row()
    session = session_key_for(returned_manifest, returned_tracks_root)
    row["session"] = session

    returned_initial = returned_manifest.parent
    server_initial = (server_tracks_root / session).resolve()
    if not server_initial.is_dir():
        row["status"] = "server_missing"
        row["note"] = "服务器上不存在该会话，请确认回传包与服务器来源一致"
        return row

    returned_manifest_data = read_json(returned_manifest) or {}
    server_manifest_data = read_json(server_initial / "manifest.json") or {}
    returned_bytes = returned_manifest.read_bytes()
    server_sha = sha1_file(server_initial / "manifest.json")

    row["revision_before"] = server_manifest_data.get("revision")
    row["revision_after"] = returned_manifest_data.get("revision")
    row["tracker_dirty"] = returned_manifest_data.get("tracker_dirty")
    row["edited_at"] = returned_manifest_data.get("edited_at")
    row["objects_before"] = len(server_manifest_data.get("objects") or [])
    row["objects_after"] = len(returned_manifest_data.get("objects") or [])
    row["keyframes_before"] = len(server_manifest_data.get("discovery_frames") or [])
    row["keyframes_after"] = len(returned_manifest_data.get("discovery_frames") or [])

    baseline = load_baseline(returned_initial)
    row["baseline_present"] = baseline is not None
    conflict = bool(
        baseline
        and str(baseline.get("manifest_sha1") or "")
        and server_sha != str(baseline.get("manifest_sha1"))
    )
    row["conflict"] = conflict

    if not conflict and sha1_bytes(returned_bytes) == server_sha:
        row["status"] = "unchanged"
        row["note"] = "该会话没有审核修改"
        return row
    if conflict and not allow_conflict:
        row["status"] = "conflict"
        row["note"] = "服务器 manifest 与导出基线不一致：导出后服务器被改过；确认后使用 --allow-conflict"
        return row
    if conflict:
        row["note"] = "服务器 manifest 与导出基线不一致，但已通过 --allow-conflict 强制导入"
    if dry_run:
        row["status"] = "planned"
        row["note"] = "dry-run 预览"
        return row

    snapshot_server_session(server_initial)
    merged = merge_manifest_and_masks(server_initial, returned_initial)
    row["masks_added"] = merged["added"]
    row["masks_changed"] = merged["changed"]
    row["masks_removed"] = merged["removed"]
    frames_copied, frames_missing = merge_referenced_frames(
        returned_manifest_data, returned_project_root, server_project_root
    )
    row["frames_copied"] = frames_copied
    row["frames_missing"] = frames_missing
    if prune_orphan_frames:
        row["orphans_pruned"] = prune_orphan_keyframes(returned_manifest_data, server_initial)
    row["status"] = "imported"
    return row


def print_row(row: dict) -> None:
    session = row["session"] or ""
    if row["status"] == "imported":
        print(f"[imported]  {session}  masks {row['objects_before']}→{row['objects_after']}  "
              f"keyframes {row['keyframes_before']}→{row['keyframes_after']}  "
              f"revision {row['revision_before']}→{row['revision_after']}  "
              f"tracker_dirty={bool(row['tracker_dirty'])}")
    elif row["status"] == "unchanged":
        print(f"[unchanged] {session}（无修改）")
    elif row["status"] == "planned":
        print(f"[planned]   {session}（dry-run 预览）")
    else:
        print(f"[{row['status']}] {session}  {row['note']}", file=sys.stderr)


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fieldnames = list(rows[0].keys()) if rows else []
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned", type=Path, required=True,
                        help="数采员交回的目录（离线包根目录或其中的 objects/tracks）")
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER_TRACKS,
                        help="服务器 objects/tracks 目录（默认本项目）")
    parser.add_argument("--session", action="append", default=[],
                        help="只导入指定审核数据键，可重复；默认全部")
    parser.add_argument("--dry-run", action="store_true", help="只预览不写入")
    parser.add_argument("--allow-conflict", action="store_true",
                        help="服务器自导出后已变更时仍强制导入")
    parser.add_argument("--prune-orphan-frames", action="store_true",
                        help="删除新 manifest 不再引用且不被 .history 引用的关键帧图")
    parser.add_argument("--report", type=Path, default=None,
                        help="写出结构化汇总（.json 或 .csv）")
    args = parser.parse_args()

    located = locate_returned_tracks(args.returned)
    if located is None:
        parser.error("--returned 目录里找不到 objects/tracks")
    returned_tracks, returned_project_root = located
    server_tracks = args.server.expanduser().resolve()
    server_project_root = server_tracks.parent.parent
    if not server_tracks.is_dir():
        parser.error(f"服务器 tracks 目录不存在：{server_tracks}")

    manifests = iter_returned_manifests(returned_tracks)
    if args.session:
        wanted = set(args.session)
        manifests = [m for m in manifests if session_key_for(m, returned_tracks) in wanted]

    rows = []
    statuses: dict[str, int] = {}
    for manifest in manifests:
        row = import_session(
            manifest,
            returned_tracks,
            returned_project_root,
            server_tracks,
            server_project_root,
            allow_conflict=args.allow_conflict,
            prune_orphan_frames=args.prune_orphan_frames,
            dry_run=args.dry_run,
        )
        print_row(row)
        rows.append(row)
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1

    print("汇总：" + "  ".join(f"{key}={count}" for key, count in sorted(statuses.items())))
    if args.report is not None:
        write_report(args.report, rows)
        print(f"报告已写入：{args.report}")
    if statuses.get("conflict") or statuses.get("server_missing") or statuses.get("error"):
        sys.exit(2)


if __name__ == "__main__":
    main()
