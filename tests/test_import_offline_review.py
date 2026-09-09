import hashlib
import json
from pathlib import Path

import import_offline_review as imp


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def _make_manifest(frame_images, mask_names, revision=0):
    frames = []
    for image in frame_images:
        name = Path(image).name
        frames.append({
            "frame_id": Path(name).stem,
            "frame": image,
            "source_frame_index": 0,
            "timestamp_seconds": 0.0,
        })
    objects = [
        {"object_id": index, "mask": name, "source": "sam3",
         "discovery_frame_id": Path(frame_images[0]).stem, "score": 1.0}
        for index, name in enumerate(mask_names)
    ]
    return {
        "frame": frame_images[0],
        "frame_extraction_method": "test",
        "discovery_frames": frames,
        "objects": objects,
        "revision": revision,
        "next_object_id": len(mask_names),
        "tracker_dirty": False,
    }


def _session_layout(tmp_path):
    """Create parallel server/returned project layouts under tmp_path."""
    server_project = tmp_path / "server"
    returned_project = tmp_path / "returned"
    server_tracks = server_project / "objects" / "tracks"
    returned_tracks = returned_project / "objects" / "tracks"
    return server_tracks, server_project, returned_tracks, returned_project


def _session_dir(tracks_root: Path, session: str) -> Path:
    return tracks_root / session


def _populate_session(
    tracks_root: Path,
    session: str,
    mask_bytes: dict[str, bytes],
    frame_bytes: dict[str, bytes],
    revision: int,
) -> Path:
    initial = tracks_root / session
    initial.mkdir(parents=True, exist_ok=True)
    frame_dir = initial.parent / "first_frame_sr_2k"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in mask_bytes.items():
        (initial / name).write_bytes(payload)
    key_parent = Path(session).parent
    rel_frames = [f"objects/tracks/{key_parent.as_posix()}/first_frame_sr_2k/{name}"
                  for name in sorted(frame_bytes)]
    for name, payload in frame_bytes.items():
        (frame_dir / name).write_bytes(payload)
    manifest = _make_manifest(rel_frames, sorted(mask_bytes), revision=revision)
    _write_json(initial / "manifest.json", manifest)
    return initial


def _write_baseline(initial: Path, manifest_sha: str) -> None:
    _write_json(initial / ".export_baseline.json", {
        "version": 1, "manifest_sha1": manifest_sha,
    })


SESSION = "task/videos/cam/episode_000000/initial_sam3_sr_2k"
MASK_A = b"mask-a-content"
MASK_B = b"mask-b-content"
MASK_C = b"mask-c-content"
FRAME_0 = b"frame-0"
FRAME_1 = b"frame-1"

def _call(returned_manifest, returned_tracks, returned_project, server_tracks, server_project, **kwargs):
    return imp.import_session(
        returned_manifest, returned_tracks, returned_project,
        server_tracks, server_project, **kwargs,
    )


def test_unchanged_session_is_skipped(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    server_initial = _populate_session(server_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    returned_initial = _populate_session(returned_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    baseline_sha = _sha((server_initial / "manifest.json").read_bytes())
    _write_baseline(returned_initial, baseline_sha)

    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "unchanged"
    assert not (server_initial / ".history").exists()


def test_imported_edit_snapshots_and_mirrors_masks(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    server_initial = _populate_session(
        server_tracks, SESSION,
        {"object_0000.png": MASK_A, "object_0001.png": MASK_B},
        {"source_000000.jpg": FRAME_0}, revision=0,
    )
    baseline_sha = _sha((server_initial / "manifest.json").read_bytes())
    # annotator replaced one mask, dropped another, and added a new one
    returned_initial = _populate_session(
        returned_tracks, SESSION,
        {"object_0001.png": b"mask-b-edited", "object_0002.png": MASK_C},
        {"source_000000.jpg": FRAME_0}, revision=2,
    )
    _write_baseline(returned_initial, baseline_sha)

    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "imported"
    assert row["masks_removed"] == ["object_0000.png"]
    assert row["masks_changed"] == ["object_0001.png"]
    assert row["masks_added"] == ["object_0002.png"]
    server_manifest = json.loads((server_initial / "manifest.json").read_text())
    assert server_manifest["revision"] == 2
    names = {path.name for path in server_initial.glob("object_*.png")}
    assert names == {"object_0001.png", "object_0002.png"}
    assert (server_initial / "object_0001.png").read_bytes() == b"mask-b-edited"
    snapshots = list((server_initial / ".history").glob("*"))
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert json.loads((snapshot / "manifest.json").read_text())["revision"] == 0
    assert {p.name for p in snapshot.glob("object_*.png")} == {"object_0000.png", "object_0001.png"}


def test_conflict_blocks_without_flag_and_forces_with_flag(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    server_initial = _populate_session(server_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    baseline_sha = _sha((server_initial / "manifest.json").read_bytes())

    # server is edited after the export (revision bump + extra object)
    server_initial = _populate_session(server_tracks, SESSION, {"object_0000.png": MASK_A, "object_0005.png": MASK_B}, {"source_000000.jpg": FRAME_0}, revision=1)
    returned_initial = _populate_session(returned_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, revision=0)
    _write_baseline(returned_initial, baseline_sha)

    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "conflict"
    assert row["conflict"] is True
    assert json.loads((server_initial / "manifest.json").read_text())["revision"] == 1
    assert not (server_initial / ".history").exists()

    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=True,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "imported"
    assert json.loads((server_initial / "manifest.json").read_text())["revision"] == 0


def test_missing_baseline_is_backward_compatible(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    _populate_session(server_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    returned_initial = _populate_session(returned_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 2)
    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "imported"
    assert row["baseline_present"] is False


def test_referenced_keyframe_copied_but_review_video_and_extraction_untouched(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    server_initial = _populate_session(server_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    server_extraction = server_initial.parent / "first_frame_sr_2k" / "extraction.json"
    server_extraction.write_text('{"source": "RoboCOIN_datasets/original.mp4"}')
    returned_initial = _populate_session(returned_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0, "source_000001.jpg": FRAME_1}, 0)
    returned_extraction = returned_initial.parent / "first_frame_sr_2k" / "extraction.json"
    returned_extraction.write_text('{"source": "objects/tracks/.../review_video/x.mp4"}')
    (returned_initial.parent / "review_video").mkdir(parents=True)
    (returned_initial.parent / "review_video" / "x.mp4").write_bytes(b"video")
    _write_baseline(returned_initial, _sha((server_initial / "manifest.json").read_bytes()))

    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "imported"
    copied = server_initial.parent / "first_frame_sr_2k" / "source_000001.jpg"
    assert copied.read_bytes() == FRAME_1
    # server's own extraction.json must keep pointing at the original video
    assert server_extraction.read_text() == '{"source": "RoboCOIN_datasets/original.mp4"}'
    assert not (server_initial.parent / "review_video").exists()


def test_server_missing_session_reported(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    returned_initial = _populate_session(returned_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=False)
    assert row["status"] == "server_missing"


def test_prune_orphan_keyframes_respects_history(tmp_path):
    server_tracks, _, returned_tracks, _ = _session_layout(tmp_path)
    # orphan without any history reference -> pruned
    initial_a = _populate_session(server_tracks, "A/videos/cam/ep1/initial_sam3_sr_2k",
                                  {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    (initial_a.parent / "first_frame_sr_2k" / "source_000123.jpg").write_bytes(b"orphan")
    manifest_a = json.loads((initial_a / "manifest.json").read_text())
    pruned = imp.prune_orphan_keyframes(manifest_a, initial_a)
    assert pruned == ["source_000123.jpg"]
    assert not (initial_a.parent / "first_frame_sr_2k" / "source_000123.jpg").exists()

    # orphan referenced by a .history snapshot -> kept
    initial_b = _populate_session(server_tracks, "B/videos/cam/ep1/initial_sam3_sr_2k",
                                  {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    orphan = initial_b.parent / "first_frame_sr_2k" / "source_000456.jpg"
    orphan.write_bytes(b"orphan-in-history")
    history_manifest = _make_manifest(
        ["objects/tracks/B/videos/cam/ep1/first_frame_sr_2k/source_000456.jpg"],
        ["object_0000.png"], revision=0,
    )
    _write_json(initial_b / ".history" / "20200101T000000.000000Z-deadbeef" / "manifest.json", history_manifest)
    manifest_b = json.loads((initial_b / "manifest.json").read_text())
    assert imp.prune_orphan_keyframes(manifest_b, initial_b) == []
    assert orphan.exists()


def test_dry_run_makes_no_changes(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    server_initial = _populate_session(server_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 0)
    returned_initial = _populate_session(returned_tracks, SESSION, {"object_0000.png": MASK_A}, {"source_000000.jpg": FRAME_0}, 3)
    _write_baseline(returned_initial, _sha((server_initial / "manifest.json").read_bytes()))
    row = _call(returned_initial / "manifest.json", returned_tracks, returned_project,
                server_tracks, server_project, allow_conflict=False,
                prune_orphan_frames=False, dry_run=True)
    assert row["status"] == "planned"
    assert json.loads((server_initial / "manifest.json").read_text())["revision"] == 0
    assert not (server_initial / ".history").exists()


def test_locate_returned_tracks_accepts_package_and_tracks_dir(tmp_path):
    server_tracks, server_project, returned_tracks, returned_project = _session_layout(tmp_path)
    returned_project.mkdir(parents=True)
    returned_tracks.mkdir(parents=True)
    assert imp.locate_returned_tracks(returned_project) == (returned_tracks, returned_project)
    assert imp.locate_returned_tracks(returned_tracks) == (returned_tracks, returned_project)
