import json
from pathlib import Path

import numpy as np
from PIL import Image

import stage1_track_select
import stage4_dedup
from dedup_tree import generate_layout, inherit_unchanged_adjustments, move_instance
from stage4_dedup import items_fingerprint


def _item(instance_id: str, category: str, fingerprint: str = "image") -> dict:
    return {
        "instance_id": instance_id,
        "fingerprint": fingerprint,
        "attributes": {
            "category": category,
            "color": "unknown",
            "material": "unknown",
            "shape": "unknown",
            "texture": "unknown",
        },
        "category_path": [
            {"id": "entity.n.01", "label": "entity"},
            {"id": f"category.{category}", "label": category},
        ],
    }


def test_items_fingerprint_changes_with_attributes():
    before = items_fingerprint([_item("one", "cup")])
    after = items_fingerprint([_item("one", "bowl")])
    assert before != after


def test_library_ids_are_zero_based_per_category_and_deterministic():
    cups = [[_item("cup-b", "cup")], [_item("cup-a", "cup")], [_item("cup-c", "cup")]]
    bowls = [[_item("bowl-a", "mixing bowl")]]
    empty = {"version": 1, "next_indices": {}, "objects": {}}
    first, _ = stage4_dedup.assign_persistent_library_ids(cups + bowls, empty)
    second, _ = stage4_dedup.assign_persistent_library_ids(list(reversed(cups + bowls)), empty)

    assert first == second
    assert sorted(object_id for object_id in first.values() if object_id.startswith("cup_")) == [
        "cup_0", "cup_1", "cup_2"
    ]
    assert "mixing_bowl_0" in first.values()


def test_library_id_survives_instance_changes_and_numbers_are_not_reused():
    initial_groups = [[_item("a", "cup")], [_item("b", "cup")]]
    initial, registry = stage4_dedup.assign_persistent_library_ids(
        initial_groups, {"version": 1, "next_indices": {}, "objects": {}}
    )
    id_a = initial["a"]
    id_b = initial["b"]

    changed, registry = stage4_dedup.assign_persistent_library_ids(
        [[_item("a", "cup"), _item("c", "cup")]], registry
    )
    assert changed["a|c"] == id_a
    assert registry["objects"][id_b]["active"] is False

    added, _ = stage4_dedup.assign_persistent_library_ids(
        [[_item("a", "cup"), _item("c", "cup")], [_item("d", "cup")]],
        registry,
    )
    assert added["a|c"] == id_a
    assert added["d"] == "cup_2"


def test_library_id_does_not_change_when_category_changes():
    initial, registry = stage4_dedup.assign_persistent_library_ids(
        [[_item("a", "cup")]],
        {"version": 1, "next_indices": {}, "objects": {}},
    )
    changed, _ = stage4_dedup.assign_persistent_library_ids(
        [[_item("a", "mug")]], registry
    )
    assert changed["a"] == initial["a"] == "cup_0"


def test_category_override_drops_stale_wordnet_path(tmp_path, monkeypatch):
    image_path = tmp_path / "best.jpg"
    Image.new("RGB", (4, 4), "white").save(image_path)
    cache_path = tmp_path / "attributes.jsonl"
    cache_path.write_text(json.dumps({
        **_item("one", "cup"),
        "tracker_backend": "sam3",
        "best_quality_path": str(image_path),
        "library_eligible": True,
        "category_synset": "cup.n.01",
        "wordnet_warning": {"reason": "old"},
    }) + "\n")
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"one": {"category": "bowl"}}))
    monkeypatch.setattr(stage4_dedup, "ATTR_CACHE", cache_path)
    monkeypatch.setattr(stage4_dedup, "OVERRIDES_PATH", overrides_path)

    item = stage4_dedup.load_attributes()[0]
    assert item["attributes"]["category"] == "bowl"
    assert "category_path" not in item
    assert "category_synset" not in item
    assert "wordnet_warning" not in item


def test_changed_instance_does_not_inherit_old_category_placement():
    old_item = _item("one", "cup")
    previous = generate_layout([old_item])
    previous["clusters"][0]["node_id"] = "category.manual"
    previous["nodes"].append({
        "id": "category.manual", "label": "manual", "parent_id": "entity.n.01"
    })
    previous["instance_fingerprints"].pop("one")

    generated = generate_layout([_item("one", "bowl")])
    inherited = inherit_unchanged_adjustments(previous, generated)
    cluster = next(c for c in inherited["clusters"] if "one" in c["member_ids"])
    assert cluster["node_id"] == "category.bowl"


def test_move_trashed_object_back_to_root_creates_manual_cluster():
    layout = {
        "nodes": [{"id": "entity.n.01", "label": "entity", "parent_id": None}],
        "clusters": [],
        "trash_instance_ids": ["one"],
    }

    move_instance(layout, "one", None, "entity.n.01")

    assert layout["trash_instance_ids"] == []
    assert len(layout["clusters"]) == 1
    assert layout["clusters"][0]["node_id"] == "entity.n.01"
    assert layout["clusters"][0]["member_ids"] == ["one"]
    assert layout["clusters"][0]["origin"] == "manual"


def test_category_tree_preserves_chinese_display_fields():
    item = _item("one", "cup")
    item["category_path"][0].update({
        "label_zh": "实体", "definition_zh": "独立存在的事物"
    })

    generated = generate_layout([item])
    root = next(node for node in generated["nodes"] if node["id"] == "entity.n.01")
    assert root["label_zh"] == "实体"
    assert root["definition_zh"] == "独立存在的事物"


def test_tracking_revision_guard_keeps_new_edits_dirty(tmp_path, monkeypatch):
    directory = tmp_path / "initial_sam3_sr_2k"
    directory.mkdir()
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps({"revision": 4, "tracker_dirty": True}))
    monkeypatch.setattr(stage1_track_select, "initial_mask_dir", lambda _video: directory)

    assert not stage1_track_select.mark_initial_masks_tracked(Path("video.mp4"), 3)
    assert json.loads(manifest_path.read_text())["tracker_dirty"] is True

    assert stage1_track_select.mark_initial_masks_tracked(Path("video.mp4"), 4)
    saved = json.loads(manifest_path.read_text())
    assert saved["tracker_dirty"] is False
    assert saved["tracked_revision"] == 4


def test_track_export_failure_keeps_previous_directory(tmp_path, monkeypatch):
    output_root = tmp_path / "tracks"
    output_dir = output_root / "session" / "tracker_sam3"
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text('{"old": true}')
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (12, 12), "white").save(frame_path)
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    candidate = stage1_track_select.FrameCandidate(
        frame_index=0,
        frame_path=str(frame_path),
        mask_area=int(mask.sum()),
        area_ratio=float(mask.mean()),
        sharpness=1.0,
        confidence=1.0,
        touches_boundary=False,
    )
    monkeypatch.setattr(stage1_track_select, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(stage1_track_select, "video_key", lambda _path: "session")
    monkeypatch.setattr(
        stage1_track_select,
        "export_choice",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    monkeypatch.setattr(
        "super_resolution.upscale_for_sam3", lambda image: image
    )

    try:
        stage1_track_select.export_tracks(
            Path("video.mp4"),
            [frame_path],
            {1: [candidate]},
            {(1, 0): stage1_track_select.pack_mask(mask)},
            1.0,
            "object",
            "sam3",
            "sam3.1",
        )
    except RuntimeError as exc:
        assert str(exc) == "write failed"
    else:
        raise AssertionError("export_tracks should propagate the write failure")

    assert json.loads((output_dir / "manifest.json").read_text()) == {"old": True}
    assert not list(output_dir.parent.glob("tracker_sam3.building-*"))
