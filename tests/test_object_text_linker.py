import json

import object_text_linker as linker


def _object(object_id, category, dataset, color="unknown"):
    return {
        "id": object_id,
        "name": object_id,
        "canonical_path": f"objects/new_library/{object_id}/canonical.jpg",
        "attributes": {
            "category": category,
            "color": color,
            "material": "plastic",
        },
        "source_instances": [{"session_key": f"{dataset}/camera/episode_000000"}],
    }


def test_auto_link_requires_unique_candidate_in_same_dataset():
    catalog = {
        "cup_0": _object("cup_0", "cup", "dataset_a", "white"),
        "cup_1": _object("cup_1", "cup", "dataset_b", "white"),
    }
    aliases, global_by_category, dataset_by_category = linker.build_indexes(catalog)

    local = linker.extract_mentions(
        "Pick up the white cup", "dataset_a", catalog, aliases,
        global_by_category, dataset_by_category,
    )[0]
    assert local["selected_ids"] == ["cup_0"]
    assert local["status"] == "auto"

    global_only = linker.extract_mentions(
        "Pick up the white cup", "dataset_c", catalog, aliases,
        global_by_category, dataset_by_category,
    )[0]
    assert global_only["selected_ids"] == []
    assert global_only["status"] == "pending"


def test_generate_writes_mirror_and_keeps_source(tmp_path, monkeypatch):
    source_root = tmp_path / "RoboCOIN_datasets"
    output_root = tmp_path / "RoboCOIN_object_linked"
    library_index = tmp_path / "objects" / "new_library" / "index.json"
    manifest_path = tmp_path / "objects" / "object_text_links" / "manifest.json"
    source = source_root / "dataset_a" / "meta" / "tasks.jsonl"
    source.parent.mkdir(parents=True)
    original = '{"task_index": 0, "task": "Pick up the red apple"}\n'
    source.write_text(original)
    library_index.parent.mkdir(parents=True)
    library_index.write_text(json.dumps({
        "apple_0": _object("apple_0", "apple", "dataset_a", "red")
    }))
    monkeypatch.setattr(linker, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(linker, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(linker, "LIBRARY_INDEX", library_index)
    monkeypatch.setattr(linker, "MANIFEST_PATH", manifest_path)

    manifest = linker.generate()

    assert source.read_text() == original
    linked = json.loads((output_root / "dataset_a/meta/tasks.jsonl").read_text())
    assert linked["task"] == "Pick up the [apple_0]"
    assert linked["source_task"] == "Pick up the red apple"
    assert linked["object_ids"] == ["apple_0"]
    assert manifest["mentions"][0]["occurrence_count"] == 1
