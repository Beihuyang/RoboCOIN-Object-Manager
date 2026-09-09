import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

import semantic_prompts as prompts
import stage1_track_select as stage1


def test_text_nouns_are_singular_unique_and_compound_first(monkeypatch):
    monkeypatch.setattr(prompts, "_wordnet", lambda: None)

    result = prompts.semantic_nouns_from_text(
        "Pick up cups and cup beside two red water bottles."
    )

    assert set(result) == {"cup", "water bottle", "red water bottle"}
    assert result.count("cup") == 1


def test_prompt_uses_all_named_text_sources_and_ignores_other_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "_wordnet", lambda: None)
    root = tmp_path / "RoboCOIN_datasets"
    dataset = root / "Robot_store_cup"
    video = dataset / "videos/chunk-000/camera/episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    meta = dataset / "meta/tasks.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"task": "Put cups beside the plate"}) + "\n")
    episodes = dataset / "meta/episodes.jsonl"
    episodes.write_text(
        json.dumps({"episode_index": 0, "tasks": ["Move the water bottle"]}) + "\n" +
        json.dumps({"episode_index": 1, "tasks": ["Move the hammer"]}) + "\n"
    )
    annotations = dataset / "annotations/scene_annotations.jsonl"
    annotations.parent.mkdir(parents=True)
    annotations.write_text(json.dumps({
        "scene": "A plate and a water bottle near the specific area",
        "camera_path": "hidden_hammer_camera",
    }) + "\n")
    subtasks = dataset / "annotations/subtask_annotations.jsonl"
    subtasks.write_text(json.dumps({"subtask": "Grasp the spoon"}) + "\n")
    prompts._dataset_texts.cache_clear()

    values = prompts.semantic_noun_prompts(video, root)
    assert values[0] == "object"
    assert values.count("cup") == 1
    assert "plate" in values
    assert "water bottle" in values
    assert "hammer" not in values
    assert "spoon" in values
    assert "bottle" not in values
    assert "specific" not in values
    assert "area" not in values
    assert "camera" not in values
    assert "object" in values

    broad_values = prompts.all_annotation_noun_prompts(video, root)
    assert "hammer" not in broad_values


def test_generated_scene_relations_are_not_object_nouns(monkeypatch):
    monkeypatch.setattr(prompts, "_wordnet", lambda: None)
    values = prompts.semantic_nouns_from_text(
        "The cup is located in a specific area, relatively closer to the plate; "
        "more detail references its location."
    )
    assert values == ["cup", "plate"]


def test_color_and_size_modify_objects_but_base_nouns_remain(monkeypatch):
    monkeypatch.setattr(prompts, "_wordnet", lambda: None)
    values = prompts.semantic_nouns_from_text(
        "Put the red cup in the large box beside the yellow water bottle."
    )
    assert set(values) == {
        "red cup", "cup", "large box", "box",
        "yellow water bottle", "water bottle",
    }


def test_first_frame_discovery_calls_each_noun_independently(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(frame)

    class Processor:
        def __init__(self):
            self.calls = []

        def set_confidence_threshold(self, _value):
            pass

        def reset_all_prompts(self, _state):
            pass

        def set_text_prompt(self, state, prompt):
            self.calls.append(prompt)
            mask = np.zeros((1, 8, 8), dtype=bool)
            mask[0, 2:6, 2:6] = True
            return {
                "boxes": np.array([[2, 2, 6, 6]]),
                "masks": mask,
                "scores": np.array([0.9]),
            }

    processor = Processor()
    monkeypatch.setattr(stage1, "_set_sam3_image", lambda *_args: {})
    monkeypatch.setattr(stage1.torch, "inference_mode", lambda: nullcontext())
    monkeypatch.setattr(stage1.torch, "autocast", lambda *_args, **_kwargs: nullcontext())

    _detections, report = stage1.detect_first_frame(
        frame, object(), processor, ["cup", "plate"], 0.2,
        exclude_robot_arms=False,
    )

    assert processor.calls == ["cup", "plate"]
    assert report["semantic_discovery"]["sam3_detection_calls"] == 2
    assert report["semantic_discovery"]["cross_prompt_duplicates_removed"] == 1
    assert report["enabled"] is False


def test_first_frame_decodes_mutable_processor_state_immediately(
    tmp_path, monkeypatch
):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(frame)

    class Processor:
        def __init__(self):
            self.state = {}

        def set_confidence_threshold(self, _value): pass
        def reset_all_prompts(self, _state): self.state.clear()

        def set_text_prompt(self, state, prompt):
            if prompt == "cup":
                mask = np.zeros((1, 8, 8), dtype=bool)
                mask[0, 1:4, 1:4] = True
                values = (np.array([[1, 1, 4, 4]]), mask, np.array([0.9]))
            else:
                values = (
                    np.empty((0, 4)), np.empty((0, 8, 8), dtype=bool),
                    np.empty((0,)),
                )
            self.state.update(boxes=values[0], masks=values[1], scores=values[2])
            return self.state

    processor = Processor()
    monkeypatch.setattr(stage1, "_set_sam3_image", lambda *_args: processor.state)
    monkeypatch.setattr(stage1.torch, "inference_mode", lambda: nullcontext())
    monkeypatch.setattr(stage1.torch, "autocast", lambda *_args, **_kwargs: nullcontext())

    detections, report = stage1.detect_first_frame(
        frame, object(), processor, ["cup", "missing"], 0.2,
        exclude_robot_arms=False,
    )

    assert len(detections) == 1
    assert detections[0]["prompt"] == "cup"
    assert report["semantic_discovery"]["raw_counts"] == {"cup": 1, "missing": 0}


def test_cross_prompt_merge_keeps_highest_confidence_mask():
    low = np.zeros((8, 8), dtype=bool)
    low[2:6, 2:6] = True
    high = low.copy()
    detections, duplicates = stage1.merge_cross_prompt_detections([
        ("object", [{"mask": low, "score": 0.41, "prompt": "object"}]),
        ("cup", [{"mask": high, "score": 0.83, "prompt": "cup"}]),
    ])

    assert len(detections) == 1
    assert detections[0]["prompt"] == "cup"
    assert detections[0]["score"] == 0.83
    assert len(duplicates) == 1


def test_cross_prompt_merge_preserves_same_prompt_instances():
    left = np.zeros((8, 8), dtype=bool)
    left[1:5, 1:5] = True
    right = np.zeros((8, 8), dtype=bool)
    right[2:6, 2:6] = True

    detections, duplicates = stage1.merge_cross_prompt_detections([
        ("object", [
            {"mask": left, "score": 0.9, "prompt": "object"},
            {"mask": right, "score": 0.8, "prompt": "object"},
        ]),
    ])

    assert len(detections) == 2
    assert duplicates == []


def test_decode_preserves_overlapping_instances_from_one_prompt():
    first = np.zeros((8, 8), dtype=bool)
    first[1:6, 1:6] = True
    second = np.zeros((8, 8), dtype=bool)
    second[2:7, 2:7] = True
    output = {
        "boxes": np.array([[1, 1, 6, 6], [2, 2, 7, 7]]),
        "masks": np.stack([first, second]),
        "scores": np.array([0.9, 0.8]),
    }

    detections = stage1._decode_grounding_detections(
        output, 8, 8, "object", 0.2
    )

    assert len(detections) == 2


def test_cross_prompt_merge_does_not_chain_between_neighbours():
    left = np.zeros((6, 16), dtype=bool)
    left[1:5, 0:10] = True
    bridge = np.zeros((6, 16), dtype=bool)
    bridge[1:5, 2:12] = True
    right = np.zeros((6, 16), dtype=bool)
    right[1:5, 4:14] = True

    detections, duplicates = stage1.merge_cross_prompt_detections([
        ("object", [{"mask": left, "score": 0.9, "prompt": "object"}]),
        ("cup", [{"mask": bridge, "score": 0.8, "prompt": "cup"}]),
        ("mug", [{"mask": right, "score": 0.7, "prompt": "mug"}]),
    ])

    assert len(detections) == 2
    assert len(duplicates) == 1


def test_first_frame_keeps_boundary_masks(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(frame)

    class Processor:
        def set_confidence_threshold(self, _value): pass
        def reset_all_prompts(self, _state): pass
        def set_text_prompt(self, state, prompt):
            mask = np.zeros((1, 8, 8), dtype=bool)
            mask[0, :4, :4] = True
            return {
                "boxes": np.array([[0, 0, 4, 4]]),
                "masks": mask,
                "scores": np.array([0.9]),
            }

    monkeypatch.setattr(stage1, "_set_sam3_image", lambda *_args: {})
    monkeypatch.setattr(stage1.torch, "inference_mode", lambda: nullcontext())
    monkeypatch.setattr(stage1.torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    detections, report = stage1.detect_first_frame(
        frame, object(), Processor(), ["object"], 0.2,
        exclude_robot_arms=False,
    )

    assert len(detections) == 1
    assert report["boundary_filter"]["enabled"] is False
    assert report["boundary_filter"]["objects_removed"] == 0


def test_first_frame_removes_robot_arm_overlaps(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(frame)

    class Processor:
        def set_confidence_threshold(self, _value): pass
        def reset_all_prompts(self, _state): pass
        def set_text_prompt(self, state, prompt):
            mask = np.zeros((1, 8, 8), dtype=bool)
            mask[0, 1:5, 1:5] = True
            return {
                "boxes": np.array([[1, 1, 5, 5]]),
                "masks": mask,
                "scores": np.array([0.9]),
            }

    robot_mask = np.zeros((8, 8), dtype=bool)
    robot_mask[1:5, 1:5] = True
    monkeypatch.setattr(stage1, "_set_sam3_image", lambda *_args: {})
    monkeypatch.setattr(
        stage1, "_detect_robot_regions_from_state",
        lambda *_args: [{"mask": robot_mask, "score": 0.8, "bbox": [1, 1, 5, 5]}],
    )
    monkeypatch.setattr(stage1.torch, "inference_mode", lambda: nullcontext())
    monkeypatch.setattr(stage1.torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    detections, report = stage1.detect_first_frame(
        frame, object(), Processor(), ["object"], 0.2,
        exclude_robot_arms=True, robot_arm_overlap=0.5,
    )

    assert detections == []
    assert report["enabled"] is True
    assert report["regions_detected"] == 1
    assert report["objects_removed"] == 1
