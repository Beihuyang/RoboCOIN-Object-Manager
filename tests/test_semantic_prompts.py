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

    assert set(result) == {"cup", "water bottle"}
    assert result.count("cup") == 1


def test_prompt_uses_object_dataset_and_current_episode_meta_only(tmp_path, monkeypatch):
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
    annotations.write_text(json.dumps({"scene": "A plate and a water bottle"}) + "\n")
    prompts._dataset_texts.cache_clear()

    values = prompts.semantic_noun_prompts(video, root)
    assert values[0] == "object"
    assert values.count("cup") == 1
    assert "plate" in values
    assert "water bottle" in values
    assert "hammer" not in values
    assert "bottle" not in values
    assert "object" in values

    broad_values = prompts.all_annotation_noun_prompts(video, root)
    assert "hammer" in broad_values


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
        exclude_robot_arms=True,
    )

    assert processor.calls == ["cup", "plate"]
    assert report["semantic_discovery"]["sam3_detection_calls"] == 2
    assert report["semantic_discovery"]["cross_prompt_duplicates_removed"] == 1
    assert report["enabled"] is False
