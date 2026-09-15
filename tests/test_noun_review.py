import noun_review


class StubBackend:
    def __init__(self):
        self.prompts = []

    def generate_json(self, images, prompt, max_new_tokens):
        assert images is None
        self.prompts.append(prompt)
        nouns = [
            line.split(". ", 1)[1]
            for line in prompt.splitlines()
            if line.split(". ", 1)[0].isdigit()
        ]
        return {
            "review": [
                {
                    "noun": noun,
                    "verdict": "drop" if noun == "arrangement" else "keep",
                    "reason": "test verdict",
                }
                for noun in nouns
            ]
        }, "stub"


def _configure(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(noun_review, "CACHE_PATH", tmp_path / "noun-cache.json")
    monkeypatch.setattr(noun_review, "openai_backend_from_env", lambda: backend)
    monkeypatch.setenv("ROBOCOIN_NOUN_REVIEW", "1")
    monkeypatch.setenv("ROBOCOIN_VLM_BACKEND", "api")
    monkeypatch.setenv("VLM_MODEL", "vision-model")
    monkeypatch.setenv("VLM_API_BASE", "https://first.example/v1/")


def test_review_prompt_uses_dataset_context_and_keeps_physical_furniture(
    tmp_path, monkeypatch,
):
    backend = StubBackend()
    _configure(tmp_path, monkeypatch, backend)

    filtered, report = noun_review.filter_prompts(
        ["object", "table", "arrangement"],
        "robot_arrange_table",
        ["Move the cup onto the dining table."],
    )

    assert filtered == ["object", "table"]
    assert [item["noun"] for item in report["dropped"]] == ["arrangement"]
    assert "Dataset name: robot_arrange_table" in backend.prompts[0]
    assert "Move the cup onto the dining table." in backend.prompts[0]
    assert "table, shelf, cabinet" in backend.prompts[0]


def test_context_and_api_base_both_invalidate_noun_cache(tmp_path, monkeypatch):
    backend = StubBackend()
    _configure(tmp_path, monkeypatch, backend)

    noun_review.review_nouns(["can"], "dataset-a", ["Put the can in a box."])
    noun_review.review_nouns(["can"], "dataset-a", ["Put the can in a box."])
    assert len(backend.prompts) == 1

    noun_review.review_nouns(["can"], "dataset-b", ["Open the can."])
    assert len(backend.prompts) == 2

    monkeypatch.setenv("VLM_API_BASE", "https://second.example/v1")
    noun_review.review_nouns(["can"], "dataset-b", ["Open the can."])
    assert len(backend.prompts) == 3


def test_filter_fails_open_when_review_backend_fails(tmp_path, monkeypatch):
    class BrokenBackend:
        def generate_json(self, *_args, **_kwargs):
            raise RuntimeError("offline")

    _configure(tmp_path, monkeypatch, BrokenBackend())
    original = ["object", "cup", "arrangement"]

    filtered, report = noun_review.filter_prompts(
        original, "dataset", ["Arrange a cup."],
    )

    assert filtered == original
    assert report["enabled"] is False
    assert "VLM review failed" in report["reason"]
