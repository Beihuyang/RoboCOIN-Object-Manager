import json

import noun_translations as nt
import precompute_noun_translations as pn


def test_builtin_dictionary_wins_over_cache(monkeypatch):
    monkeypatch.setattr(nt, "EXTRA_NOUN_ZH", {"cup": "高脚杯"})
    assert nt.prompt_zh("cup") == "杯子"  # NOUN_ZH is authoritative


def test_extra_cache_exact_match_and_normalization(monkeypatch):
    monkeypatch.setattr(nt, "EXTRA_NOUN_ZH", {"fidget spinner": "指尖陀螺"})
    assert nt.prompt_zh("fidget spinner") == "指尖陀螺"
    assert nt.prompt_zh("Fidget_Spinner") == "指尖陀螺"


def test_extra_cache_supports_modifier_composition(monkeypatch):
    monkeypatch.setattr(nt, "EXTRA_NOUN_ZH", {"fidget spinner": "指尖陀螺"})
    assert nt.prompt_zh("blue fidget spinner") == "蓝色指尖陀螺"


def test_reload_extra_cache_reads_file(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"silicone pad": "硅胶垫"}, ensure_ascii=False))
    monkeypatch.setenv("NOUN_TRANSLATIONS_CACHE", str(cache))
    nt.reload_extra_cache()
    assert nt.prompt_zh("silicone pad") == "硅胶垫"


def test_missing_entry_still_falls_back(monkeypatch):
    monkeypatch.setattr(nt, "EXTRA_NOUN_ZH", {})
    assert nt.prompt_zh("future unknown noun") == "待核对物体"


def test_merge_translations_normalizes_and_overrides():
    merged = pn.merge_translations(
        {"Red_Cup": "红杯", "cup": "杯子"},
        {"red cup": "红色杯子", "  spoon  ": "勺子"},
    )
    assert merged["red cup"] == "红色杯子"
    assert merged["cup"] == "杯子"
    assert list(merged) == sorted(merged)


def test_read_existing_handles_missing_and_malformed(tmp_path):
    assert pn.read_existing(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("not json")
    assert pn.read_existing(broken) == {}


def test_collect_terms_from_manifests(tmp_path):
    tracks = tmp_path / "tracks"
    session = tracks / "task/videos/cam/ep1/initial_sam3_sr_2k"
    session.mkdir(parents=True)
    manifest = {
        "prompts": ["Red  Cup", "object", "storage_box"],
        "objects": [
            {"prompt": "manual_box"},
            {"prompt": "tea cup", "matched_prompts": [{"prompt": "Tea Cup"}]},
        ],
    }
    (session / "manifest.json").write_text(json.dumps(manifest))
    terms = pn.collect_terms(tracks)
    assert terms == {"red cup", "storage box", "tea cup"}
