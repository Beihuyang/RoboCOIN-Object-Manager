from noun_translations import normalize_prompt, prompt_zh


def test_known_sam3_prompts_have_chinese_review_names():
    assert prompt_zh("cup") == "杯子"
    assert prompt_zh("water_bottle") == "水瓶"
    assert prompt_zh("yellow box") == "黄色盒子"
    assert prompt_zh("object") == "通用物体"


def test_unknown_prompt_never_leaks_english_into_chinese_interface():
    assert prompt_zh("future unknown noun") == "待核对物体"
    assert normalize_prompt("  Water_Bottle ") == "water bottle"
