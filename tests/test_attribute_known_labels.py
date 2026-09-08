import stage3_attribute as stage3


def test_object_prompt_leaves_vlm_identification_unconstrained():
    assert stage3._known_mask_label("object") == {}
    answer = {"object_name_hint": "cup", "attribute_ids": {"color": "color.red"}}
    assert stage3._apply_known_mask_label(answer, {}) == answer


def test_named_prompt_supplies_identity_color_and_size():
    known = stage3._known_mask_label("small red cup")

    assert known == {
        "mask_prompt": "small red cup",
        "object_name": "cup",
        "attribute_ids": {
            "size": "size.small",
            "color": "color.red",
        },
    }
    merged = stage3._apply_known_mask_label({
        "object_name_hint": "mug",
        "attribute_ids": {
            "color": "color.blue",
            "material": "material.ceramic",
        },
    }, known)
    assert merged["object_name_hint"] == "cup"
    assert merged["attribute_ids"] == {
        "color": "color.red",
        "size": "size.small",
        "material": "material.ceramic",
    }


def test_size_taxonomy_is_available_to_attribute_annotation():
    ontology = stage3._load_ontology()
    size = ontology["attributes"]["size"]

    assert size["id"] == "size"
    assert {item["id"] for item in size["children"]} >= {
        "size.unknown", "size.small", "size.large"
    }


def test_wordnet_candidates_only_use_concrete_object_branch():
    wordnet = stage3._wordnet()
    concrete = wordnet.synset(stage3.WORDNET_CONCRETE_ROOT)

    paths = stage3._wordnet_candidate_paths(wordnet, "cup")
    assert paths
    assert all(concrete in path for path in paths)

    abstract_answer = {
        "object_name_hint": "attribute",
        "category_synset": "attribute.n.02",
    }
    assert stage3._validated_requested_wordnet_path(
        wordnet, abstract_answer
    ) is None
    fallback = stage3._resolve_single_pass_wordnet_path(wordnet, abstract_answer)
    assert fallback[-1] == concrete


def test_name_prompt_requests_only_plain_object_name():
    prompt = stage3._object_name_prompt()
    assert '{"object_name": "cup"}' in prompt
    assert "WordNet ID" in prompt
    assert "attribute_ids" not in prompt


def test_wordnet_lookup_prefers_full_phrase_and_keeps_ambiguous_senses():
    wordnet = stage3._wordnet()
    water_bottles, attempted = stage3._physical_name_synsets(
        wordnet, "water bottle"
    )
    assert attempted == ["water_bottle"]
    assert [node.name() for node in water_bottles] == ["water_bottle.n.01"]

    cups, attempted = stage3._physical_name_synsets(wordnet, "cup")
    assert attempted == ["cup"]
    assert len(cups) > 1
    assert {node.name() for node in cups} >= {"cup.n.01", "cup.n.05"}


def test_attribute_prompt_requests_complete_chinese_display_data():
    wordnet = stage3._wordnet()
    path = stage3._canonical_wordnet_path(wordnet, wordnet.synset("cup.n.01"))
    prompt = stage3._all_attributes_prompt("red cup", "cup", path)

    assert '"chinese_display"' in prompt
    assert '"object_name": "简短准确的中文物体名"' in prompt
    assert '"category_nodes"' in prompt
    assert '"attribute_values"' in prompt
