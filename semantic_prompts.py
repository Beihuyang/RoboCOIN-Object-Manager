"""Derive compact SAM3 noun prompts from RoboCOIN dataset names."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


PROMPT_STRATEGY = "dataset_nouns_v3"
DEFAULT_MAX_SEMANTIC_PROMPTS = 6

# Robot/platform identifiers and task grammar are metadata, not visible targets.
PLATFORM_WORDS = {
    "agi", "agibot", "agilex", "ai", "aida", "ai2", "airbot", "aloha", "alpha",
    "alphabot", "bot", "cobot", "g1", "g1edu", "galaxea", "galbot",
    "baai", "edu", "kuavo", "leju", "lite", "magic", "mmk", "mmk2", "realman",
    "rmc", "robot", "robotic", "split", "tianqin", "u3", "unitree",
}
TASK_WORDS = {
    "arrange", "away", "bagging", "bring", "brush", "build", "capture", "carry",
    "change", "classify", "clean", "cleaning", "clear", "clink", "close", "closing",
    "connect", "control", "discard", "disposal", "double", "erase", "fold", "get",
    "grind", "hang", "heat", "hold", "insert", "load", "make", "measure", "mix",
    "move", "moving", "movethe", "open", "operate", "organise", "organize", "package",
    "packaging", "pass", "passing", "pick", "picks", "place", "placement", "play",
    "pour", "prepare", "press", "pull", "pump", "push", "pushing", "put", "recover",
    "remove", "repeatedly", "reversal", "rotate", "scoop", "shake", "slide", "sliding",
    "sort", "stack", "stacking", "stamp", "stir", "stirring", "storage", "store",
    "storaje", "storge", "swap", "swipe", "swiping", "take", "throw", "toggle",
    "turn", "twice", "twist", "unload", "unplug", "unscrew", "use", "wash", "washing",
    "wipe", "write",
}
FUNCTION_WORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on",
    "the", "to", "up", "with", "position", "left", "right", "top", "bottom",
}
GENERIC_WORDS = {"object", "objects", "part", "parts", "task", "scene"}
NON_TARGET_WORDS = {
    "arrangement", "beauty", "both", "care", "classification", "closest", "color",
    "conditioning", "data", "error", "front", "hotel", "interference", "item", "items",
    "down", "large", "leftover", "line", "little", "liquid", "middle", "model", "number", "off",
    "onto", "or", "out", "personal", "service", "services", "setting", "side", "singletry",
    "powder", "small", "solid", "stuff", "test", "then", "touching", "unordered", "water", "wet",
    # Background/support surfaces are deliberately not semantic discovery targets.
    "desktop", "floor", "table",
    # Robot anatomy must not be reintroduced as a task-object prompt.
    "arm", "face", "hand", "hands",
}
COLOR_AND_SIZE_WORDS = {
    "black", "blue", "brown", "canned", "gray", "green", "grey", "large", "orange",
    "pink", "purple", "red", "small", "wet", "white", "yellow",
}
AMBIGUOUS_NOUN_COLORS = {"orange"}
KNOWN_COMPOUNDS = {
    "battery box", "bluetooth speaker", "building block", "cake plate",
    "coffee bean", "cookie cup", "dog doll", "electric kettle", "ice cream",
    "marker pen", "microwave oven", "mobile phone", "paper box", "power bank",
    "sensor card", "storage box", "table tennis ball", "takeout bag", "takeout box",
    "tennis ball", "test tube", "tissue box", "toy car", "water bottle", "wet wipe",
}
UNKNOWN_TARGETS = {
    "baozi", "bluetooth", "electronics", "nightstand", "rubik", "teaset",
}
PHYSICAL_LEXNAMES = {
    "noun.animal", "noun.artifact", "noun.communication", "noun.food",
    "noun.person", "noun.plant", "noun.shape",
}
SELECTED_SUBSTANCES = {
    "cardboard", "garbage", "glass", "iron", "marble", "paper", "sponge",
    "straw", "tissue", "trash", "wallpaper",
}


def _tokens(name: str) -> list[str]:
    expanded = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    expanded = re.sub(r"movethe", "move_the", expanded, flags=re.IGNORECASE)
    return [token for token in re.split(r"[^a-zA-Z]+", expanded.lower()) if token]


@lru_cache(maxsize=1)
def _wordnet():
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
        return wordnet
    except (ImportError, LookupError):
        return None


def _singular(token: str) -> str:
    wordnet = _wordnet()
    if wordnet is not None:
        lemma = wordnet.morphy(token, wordnet.NOUN)
        if lemma:
            return lemma.replace("_", " ")
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    # Keep the fallback conservative.  Blindly removing ``es`` produced broken
    # prompts such as ``tenni``, ``wip``, ``stor`` and ``servic`` on machines
    # without the optional NLTK WordNet corpus.
    if token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if (
        token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
        and token != "news"
        and len(token) > 3
    ):
        return token[:-1]
    return token


def _is_wordnet_compound(words: list[str]) -> bool:
    phrase = "_".join(words)
    if " ".join(words) in KNOWN_COMPOUNDS:
        return True
    if any(word in COLOR_AND_SIZE_WORDS for word in words):
        return False
    wordnet = _wordnet()
    return bool(wordnet and wordnet.synsets(phrase, pos=wordnet.NOUN))


def _is_target_word(token: str) -> bool:
    if token in NON_TARGET_WORDS:
        return False
    if token in UNKNOWN_TARGETS or token in SELECTED_SUBSTANCES:
        return True
    wordnet = _wordnet()
    if wordnet is None:
        return len(token) > 2
    return any(
        synset.lexname() in PHYSICAL_LEXNAMES
        for synset in wordnet.synsets(token, pos=wordnet.NOUN)
    )


def dataset_name_from_video(video_path: Path, video_root: Path) -> str:
    """Return the dataset directory component for a source video."""
    try:
        return video_path.resolve().relative_to(video_root.resolve()).parts[0]
    except (ValueError, IndexError):
        for parent in video_path.parents:
            if parent.name and parent.name not in {"videos", "chunk-000"}:
                return parent.name
    return video_path.stem


def semantic_prompts_from_dataset(
    dataset_name: str,
    max_prompts: int = DEFAULT_MAX_SEMANTIC_PROMPTS,
) -> list[str]:
    """Extract specific noun phrases while excluding robot and action metadata."""
    raw = _tokens(dataset_name)
    normalized = []
    for token in raw:
        if (
            token in PLATFORM_WORDS
            or token in FUNCTION_WORDS
            or token in GENERIC_WORDS
            or len(token) <= 2
        ):
            continue
        normalized.append(_singular(token))

    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip().replace("_", " ")
        if value and value not in candidates:
            candidates.append(value)

    # Preserve curated compounds before dropping action/material words. This is
    # why "wet wipes" and "water bottle" survive while bare "wipe"/"water" do not.
    compound_non_heads = set()
    compound_members = set()
    for size in (3, 2):
        for index in range(len(normalized) - size + 1):
            words = normalized[index:index + size]
            phrase = " ".join(words)
            if phrase in KNOWN_COMPOUNDS:
                add(phrase)
                compound_members.update(range(index, index + size))
                compound_non_heads.update(words[:-1])

    meaningful = []
    for index, token in enumerate(normalized):
        if index not in compound_members and (
            token in TASK_WORDS or token in NON_TARGET_WORDS
        ):
            continue
        if (
            index in compound_members
            or token in COLOR_AND_SIZE_WORDS
            or _is_target_word(token)
        ):
            meaningful.append(token)

    # Prefer validated long compounds; SAM3 generally grounds short noun
    # phrases better than a sentence or a bag of unrelated dataset tokens.
    for size in (3, 2):
        for index in range(len(meaningful) - size + 1):
            words = meaningful[index:index + size]
            if _is_wordnet_compound(words):
                add(" ".join(words))
                compound_non_heads.update(words[:-1])

    # Color/size modifiers are visually useful even when WordNet has no compound.
    for index in range(len(meaningful) - 1):
        if (
            meaningful[index] in COLOR_AND_SIZE_WORDS
            and meaningful[index] not in AMBIGUOUS_NOUN_COLORS
            and meaningful[index + 1] not in COLOR_AND_SIZE_WORDS
        ):
            if meaningful[index + 1] in compound_non_heads:
                for size in (3, 2):
                    suffix = meaningful[index + 1:index + 1 + size]
                    if " ".join(suffix) in KNOWN_COMPOUNDS:
                        add(" ".join([meaningful[index], *suffix]))
                        break
                continue
            add(f"{meaningful[index]} {meaningful[index + 1]}")

    # Head-noun fallbacks also cover unseen product names such as "bluetooth".
    for token in meaningful:
        if (
            (token not in COLOR_AND_SIZE_WORDS or token in AMBIGUOUS_NOUN_COLORS)
            and token not in compound_non_heads
        ):
            add(token)
    return candidates[:max(0, max_prompts)]


def discovery_prompts(
    dataset_name: str,
    base_prompt: str = "object",
    max_semantic_prompts: int = DEFAULT_MAX_SEMANTIC_PROMPTS,
) -> list[str]:
    prompts = [base_prompt.strip() or "object"]
    for prompt in semantic_prompts_from_dataset(dataset_name, max_semantic_prompts):
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts
