"""Derive deduplicated SAM3 noun prompts from dataset names and text metadata."""

from __future__ import annotations

import re
import json
from functools import lru_cache
from pathlib import Path


PROMPT_STRATEGY = "object_all_metadata_physical_nouns_v7"
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
    "down", "leftover", "line", "little", "liquid", "middle", "model", "number", "off",
    "onto", "or", "out", "personal", "service", "services", "setting", "side", "singletry",
    "powder", "solid", "stuff", "test", "then", "touching", "unordered", "water", "wet",
    # Background/support surfaces are deliberately not semantic discovery targets.
    "desktop", "floor", "table",
    # Robot anatomy must not be reintroduced as a task-object prompt.
    "arm", "face", "hand", "hands",
    # Abstract/spatial/discourse nouns that occur frequently in generated
    # scene descriptions but cannot be grounded as independent objects.
    "area", "center", "closer", "corner", "detail", "element", "end",
    "instance", "interaction", "location", "null", "pair", "piece",
    "point", "position", "power", "reference", "relation", "relationship",
    "section", "setup", "side", "space", "spot", "surface", "time", "view",
    # Image/annotation and robot-control vocabulary, not scene targets.
    "abnormal", "grasp", "gripper", "image", "recognition", "static",
}
COLOR_AND_SIZE_WORDS = {
    "black", "blue", "brown", "canned", "gray", "green", "grey", "large", "orange",
    "pink", "purple", "red", "small", "wet", "white", "yellow",
}
AMBIGUOUS_NOUN_COLORS = {"orange"}
KNOWN_COMPOUNDS = {
    "battery box", "bluetooth speaker", "building block", "cake plate",
    "cardboard box", "coffee bean", "cookie cup", "dog doll", "electric kettle",
    "glass cup", "ice cream", "measuring cup",
    "marker pen", "microwave oven", "mobile phone", "paper box", "power bank",
    "paper cup", "plastic bag", "rubik cube", "sensor card", "soda bottle",
    "storage box", "table tennis ball", "takeout bag", "takeout box", "tennis ball",
    "test tube", "tissue box", "toy car", "wallpaper knife", "water bottle", "wet wipe",
}
UNKNOWN_TARGETS = {
    "baozi", "electronics", "nightstand", "teaset",
}
TEXT_PHYSICAL_TASK_WORDS = {"brush", "scoop", "stamp"}
TEXT_STOP_WORDS = {
    "above", "after", "alongside", "another", "around", "back", "before",
    "behind", "beside", "between", "both", "center", "central", "down", "each",
    "eight", "end", "five", "focal", "four", "lift", "near", "nine", "one",
    "point", "rectangle", "round", "same", "section", "serving", "seven",
    "share", "six", "there", "three", "through", "toward", "two", "under",
    "upon", "view", "well",
    "abnormal", "across", "additional", "additionally", "adjacent", "all",
    "along", "also", "appear", "are", "arranged", "attached", "being",
    "beyond", "but", "can", "centered", "centrally", "closer", "clustered",
    "contain", "described", "different", "direct", "directly", "distinct",
    "duplicate", "either", "establishing", "exact", "far", "finally",
    "first", "following", "found", "further", "general", "has", "have",
    "identical", "immediate", "include", "including", "indicating", "its",
    "itself", "labeled", "last", "lastly", "likely", "located", "mentioned",
    "more", "multiple", "nearby", "next", "not", "noted", "occupy",
    "occupying", "opposite", "other", "overlapping", "placed", "positioned",
    "positional", "possibly", "present", "rear", "rectangle", "rectangular",
    "relative", "relatively", "relationship", "respective", "respectively", "same", "second",
    "separate", "several", "sharing", "shaped", "similar", "similarly",
    "single", "situated", "slightly", "smaller", "some", "spatial", "specific",
    "specifically", "specified", "spread", "static", "that", "their", "them",
    "these", "they", "third", "this", "throughout", "together", "transfer",
    "unspecified", "various", "where", "which", "while", "without", "wooden",
    "your",
}
TEXT_SOURCE_FIELDS = {
    "meta/tasks.jsonl": ("task",),
    "meta/episodes.jsonl": ("tasks",),
    "annotations/scene_annotations.jsonl": ("scene", "scene_annotation"),
    "annotations/subtask_annotations.jsonl": ("subtask",),
    "annotations/subtasks.jsonl": ("subtask",),
}
PHYSICAL_LEXNAMES = {
    "noun.animal", "noun.artifact", "noun.communication", "noun.food",
    "noun.person", "noun.plant", "noun.shape",
}
SELECTED_SUBSTANCES = {
    "cardboard", "garbage", "glass", "iron", "marble", "paper", "sponge",
    "straw", "tissue", "trash", "wallpaper",
}
FALLBACK_OBJECT_WORDS = {
    "apple", "bag", "banana", "basket", "bean", "book", "bottle", "bowl",
    "box", "bread", "brush", "cabinet", "cake", "can", "card", "cloth",
    "container", "cookie", "cup", "cube", "doll", "egg", "fruit", "glass",
    "hammer", "kettle", "lemon", "lid", "marker", "microwave", "mouse",
    "oven", "paper", "peach", "pen", "phone", "plate", "pot", "potato",
    "rack", "sensor", "shelf", "speaker", "spoon", "sponge", "straw",
    "tube", "tablecloth", "tissue", "towel", "toy", "tray", "wipe",
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
        # A missing optional corpus must fail closed for free-form annotation
        # prose. Otherwise adjectives, relations and verbs all become prompts.
        return token in FALLBACK_OBJECT_WORDS
    physical_entity = wordnet.synset("physical_entity.n.01")
    for synset in wordnet.synsets(token, pos=wordnet.NOUN):
        if synset == physical_entity:
            return True
        if physical_entity in set(synset.closure(lambda value: value.hypernyms())):
            return True
    return False


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


def semantic_nouns_from_text(text: str) -> list[str]:
    """Extract physical noun phrases from free text with stable deduplication."""
    words = [_singular(token) for token in _tokens(text)]
    phrases: list[str] = []
    compound_members: set[int] = set()
    noun_spans: list[tuple[int, int, str]] = []

    def add(value: str) -> None:
        value = value.strip().replace("_", " ")
        if value and value not in phrases:
            phrases.append(value)

    for size in (3, 2):
        for index in range(len(words) - size + 1):
            values = words[index:index + size]
            phrase = " ".join(values)
            if phrase in KNOWN_COMPOUNDS:
                add(phrase)
                compound_members.update(range(index, index + size))
                noun_spans.append((index, index + size, phrase))
                continue
            if any(
                value in TEXT_STOP_WORDS
                or value in FUNCTION_WORDS
                or (value in TASK_WORDS and value not in TEXT_PHYSICAL_TASK_WORDS)
                or value in NON_TARGET_WORDS
                for value in values
            ):
                continue
            if _is_wordnet_compound(values):
                add(phrase)
                compound_members.update(range(index, index + size))
                noun_spans.append((index, index + size, phrase))

    # Keep one directly adjacent color/size modifier, while retaining the base
    # noun phrase as a recall fallback. In "blue yellow large test tube", only
    # the adjacent "large test tube" is formed; unrelated colors are not chained.
    for start, _end, phrase in noun_spans:
        if start > 0 and words[start - 1] in COLOR_AND_SIZE_WORDS:
            add(f"{words[start - 1]} {phrase}")

    for index, word in enumerate(words):
        if index in compound_members:
            continue
        if (
            word in PLATFORM_WORDS
            or word in FUNCTION_WORDS
            or word in GENERIC_WORDS
            or word in NON_TARGET_WORDS
            or word in COLOR_AND_SIZE_WORDS
            or word in TEXT_STOP_WORDS
            or len(word) <= 2
        ):
            continue
        if word in TASK_WORDS and word not in TEXT_PHYSICAL_TASK_WORDS:
            continue
        if _is_target_word(word):
            if index > 0 and words[index - 1] in COLOR_AND_SIZE_WORDS:
                add(f"{words[index - 1]} {word}")
            add(word)
    return phrases


def _json_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _json_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _json_strings(item)]
    return []


@lru_cache(maxsize=4096)
def _dataset_texts(dataset_dir: str, episode_index: int | None = None) -> tuple[str, ...]:
    """Read only human-language fields, never arbitrary JSON string values."""
    root = Path(dataset_dir)
    texts = []
    for relative, fields in TEXT_SOURCE_FIELDS.items():
        path = root / relative
        if not path.is_file():
            continue
        try:
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        row_index = None
                        if relative == "meta/episodes.jsonl":
                            row_index = row.get("episode_index")
                        elif relative == "annotations/scene_annotations.jsonl":
                            row_index = row.get("episode_idx", row.get("scene_index"))
                        if (
                            episode_index is not None
                            and row_index is not None
                            and int(row_index) != episode_index
                        ):
                            continue
                        for field in fields:
                            texts.extend(_json_strings(row.get(field)))
        except (OSError, json.JSONDecodeError):
            continue
    return tuple(texts)


def _deduplicate_compound_nouns(nouns: list[str]) -> list[str]:
    compound_heads = {
        value.rsplit(" ", 1)[-1]
        for value in nouns
        if " " in value and value.split(" ", 1)[0] not in COLOR_AND_SIZE_WORDS
    }
    return [value for value in nouns if " " in value or value not in compound_heads]


def all_annotation_noun_prompts(video_path: Path, video_root: Path) -> list[str]:
    """Return the broad dataset/meta/annotation prompt set used by experiment v5."""
    dataset_name = dataset_name_from_video(video_path, video_root)
    dataset_dir = video_root / dataset_name
    nouns = []

    def add(value: str) -> None:
        if value and value not in nouns:
            nouns.append(value)

    for value in semantic_prompts_from_dataset(dataset_name, max_prompts=10_000):
        add(value)
    episode_index = _episode_index_from_video(video_path)
    for text in _dataset_texts(str(dataset_dir.resolve()), episode_index):
        for value in semantic_nouns_from_text(text):
            add(value)
    nouns = _deduplicate_compound_nouns(nouns)
    if not nouns:
        raise ValueError(f"No physical nouns found for dataset: {dataset_name}")
    return nouns


def _episode_index_from_video(video_path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", video_path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _task_texts_for_episode(dataset_dir: Path, episode_index: int | None) -> list[str]:
    """Read only explicit task fields, selecting the current episode when possible."""
    texts: list[str] = []
    tasks_path = dataset_dir / "meta/tasks.jsonl"
    if tasks_path.is_file():
        try:
            for line in tasks_path.open():
                row = json.loads(line)
                if isinstance(row.get("task"), str):
                    texts.append(row["task"])
        except (OSError, json.JSONDecodeError):
            pass
    episodes_path = dataset_dir / "meta/episodes.jsonl"
    if episodes_path.is_file():
        try:
            for line in episodes_path.open():
                row = json.loads(line)
                if episode_index is not None and row.get("episode_index") != episode_index:
                    continue
                tasks = row.get("tasks", [])
                if isinstance(tasks, list):
                    texts.extend(value for value in tasks if isinstance(value, str))
                if episode_index is not None:
                    break
        except (OSError, json.JSONDecodeError):
            pass
    return texts


def semantic_noun_prompts(video_path: Path, video_root: Path) -> list[str]:
    """Return ``object`` plus nouns from names, meta, scenes, and subtasks."""
    try:
        nouns = all_annotation_noun_prompts(video_path, video_root)
    except ValueError:
        nouns = []
    return ["object", *nouns]


def annotation_context_for_video(
    video_path: Path, video_root: Path, *, max_items: int = 8, max_chars: int = 500,
) -> list[str]:
    """Return compact human-language context for contextual noun review."""
    dataset_name = dataset_name_from_video(video_path, video_root)
    dataset_dir = video_root / dataset_name
    episode_index = _episode_index_from_video(video_path)
    context = []
    for value in _dataset_texts(str(dataset_dir.resolve()), episode_index):
        normalized = " ".join(value.split())[:max_chars]
        if normalized and normalized not in context:
            context.append(normalized)
        if len(context) >= max_items:
            break
    return context


def combined_semantic_prompt(video_path: Path, video_root: Path) -> str:
    """Return the former comma-joined prompt for A/B comparison only."""
    return ", ".join(all_annotation_noun_prompts(video_path, video_root))
