"""Generates `p19a_part_part_whole` questions: given a whole of two to ten and one part, state the other part.

Scope: "Given a whole collection size from two to ten and one of
its two parts, state the value of the other part."

## BOUNDARY with `number-bonds-to-10` / `number-bonds-to-5`

Those two fix the total (10 or 5) and ask "a and what make N". This skill
varies the whole 2-10 and states the whole and one part of a COLLECTION, so
the learner must first find which stated number IS the whole before finding
the other part. The bond stem "and what make" is never used here.

## Construct-to-target joint distribution over (whole, missing part)

A naive draw -- whole uniform 2-10, then part uniform over 1..whole-1 -- puts
nearly a third of all draws on `key == 1` (small wholes have almost no other
option) or, symmetrically, nearly a third on `whole == 10` if the roles are
swapped. Simulated at 4000 draws: both naive schemes fail the <15% top-key and <30%
`whole == 10` targets.

The scheme used here draws the MISSING PART (`key`) first, uniformly over
1-9 -- this alone fixes the key marginal at ~11% each, comfortably under the
15% target, independently of anything that happens next. It then draws the
GIVEN part with a weight that favours the small end of its valid range,
which is what keeps `whole == 10` (the value that needs the given part to
land on the TOP of its range) from being over-drawn: weight(part) =
`hi + 1 - part` where `hi = 10 - key`, so `part == 1` is the most likely and
`part == hi` (the one value that produces `whole == 10`) is the least
likely. Measured at 4000 draws: top key share ~12%, `whole == 10` share
~20%, `whole == 5` share ~8% -- all comfortably inside target.

## Independent check for every draw

`check_equivalent(key, whole - part, ...)` recomputes the missing part by
subtraction -- the arithmetic a learner performs -- as a check against the
value construction drew directly, a different route from the addition
(`part + key`) that produced `whole`.

The oracle identifies the whole
and the given part from `question.text` by its own keyword rules (never
importing this module) and re-derives the missing part by subtraction.
"""

from __future__ import annotations

from ..core import check, check_equivalent, check_pairwise_distinct, generator, phrase

TOPIC = "p19a_part_part_whole"
_TIERS = ("easy", "medium", "hard")

_ITEMS = [
    ("counter", "counters"),
    ("shell", "shells"),
    ("marble", "marbles"),
    ("button", "buttons"),
    ("block", "blocks"),
]
_COLOR_PAIRS = [
    ("red", "blue"),
    ("yellow", "green"),
    ("orange", "purple"),
    ("red", "green"),
]
_CONTAINER_PAIRS = [
    ("bucket", "sand"),
    ("box", "table"),
    ("bag", "floor"),
]
_ANIMAL_PAIRS = [
    ("duck", "ducks", "goose", "geese", "birds"),
    ("cat", "cats", "dog", "dogs", "pets"),
    ("apple", "apples", "orange", "oranges", "pieces of fruit"),
]
_NAMES = ("Mia", "Sam", "Ali", "Zoe", "Ben", "Leo")


def _unit_word(n: int, sg: str, pl: str) -> str:
    return sg if n == 1 else pl


def _draw_whole_part(rng) -> tuple[int, int, int]:
    """Draws (whole, part, key) -- see module docstring for the joint
    distribution rationale. `part` is the number STATED to the learner;
    `key` is the missing part the question asks for."""
    key = int(rng.random() * 9) + 1
    hi = 10 - key
    parts = list(range(1, hi + 1))
    weights = [hi + 1 - p for p in parts]
    part = rng.choices(parts, weights=weights, k=1)[0]
    whole = part + key
    check(1 <= part < whole <= 10, "p19a_part_part_whole: whole/part out of range")
    check_equivalent(key, whole - part, "p19a_part_part_whole: key does not equal whole - part")
    return whole, part, key


def _candidates(whole: int, part: int, key: int) -> list[int]:
    """Misconception candidates, in priority order:
    - `whole + part`: adds instead of subtracting.
    - `part`: repeats the given part as if it were the answer.
    - `whole`: repeats the whole as if it were the answer.
    - `key - 1`, `key + 1`: an off-by-one slip.
    """
    return [whole + part, part, whole, key - 1, key + 1]


def _finalise(text: str, whole: int, part: int, key: int) -> dict:
    raw = _candidates(whole, part, key)
    seen = {key}
    distinct_ints: list[int] = []
    distractors: list[str] = []
    for c in raw:
        if c in seen:
            continue
        seen.add(c)
        distinct_ints.append(c)
        distractors.append(str(c))
    check(len(distractors) >= 1, "p19a_part_part_whole: no distractor survived dedup")
    check_pairwise_distinct(
        key, *distinct_ints, msg="p19a_part_part_whole: a distractor collides with the key or another distractor"
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        distractors=distractors,
        working=[f"answer: {key}"],
    )


_EASY = (
    "There are {whole} {item_pl}. {part} {part_verb} {colorA} and the rest {rest_verb} {colorB}. "
    "How many are {colorB}?",
    "{name} has {whole} {item_pl}. {part} of them {part_verb} {colorA}, and the rest {rest_verb} {colorB}. "
    "How many are {colorB}?",
    "There are {whole} {item_pl} in a jar. {part} {part_verb} {colorA}, and the rest {rest_verb} {colorB}. "
    "How many are {colorB}?",
)

_MEDIUM = (
    "Every one of the {whole} {item_pl} is either in the {containerA} or on the {containerB}. "
    "{part} {part_verb} on the {containerB}. How many are in the {containerA}?",
    "{name} has {whole} {item_pl}. {part} {part_verb} on the {containerB}. The rest {rest_verb} in "
    "the {containerA}. How many are in the {containerA}?",
    "Out of {whole} {item_pl}, {part} {part_verb} {colorA}. How many are not {colorA}?",
)

_HARD_COMBINE = (
    "Some {catA_pl} and {part} {catB_word} make {whole} {group}. How many {catA_pl}?",
    "A group of {whole} {group} is made up of some {catA_pl} and {part} {catB_word}. How many {catA_pl}?",
    "{part} {catB_word} and some {catA_pl} make {whole} {group} in total. How many {catA_pl}?",
)
_HARD_LABELLED_WHOLE_FIRST = ("Whole: {whole}. One part: {part}. Other part: ?",)
_HARD_LABELLED_WHOLE_LAST = ("One part: {part}. Whole: {whole}. Other part: ?",)


def _build_easy(rng) -> dict:
    whole, part, key = _draw_whole_part(rng)
    item_sg, item_pl = rng.choice(_ITEMS)
    colorA, colorB = rng.choice(_COLOR_PAIRS)
    name = rng.choice(_NAMES)
    part_verb = _unit_word(part, "is", "are")
    rest_verb = _unit_word(key, "is", "are")
    text = phrase(
        rng,
        _EASY,
        whole=whole,
        part=part,
        item_pl=item_pl,
        colorA=colorA,
        colorB=colorB,
        name=name,
        part_verb=part_verb,
        rest_verb=rest_verb,
    )
    return _finalise(text, whole, part, key)


def _build_medium(rng) -> dict:
    whole, part, key = _draw_whole_part(rng)
    item_sg, item_pl = rng.choice(_ITEMS)
    colorA, colorB = rng.choice(_COLOR_PAIRS)
    containerA, containerB = rng.choice(_CONTAINER_PAIRS)
    name = rng.choice(_NAMES)
    part_verb = _unit_word(part, "is", "are")
    rest_verb = _unit_word(key, "is", "are")
    text = phrase(
        rng,
        _MEDIUM,
        whole=whole,
        part=part,
        item_pl=item_pl,
        colorA=colorA,
        containerA=containerA,
        containerB=containerB,
        name=name,
        part_verb=part_verb,
        rest_verb=rest_verb,
    )
    return _finalise(text, whole, part, key)


def _build_hard(rng) -> dict:
    whole, part, key = _draw_whole_part(rng)
    combine = rng.random() < 0.5
    if combine:
        catA_sg, catA_pl, catB_sg, catB_pl, group = rng.choice(_ANIMAL_PAIRS)
        catB_word = _unit_word(part, catB_sg, catB_pl)
        text = phrase(rng, _HARD_COMBINE, whole=whole, part=part, catA_pl=catA_pl, catB_word=catB_word, group=group)
    else:
        whole_first = rng.random() < 0.5
        templates = _HARD_LABELLED_WHOLE_FIRST if whole_first else _HARD_LABELLED_WHOLE_LAST
        text = phrase(rng, templates, whole=whole, part=part)
    return _finalise(text, whole, part, key)


def _build(rng, tier: str = "easy") -> dict:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    return _build_hard(rng)


@generator(
    TOPIC,
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_part_part_whole(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
