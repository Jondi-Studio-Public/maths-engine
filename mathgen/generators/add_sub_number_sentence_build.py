"""Generates `add_sub_number_sentence` questions: pick the number sentence that matches an addition or
subtraction story.

Scope: given a one- or two-digit addition or subtraction word problem, select from among 3-5 candidate
number sentences, written in a stated part-part-whole form with the unknown part or whole shown as a
blank, the one that correctly represents the situation.

## What a learner is shown

A short story, then the instruction that every option is written as `part + part = whole` or
`whole - part = part`, with a square blank for the number to find. The options are number sentences such as
`14 + □ = 31` or `31 - 14 = □`. The learner is never asked to find the blank's value; the graded step is
choosing the sentence that matches the story's parts and whole.

## Why options of the other kind cannot be equivalent to the key

`14 + □ = 31`, `□ + 14 = 31`, `31 - 14 = □` and `31 - □ = 14` all say the same thing, so offering any of
them beside the key would give a question with two correct options. Every candidate sentence is therefore
solved for its blank. A candidate whose blank takes the story's answer is a correct way to write the story
and is never offered as a distractor; only sentences whose blank takes a different value, or has no whole-
number value at all, can appear. Two options that are the same sentence with the parts in the other order
(`14 + 31 = □` and `31 + 14 = □`) are one sentence, and only one is offered.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: the unknown is the END RESULT of the story (a total of two parts, or what is left after some are
  taken away), numbers to 20. DECISION: the story's own action names the operation.
- medium: the unknown is the CHANGE (how many were added, how many were taken), numbers to 50. DECISION:
  the end state is given, so the learner must see the missing quantity as one part of a whole and place the
  blank there, not just apply the story's verb.
- hard: the unknown is the START of the story, or the story compares two amounts; numbers to 99. DECISION:
  the story's verb points at the wrong operation ("was given more" asks for a subtraction, "gave away" for
  an addition), so the learner must work out which quantity is the whole before choosing a sentence.

## Independent check for every draw

The learner receives option strings, so the check parses the option strings themselves. `_solve_text` reads a
rendered sentence and solves it for its blank by its own rules; the key must solve to the story's answer, and
every distractor must either not solve to a positive whole number or solve to something else.
`check_equivalent` compares the key's value with the answer worked out from the story's numbers, and
every option's canonical form is checked to be different.

The oracle parses `question.text` and every option string, recomputes the story's answer from the
quantities in the text, and never imports this module.
"""

from __future__ import annotations

import re
from itertools import permutations

from ..core import check, check_distinct, check_equivalent, generator, phrase
from ..slots import NAMES, OBJECTS

TOPIC = "add_sub_number_sentence"
_TIERS = ("easy", "medium", "hard")
BLANK = "□"

_INSTRUCTION = (
    "Which number sentence matches the story? {blank} is the number to find.",
    "Choose the number sentence that matches the story. {blank} is the number to find.",
    "Which of these number sentences fits the story? {blank} is the unknown number.",
)

# -- stories --------------------------------------------------------------------------------------------------
# Every template names its known numbers in the order they are mentioned; `ka` is the first and `kb` the second.

_COMBINE_TOTAL = (
    "{n} has {ka} {items} in one bag and {kb} {items} in another bag. How many {items} does {n} have altogether?",
    "There are {ka} {items} on a table and {kb} {items} on a shelf. How many {items} are there in all?",
    "{n} has {ka} {items}. A friend gives {n} {kb} more. How many {items} does {n} have in total?",
)
_TAKE_REMAINING = (
    "{n} had {ka} {items}. {n} gave {kb} {items} to a friend. How many {items} does {n} have now?",
    "There were {ka} {items} in a box. {kb} {items} were taken out. How many {items} are left in the box?",
    "{n} had {ka} {items} and lost {kb} of them. How many {items} does {n} have left?",
)
_GROUP_SPLIT = (
    "A garden has {ka} flowers. {kb} of the flowers are red and the rest are yellow. How many flowers are yellow?",
    "There are {ka} children in a class. {kb} of the children walk to school and the rest ride. "
    "How many children ride?",
    "A car park has {ka} cars. {kb} of the cars are blue and the rest are white. How many cars are white?",
)
_JOIN_CHANGE = (
    "{n} had {ka} {items}. Then {n} was given some more. Now {n} has {kb} {items}. How many {items} was {n} given?",
    "{n} had {ka} {items}. {n} found some more {items}. Now {n} has {kb} {items}. How many {items} did {n} find?",
    "There were {ka} {items} in a box. Some more {items} were put in. Now there are {kb} {items} in the box. "
    "How many {items} were put in?",
)
_SEPARATE_CHANGE = (
    "{n} had {ka} {items}. {n} gave some away. Now {n} has {kb} {items}. How many {items} did {n} give away?",
    "There were {ka} {items} in a box. Some {items} were taken out. Now there are {kb} {items} in the box. "
    "How many {items} were taken out?",
    "{n} had {ka} {items}. {n} lost some of them. Now {n} has {kb} {items}. How many {items} did {n} lose?",
)
_START_JOIN = (
    "{n} had some {items}. Then {n} was given {ka} more. Now {n} has {kb} {items}. "
    "How many {items} did {n} start with?",
    "There were some {items} in a box. {ka} more {items} were put in. Now there are {kb} {items} in the box. "
    "How many {items} were in the box at first?",
)
_START_SEPARATE = (
    "{n} had some {items}. {n} gave away {ka} {items}. Now {n} has {kb} {items} left. "
    "How many {items} did {n} start with?",
    "There were some {items} in a box. {ka} {items} were taken out. Now there are {kb} {items} in the box. "
    "How many {items} were in the box at first?",
)
_COMPARE_DIFFERENCE = (
    "{n} has {ka} {items}. {m} has {kb} {items}. How many more {items} does {n} have than {m}?",
    "{n} has {ka} {items} and {m} has {kb} {items}. How many fewer {items} does {m} have than {n}?",
)
_COMPARE_MORE = (
    "{n} has {ka} {items}. {m} has {kb} more {items} than {n}. How many {items} does {m} have?",
    "{m} has {kb} more {items} than {n}, who has {ka} {items}. How many {items} does {m} have?",
)

# kind -> (templates, which of the two draws is the unknown). The two draws are the parts `p` and `q`; the whole
# is `p + q`. "unknown" says which role the blank plays in the story.
#   total   -> the whole is unknown; the story gives both parts (ka, kb)
#   part    -> one part is unknown; the story gives the whole and the other part
_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "combine_total": (_COMBINE_TOTAL, "total"),
    "take_remaining": (_TAKE_REMAINING, "part"),
    "group_split": (_GROUP_SPLIT, "part"),
    "join_change": (_JOIN_CHANGE, "part"),
    "separate_change": (_SEPARATE_CHANGE, "part"),
    "start_join": (_START_JOIN, "part"),
    "start_separate": (_START_SEPARATE, "total"),
    "compare_difference": (_COMPARE_DIFFERENCE, "part"),
    "compare_more": (_COMPARE_MORE, "total"),
}

# Which story kinds each tier draws, the largest whole it may use and the smallest part.
_TIER_KINDS = {
    "easy": (("combine_total", "take_remaining", "group_split"), 20, 2),
    "medium": (("join_change", "separate_change"), 50, 4),
    "hard": (("start_join", "start_separate", "compare_difference", "compare_more"), 99, 6),
}


# -- sentences ------------------------------------------------------------------------------------------------
# A sentence is (op, x, y, z) meaning `x op y = z`; each slot holds an int or None (the blank).

_SENTENCE_RE = re.compile(rf"^(\d+|{BLANK}) ([+-]) (\d+|{BLANK}) = (\d+|{BLANK})$")


def _render(sentence) -> str:
    op, x, y, z = sentence
    cells = [BLANK if c is None else str(c) for c in (x, y, z)]
    return f"{cells[0]} {op} {cells[1]} = {cells[2]}"


def _solve_cells(op: str, x, y, z) -> int | None:
    """The whole-number value (>= 1) that makes `x op y = z` true when exactly one cell is None, else None."""
    if op == "+":
        if z is None:
            value = x + y
        elif x is None:
            value = z - y
        else:
            value = z - x
    else:
        if x is None:
            value = y + z
        elif y is None:
            value = x - z
        else:
            value = x - y
    return value if value >= 1 else None


def _solve_text(sentence_text: str) -> int | None:
    """Solve a RENDERED sentence for its blank. Reads the string the learner sees, not the tuple that made it."""
    match = _SENTENCE_RE.match(sentence_text)
    check(match is not None, f"{TOPIC}: option {sentence_text!r} is not a well-formed sentence")
    x, op, y, z = match.groups()
    cells = [None if c == BLANK else int(c) for c in (x, y, z)]
    check(cells.count(None) == 1, f"{TOPIC}: option {sentence_text!r} must have exactly one blank")
    return _solve_cells(op, *cells)


def _canonical(sentence):
    """The same sentence with its parts in either order is one sentence: `a + b = c` and `b + a = c`; `c - a = b`
    and `c - b = a`."""
    op, x, y, z = sentence

    def key(cell):
        return -1 if cell is None else cell

    if op == "+":
        return ("+", tuple(sorted((x, y), key=key)), z)
    return ("-", x, tuple(sorted((y, z), key=key)))


def _all_sentences(ka: int, kb: int) -> list[tuple]:
    return [(op, x, y, z) for op in "+-" for (x, y, z) in permutations((ka, kb, None))]


def _classes(ka: int, kb: int, answer: int):
    """Group the canonical sentences over (ka, kb, blank) into those that match the answer and the two kinds that
    do not: a different positive value, and no whole-number value at all."""
    groups: dict[tuple, list[tuple]] = {}
    for sentence in _all_sentences(ka, kb):
        groups.setdefault(_canonical(sentence), []).append(sentence)
    matching, wrong_value, impossible = [], [], []
    for members in groups.values():
        value = _solve_cells(*members[0])
        if value is None:
            impossible.append(members)
        elif value == answer:
            matching.append(members)
        else:
            wrong_value.append(members)
    return matching, wrong_value, impossible


# -- drawing -------------------------------------------------------------------------------------------------


def _draw_parts(rng, cap: int, lo: int) -> tuple[int, int]:
    """Two distinct parts, each at least `lo`, whose whole is at most `cap` and at least 6."""
    for _ in range(500):
        p = rng.randint(lo, cap - lo)
        q = rng.randint(lo, cap - p)
        if p != q and 6 <= p + q <= cap:
            return p, q
    check(False, f"{TOPIC}: could not draw two parts under {cap}")


def _known_and_answer(kind: str, p: int, q: int, flip: bool) -> tuple[int, int, int]:
    """(ka, kb, answer) for the story `kind`, with `p` and `q` the two parts and `p + q` the whole."""
    if flip:
        p, q = q, p
    whole = p + q
    if kind == "combine_total":
        return p, q, whole
    if kind == "take_remaining" or kind == "group_split":
        return whole, p, q
    if kind == "join_change":
        return p, whole, q
    if kind == "separate_change":
        return whole, p, q
    if kind == "start_join":
        return p, whole, q
    if kind == "start_separate":
        return p, q, whole
    if kind == "compare_difference":
        return whole, p, q
    if kind == "compare_more":
        return p, q, whole
    raise ValueError(f"unknown story kind {kind!r}")


def _story_answer(kind: str, ka: int, kb: int) -> int:
    """The unknown quantity worked out from the story's own two numbers, as the story is told (not from the draw)."""
    if kind in ("combine_total", "start_separate", "compare_more"):
        return ka + kb
    if kind in ("take_remaining", "group_split", "compare_difference", "separate_change"):
        return ka - kb
    if kind in ("join_change", "start_join"):
        return kb - ka
    raise ValueError(f"unknown story kind {kind!r}")


def _build(rng, tier: str = "easy") -> dict:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    kinds, cap, lo = _TIER_KINDS[tier]
    kind = rng.choice(kinds)
    templates, _unknown = _KINDS[kind]
    p, q = _draw_parts(rng, cap, lo)
    ka, kb, drawn_answer = _known_and_answer(kind, p, q, rng.random() < 0.5)
    answer = _story_answer(kind, ka, kb)
    check_equivalent(answer, drawn_answer, f"{TOPIC} {tier}: the story's answer disagrees with the drawn parts")
    check(ka != kb and answer not in (ka, kb), f"{TOPIC} {tier}: a repeated number would blur the options")

    name, other = rng.sample(NAMES, 2)
    _singular, items = rng.choice(OBJECTS)
    story = phrase(rng, templates, n=name, m=other, items=items, ka=ka, kb=kb)
    text = story + " " + phrase(rng, _INSTRUCTION, blank=BLANK)

    matching, wrong_value, impossible = _classes(ka, kb, answer)
    check(len(matching) >= 1 and len(wrong_value) >= 2 and len(impossible) >= 2, f"{TOPIC}: option pools too small")
    key_sentence = rng.choice(rng.choice(matching))
    n_options = rng.choice((3, 4, 4, 5))
    n_distractors = n_options - 1
    n_wrong = {2: 1, 3: rng.randint(1, 2), 4: 2}[n_distractors]
    n_impossible = n_distractors - n_wrong
    chosen = rng.sample(wrong_value, n_wrong) + rng.sample(impossible, n_impossible)
    distractor_sentences = [rng.choice(members) for members in chosen]

    answer_text = _render(key_sentence)
    distractors = [_render(s) for s in distractor_sentences]

    # The learner sees strings, so the checks read strings.
    check_equivalent(_solve_text(answer_text), answer, f"{TOPIC} {tier}: the key does not solve to {answer}")
    for option in distractors:
        value = _solve_text(option)
        if value is not None:
            check_distinct(answer, value, msg=f"{TOPIC} {tier}: distractor {option!r} also solves to the answer")
    canonical_forms = [_canonical(s) for s in [key_sentence, *distractor_sentences]]
    check(len(set(canonical_forms)) == len(canonical_forms), f"{TOPIC} {tier}: two options are the same sentence")
    check(len(set([answer_text, *distractors])) == len(distractors) + 1, f"{TOPIC}: duplicate option text")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        working=[
            "Decide which numbers in the story are the parts and which is the whole.",
            f"The sentence {answer_text} puts the unknown number ({BLANK}) in its place; the unknown is {answer}.",
        ],
        distractors=distractors,
    )


@generator(
    TOPIC,
    level=2,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def add_sub_number_sentence(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
