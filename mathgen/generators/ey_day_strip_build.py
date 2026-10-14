"""Early-years sequencing of the day: picture cards drawn with the ``day_strip`` figure kind.

``ey_day_strip`` (skill ``ey-day-strip-sequence``, ``answer: number``). Every strip has numbered slots
(1, 2, 3 ... from the left), and the learner answers with a slot number, so the answer is never a
word written on a card.

- easy: the cards are already in day order; name the card just after, or just before, a named one.
- medium: the cards are scrambled; name the one card that happens in a stated part of the day
  (morning, lunchtime, afternoon or night), so each card's time of day must be known.
- hard: the cards are scrambled; name the card that comes first, second, third, fourth or last once
  the day is put in order, so the learner sorts before counting. The slot number is not the place.

A strip never shows two events whose order or time of day a child could fairly argue (sunrise and
waking up, bed and the moon, school and lunch). The figure reader re-derives the answer from the
drawn icons and the prompt alone.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, phrase, pick_distractors
from ..figure_kinds.calendar import (
    CONFLICTS,
    DAY_PARTS,
    ICON_LABELS,
    ICON_ORDER,
    ICON_PART,
    day_order,
    day_strip,
    slot_of,
    slots_in_part,
)
from ..slots import NAMES

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_NEIGHBOUR_TEXT = (
    "What number is the card {d} {label}?",
    "Which card comes {d} {label}? Say its number.",
    "{n} looks at the day. What number is the card {d} {label}?",
)
_WHEN = {
    "morning": "in the morning",
    "lunchtime": "at lunchtime",
    "afternoon": "in the afternoon",
    "night": "at night",
}
_PART_TEXT = (
    "What number is the card that happens {when}?",
    "Which number card happens {when}?",
    "{n} does one thing {when}. What number is its card?",
)
_ORDINALS = ("first", "second", "third", "fourth", "fifth")
_ORDER_TEXT = (
    "In day order, which number card comes {o}?",
    "What number card is {o} in the day?",
    "Put the day in order. Which number card comes {o}?",
)
_ORD_RE = re.compile(r"\b(first|second|third|fourth|fifth|last)\b")
_NEIGHBOUR_RE = re.compile(r"\b(after|before) ([A-Za-z ]+?)\?")


def _ok(icons) -> bool:
    return not any(pair <= set(icons) for pair in CONFLICTS)


def _sorted_icons(rng, n: int) -> list[str]:
    for _ in attempts(500, "a strip with one clear order"):
        icons = rng.sample(ICON_ORDER, n)
        if _ok(icons):
            return sorted(icons, key=ICON_ORDER.index)
    raise AssertionError("unreachable")


def _scrambled(rng, icons: list[str]) -> list[str]:
    ordered = sorted(icons, key=ICON_ORDER.index)
    for _ in attempts(200, "a scramble that is not the day order"):
        mixed = rng.sample(icons, len(icons))
        # A scramble must put the cards out of order in at least two places, so the position is no clue.
        if sum(a != b for a, b in zip(mixed, ordered)) >= 3:
            return mixed
    raise AssertionError("unreachable")


def _build_easy(rng):
    n = rng.randint(3, 4)
    icons = _sorted_icons(rng, n)
    after = rng.random() < 0.5
    i = rng.randint(0, n - 2) if after else rng.randint(1, n - 1)
    named = icons[i]
    target = i + 1 if after else i - 1
    fig = day_strip(cards=icons, numbered=True, alt="picture cards of the day in order, numbered")
    cands = [c for c in (i + 1, (i - 1 if after else i + 1) + 1) if 1 <= c <= n]
    cands += [s for s in range(1, n + 1)]
    return fig, "after" if after else "before", ICON_LABELS[named], target + 1, cands


def _build_medium(rng):
    n = rng.randint(4, 5)
    part = rng.choice(DAY_PARTS)
    for _ in attempts(500, "a strip with one card in the part of the day"):
        key = rng.choice([i for i in ICON_ORDER if ICON_PART[i] == part])
        others = rng.sample([i for i in ICON_ORDER if ICON_PART[i] != part], n - 1)
        icons = [key, *others]
        if _ok(icons):
            break
    mixed = _scrambled(rng, icons)
    fig = day_strip(cards=mixed, numbered=True, alt="picture cards of the day, scrambled and numbered")
    (answer,) = slots_in_part(fig, part)
    # Confusions: the cards next to the key in day order (the neighbouring times of day).
    by_day = day_order(fig)
    at = by_day.index(answer)
    cands = [by_day[j] for j in (at - 1, at + 1, at - 2, at + 2) if 0 <= j < n]
    return fig, part, answer, cands


def _build_hard(rng):
    n = rng.randint(4, 6)
    icons = _sorted_icons(rng, n)
    mixed = _scrambled(rng, icons)
    fig = day_strip(cards=mixed, numbered=True, alt="picture cards of the day, scrambled and numbered")
    by_day = day_order(fig)
    k = rng.randint(2, n - 1) if rng.random() < 0.8 else n  # the 1st and the last are the easy places
    if rng.random() < 0.15:
        k = 1
    answer = by_day[k - 1]
    ordinal = "last" if k == n else _ORDINALS[k - 1]
    cands = [k, by_day[-k], by_day[k] if k < n else by_day[0], by_day[k - 2] if k > 1 else by_day[1]]
    return fig, ordinal, answer, cands


@generator("ey_day_strip", level=1, presets=_PRESETS)
def ey_day_strip(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    if tier == "easy":
        fig, direction, label, answer, cands = _build_easy(rng)
        text = phrase(rng, _NEIGHBOUR_TEXT, n=name, d=direction, label=label)
        working = f"Put the cards in day order. The card {direction} {label} is number {answer}."
    elif tier == "medium":
        fig, part, answer, cands = _build_medium(rng)
        text = phrase(rng, _PART_TEXT, n=name, when=_WHEN[part])
        working = f"Find the card that happens {_WHEN[part]}. It is in slot {answer}."
    else:
        fig, ordinal, answer, cands = _build_hard(rng)
        text = phrase(rng, _ORDER_TEXT, o=ordinal)
        working = f"Put the cards in the order of the day. The {ordinal} one is in slot {answer}."
    distractors = pick_distractors(str(answer), [str(c) for c in cands])
    check(len(distractors) >= 2, "too few distractors")
    answer_text = str(answer)
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_strip(fig, text) == answer_text, "the figure does not show the keyed card")
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[working],
        figure=fig,
    )


def _read_strip(fig: dict, text: str) -> str:
    by_day = day_order(fig)
    ordinal = _ORD_RE.search(text)
    if ordinal:
        word = ordinal.group(1)
        return str(by_day[-1] if word == "last" else by_day[_ORDINALS.index(word)])
    for part, when in _WHEN.items():
        if when in text:
            slots = slots_in_part(fig, part)
            check(len(slots) == 1, "the part of the day must match exactly one card")
            return str(slots[0])
    near = _NEIGHBOUR_RE.search(text)
    check(near is not None, "no neighbour question in the prompt")
    at = by_day.index(slot_of(fig, near.group(2)))
    return str(by_day[at + (1 if near.group(1) == "after" else -1)])


figures.register_reader("ey_day_strip", lambda q: _read_strip(q.figure, q.text))
