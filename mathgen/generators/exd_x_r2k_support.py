"""Shared building blocks for the multi-part statistics exam generators in this family.

Not a generator. It holds the small pieces every chain needs: a rounded-number part whose wrong answers are tagged
with misconception ids, a multiple-choice part, a boundary test that keeps a rounded value stable whichever way a
learner reaches it, and the normal tail probabilities used by the sampling chains.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    attempts,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

PROBABILITY = (0, 1)
POSITIVE = (0, None)


def rd(value, dp: int) -> Fraction:
    """`value` rounded half up to `dp` decimal places, as an exact Fraction."""
    return round_half_up(Fraction(value), dp)


def tx(value, dp: int) -> str:
    """`value` rounded to `dp` places and written with exactly `dp` places."""
    return stated_precision_string(rd(value, dp), dp)


def clear_of_boundary(value, dp: int, margin: Fraction = Fraction(1, 20)) -> bool:
    """True when `value` is further than `margin` of a unit in the last place from a rounding boundary."""
    offset = (Fraction(value) * 10**dp) % 1
    return abs(offset - Fraction(1, 2)) > margin


def normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def normal_two_tail(z: float) -> float:
    """P(|Z| > |z|) for a standard normal Z."""
    return math.erfc(abs(z) / math.sqrt(2))


def exact_from_float(x: float, dp: int) -> Fraction:
    """A float probability rounded half up to `dp` places, through its shortest decimal form (not its binary one)."""
    return round_half_up(Fraction(repr(float(x))), dp)


def number_part(
    prompt: str,
    value,
    dp: int,
    marks: int,
    working: list[str],
    wrongs,
    *,
    follow_through: str | None = None,
    sensible_range=None,
    wanted: int = 3,
) -> dict | None:
    """A `number` part whose key is `value` rounded to `dp` places.

    `wrongs` is a list of (value, misconception id) in priority order. Each is rounded to the part's precision and
    dropped when the real grader would take it for the key. Returns None when fewer than two distinct wrong answers
    survive, so a caller can redraw."""
    key = rd(value, dp)
    key_text = stated_precision_string(key, dp)
    tol = stated_precision_tolerance(dp)
    candidates = []
    for wrong, tag in wrongs:
        if wrong is None:
            continue
        shown = rd(wrong, dp)
        if abs(shown - key) <= tol:
            continue
        candidates.append(Distractor(stated_precision_string(shown, dp), tag, shown))
    picked = pick_distractors(key_text, candidates, wanted, tolerance=tol)
    if len(picked) < 2:
        return None
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def fraction_part(
    prompt: str,
    value,
    marks: int,
    working: list[str],
    wrongs,
    *,
    follow_through: str | None = None,
    sensible_range=None,
    wanted: int = 3,
) -> dict | None:
    """A `fraction` part whose key is the exact rational `value` (an int, Fraction or SymPy Rational)."""
    key = Fraction(int(value.p), int(value.q)) if hasattr(value, "p") else Fraction(value)
    key_text = rational_text(key)
    candidates = []
    for wrong, tag in wrongs:
        if wrong is None:
            continue
        shown = Fraction(int(wrong.p), int(wrong.q)) if hasattr(wrong, "p") else Fraction(wrong)
        if shown == key:
            continue
        candidates.append(Distractor(rational_text(shown), tag, shown))
    picked = pick_distractors(key_text, candidates, wanted)
    if len(picked) < 2:
        return None
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer={"p": key.numerator, "q": key.denominator},
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def word_part(prompt: str, key: str, wrongs, marks: int, working: list[str]) -> dict:
    """A part answered by one typed word or phrase from a list the prompt spells out.

    `key` is the right phrase and `wrongs` a list of (phrase, misconception id or None). It is graded as a
    one-item ordered list, which compares the phrase case-insensitively, so it is marked part by part from its own
    `answer_text` like every other part."""
    seen = {key}
    options = []
    for text, tag in wrongs:
        if text in seen:
            continue
        seen.add(text)
        options.append(Distractor(text, tag))
    return dict(
        prompt=prompt,
        answer_kind="ordered_list",
        answer=[key],
        answer_text=key,
        answer_latex="plain:" + key,
        marks=marks,
        working=working,
        distractors=options,
    )


def retry_build(rng, builder, what: str, tries: int = 200) -> dict:
    """The first question `builder(rng)` returns (it returns None for a draw whose parts lack distractors)."""
    for _ in attempts(tries, what):
        built = builder(rng)
        if built is not None:
            return built
    raise AssertionError("unreachable")
