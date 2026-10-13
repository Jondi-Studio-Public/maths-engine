"""Small helpers shared by the `exd_x_u6_*` exam-tier generators. Registers no topic."""

from __future__ import annotations

import re
from fractions import Fraction

from ..answers import Distractor

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[Fraction]:
    """Every number written in `text`, as exact fractions (so 0.50 and 0.5 are one number)."""
    return {Fraction(t) for t in _NUMBER_RE.findall(text)}


def keys_clear(text: str, keys) -> bool:
    """True when no part's answer already appears as a number in the stem or the part prompts."""
    shown = numbers_in(text)
    return all(Fraction(key) not in shown for key in keys)


def tagged(text: str, misconception: str, value) -> Distractor:
    """A `Distractor` whose value is held as an exact `Fraction`."""
    return Distractor(text, misconception, Fraction(value))


def within_unit_interval(candidates) -> list:
    """The candidates whose value is a probability (0 to 1 inclusive); untagged candidates are dropped."""
    return [c for c in candidates if c.value is not None and 0 <= Fraction(c.value) <= 1]
