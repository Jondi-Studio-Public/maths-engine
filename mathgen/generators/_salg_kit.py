"""Small helpers shared by the senior algebra and trigonometry builds (binomial expansion, reciprocal
trigonometric functions and the polar form of complex numbers): superscripts for the stem, typed-answer spelling
for SymPy expressions, and an exact-fraction distractor."""

from __future__ import annotations

import re
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import rational_text

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sup(n: int) -> str:
    """`n` as superscript digits: 4 -> ⁴."""
    return str(n).translate(_SUPERSCRIPT)


def typed(expr) -> str:
    """A SymPy expression spelled the way a learner types it: `**` becomes `^`, `*` is kept."""
    return str(expr).replace("**", "^")


def shown(text: str) -> str:
    """A typed expression as the working shows it: "8*x + 3*y" -> "8x + 3y"."""
    return re.sub(r"\*(?=[a-zA-Z(])", "", text)


def frac_dist(value, tag: str) -> Distractor:
    """A wrong fraction (or whole number) as a tagged distractor with its exact value."""
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def int_dist(value: int, tag: str) -> Distractor:
    return Distractor(str(int(value)), tag, int(value))
