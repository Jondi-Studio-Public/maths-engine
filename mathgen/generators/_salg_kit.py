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


class PolarPair:
    """A modulus and an argument in degrees, answered as the pair (r, θ).

    The answer text is the bare pair "(128, 330)", not "r = 128, theta = 330": a labelled text makes the grader insist
    on the exact label spelling, and a learner may write the symbol θ or the word theta. A bare pair accepts either,
    with or without "r =" and "θ =", and still rejects the two values in the wrong order."""

    def __init__(self, r: int, theta: int):
        self.r, self.theta = int(r), int(theta)

    def answer_dict(self) -> dict:
        return {"r": float(self.r), "theta": float(self.theta)}

    def answer_text(self) -> str:
        return f"({self.r}, {self.theta})"


def polar_pair(r: int, theta: int) -> PolarPair:
    """A modulus and an argument in degrees as the pair the polar-form skills answer with."""
    return PolarPair(r, theta)


def polar_dist(r: int, theta: int, tag: str) -> Distractor:
    """A wrong (r, theta) as a tagged distractor."""
    pair = polar_pair(r, theta)
    return Distractor(pair.answer_text(), tag, pair.answer_dict())
