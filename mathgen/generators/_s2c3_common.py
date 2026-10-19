"""Shared helpers for the second-year Fourier, PDE and Laplace builds (the ``s2c3_*`` generators).

Every numeric answer in these builds is a decimal typed to a stated number of places, so the helpers here keep three
things together: rounding half-up on an exact value, printing at EXACTLY the stated places, and building wrong
options that stay at least two units in the last place away from the key (the grader accepts half a unit either
side, so a wrong option must clear that by construction).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..answers import Distractor
from ..core import (
    check,
    join_terms,
    numeric_answer,
    pick_distractors,
    rounds_unambiguously,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

PI = math.pi


def places_text(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


def exact_float(value: float) -> Fraction:
    """The exact rational of a float's shortest repr (so 0.1 is 1/10, not the binary neighbour)."""
    return Fraction(repr(float(value)))


def rounded(value: float, dp: int) -> Fraction:
    exact = exact_float(value)
    return round_half_up(exact, dp) if exact >= 0 else -round_half_up(-exact, dp)


def fixed(value: float, dp: int) -> str:
    """`value` written at exactly `dp` decimal places (rounded half up away from zero)."""
    r = rounded(value, dp)
    text = stated_precision_string(abs(r), dp)
    return ("-" if r < 0 else "") + text


def key_ok(value: float, dp: int, smallest: float = 0.0) -> bool:
    """True when `value` is a safe key: finite, away from a rounding tie, and at least `smallest` in size."""
    return math.isfinite(value) and abs(value) >= smallest and rounds_unambiguously(value, dp, 0.08)


def number_part(value: float, dp: int) -> dict:
    """The answer fields of a typed decimal answer."""
    text = fixed(value, dp)
    return dict(answer_kind="number", answer=numeric_answer(rounded(value, dp)), answer_text=text)


def dist(value: float, dp: int, tag: str) -> Distractor:
    return Distractor(fixed(value, dp), tag, numeric_answer(rounded(value, dp)))


def wrong_numbers(key: float, dp: int, candidates, n: int = 4, floor: float | None = None) -> list[Distractor]:
    """Distractors for a decimal answer: `candidates` is a list of ``(value, misconception_id)``. A candidate stays
    only when it is finite, differs from the key by at least two units in the last place, and is not a repeat."""
    unit = Fraction(1, 10**dp)
    key_r = rounded(key, dp)
    out = []
    for value, tag in candidates:
        if value is None or not math.isfinite(value) or abs(value) > 1e7:
            continue
        if floor is not None and value < floor:
            continue
        if abs(rounded(value, dp) - key_r) < 2 * unit:
            continue
        out.append(dist(value, dp, tag))
    return pick_distractors(fixed(key, dp), out, n, tolerance=stated_precision_tolerance(dp) * 2)


def need(distractors: list, minimum: int = 2) -> list:
    check(len(distractors) >= minimum, "too few distractors")
    return distractors


def simpson(fn, lo: float, hi: float, steps: int = 4000) -> float:
    """Composite Simpson's rule on a vectorised callable: the independent numerical check of an integral."""
    if hi <= lo:
        return 0.0
    steps += steps % 2
    x = np.linspace(lo, hi, steps + 1)
    y = fn(x)
    h = (hi - lo) / steps
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def plain(x: float) -> str:
    """A short number for a stem: integers bare, others at their shortest exact decimal."""
    if float(x) == int(x):
        return str(int(x))
    return repr(round(float(x), 6))


# -- question text and LaTeX written together ------------------------------------------------------------------------

from ..core import _escape_prose  # noqa: E402


def say(*items) -> tuple[str, str]:
    """``(text, latex)`` for a sentence. A ``str`` item is prose; a ``(plain, tex)`` tuple is a maths fragment, shown
    as plain Unicode in the text and as the given TeX in the LaTeX version."""
    plain_parts: list[str] = []
    tex_parts: list[str] = []
    for item in items:
        if isinstance(item, tuple):
            plain_parts.append(item[0])
            tex_parts.append(item[1] + "{}")
        elif item:
            plain_parts.append(item)
            tex_parts.append(r"\text{" + _escape_prose(item) + "}")
    return "".join(plain_parts), "".join(tex_parts)


def prompt_fields(*items) -> dict:
    text, latex = say(*items)
    return dict(prompt=text, prompt_latex=latex)


def stem_fields(*items) -> dict:
    text, latex = say(*items)
    return dict(text=text, latex=latex)


def linear_text(c0, c1, var: str = "x") -> str:
    """c0 + c1 var as a learner writes it (constant first): `3 - 2x`, `x`, `4 - x`, `5`."""
    return join_terms([(c0, ""), (c1, var)])
