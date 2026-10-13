"""Shared helpers for the first-year single-variable calculus builds (the `fy1ca_*_build` modules).

What lives here is only what several of those modules need: building tagged distractors for a fraction, number or
expression answer, assembling the parts of an exam chain, a small numerical integrator for the second-route check
of every answer, and a few text formatters. Nothing here registers a generator.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy
from sympy import Rational

from ..answers import Distractor, exact_value
from ..core import (
    VerificationError,
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)

# -- numbers -----------------------------------------------------------------------------------------------------


def frac(value) -> Fraction:
    """`value` (an int, Fraction or SymPy Rational) as a Fraction."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(int(value.p), int(value.q))


def ratio_text(value) -> str:
    """An exact rational as a learner types it: "3", "-7/4"."""
    return rational_text(Rational(frac(value).numerator, frac(value).denominator))


def grid(value, dp: int) -> Fraction:
    """`value` rounded half up to `dp` decimal places, as an exact Fraction."""
    return round_half_up(Fraction(value), dp)


def dec_text(value, dp: int) -> str:
    """`value` rounded to `dp` decimal places and shown with exactly that many (never fewer)."""
    return stated_precision_string(grid(value, dp), dp)


def clear_of_rounding(value: float, dp: int, margin: float = 0.08) -> bool:
    """Whether rounding `value` to `dp` places is not a coin-flip on a ".5" boundary."""
    scaled = abs(value) * 10**dp
    return abs(scaled - math.floor(scaled) - 0.5) > margin


def close(a: float, b: float, rel: float = 1e-7) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def simpson(fn, lo: float, hi: float, n: int = 1200) -> float:
    """Composite Simpson's rule: the numerical second route for every integral a generator states exactly."""
    h = (hi - lo) / n
    total = fn(lo) + fn(hi)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * fn(lo + i * h)
    return total * h / 3


def is_square(value: Fraction) -> bool:
    """Whether the non-negative rational `value` is the square of a rational."""
    value = Fraction(value)
    if value < 0:
        return False
    return (
        math.isqrt(value.numerator) ** 2 == value.numerator and math.isqrt(value.denominator) ** 2 == value.denominator
    )


def sqrt_rational(value: Fraction) -> Fraction:
    value = Fraction(value)
    check(is_square(value), f"{value} is not a rational square")
    return Fraction(math.isqrt(value.numerator), math.isqrt(value.denominator))


def typed(expr) -> str:
    """A SymPy expression as a learner types it: caret powers, `pi`, `sqrt(...)`, `ln(...)`."""
    return str(expr).replace("**", "^").replace("log(", "ln(")


# -- distractors -------------------------------------------------------------------------------------------------


def _shape(text: str) -> str:
    """A coarse look of an option, so the answer is never the only one of its kind."""
    t = text.strip()
    if t.lstrip("-").isdigit():
        return "integer"
    if "/" in t and t.lstrip("-").replace("/", "").isdigit():
        return "fraction"
    try:
        float(t)
        return "decimal"
    except ValueError:
        return "other"


def _balanced(answer_text: str, chosen: list[Distractor], pool: list[Distractor], n: int) -> list[Distractor]:
    """`chosen`, with the last option swapped for a same-shaped one from `pool` when none of them matches the answer's
    shape: an answer that is the only integer, say, can be picked without doing the work."""
    want = _shape(answer_text)
    if want == "other" or any(_shape(d.text) == want for d in chosen):
        return chosen
    for d in pool:
        if d not in chosen and _shape(d.text) == want:
            return [*chosen[: n - 1], d]
    return chosen


def fraction_distractors(answer, candidates, n: int = 3) -> list[Distractor]:
    """Tagged distractors for an answer graded as an exact rational. `candidates` is a list of (value, misconception)."""
    answer = frac(answer)
    seen = [answer]
    pool: list[Distractor] = []
    for value, tag in candidates:
        if value is None:
            continue
        try:
            value = frac(value)
        except (TypeError, ValueError, AttributeError):
            continue
        if value in seen:
            continue
        seen.append(value)
        pool.append(Distractor(ratio_text(value), tag, exact_value(Rational(value.numerator, value.denominator))))
    return _balanced(ratio_text(answer), pool[:n], pool, n)


def number_distractors(answer, candidates, dp: int, n: int = 3) -> list[Distractor]:
    """Tagged distractors for a typed number answer at `dp` places. Every kept option shows exactly `dp` places and
    lies at least one unit of the last place from the answer, so the grader (which allows half a unit) rejects it."""
    answer = grid(answer, dp)
    unit = Fraction(1, 10**dp)
    seen = [answer]
    pool: list[Distractor] = []
    for value, tag in candidates:
        if value is None:
            continue
        try:
            value = grid(Fraction(value), dp)
        except (TypeError, ValueError, OverflowError):
            continue
        if any(abs(value - s) < unit for s in seen):
            continue
        seen.append(value)
        pool.append(Distractor(stated_precision_string(value, dp), tag, numeric_answer(value)))
    return _balanced(stated_precision_string(answer, dp), pool[:n], pool, n)


def expression_distractors(answer, candidates, n: int = 3) -> list[Distractor]:
    """Tagged distractors for an answer graded as an expression (an exact value such as 4*pi/3 or ln(2))."""
    seen = [float(sympy.N(answer))]
    pool: list[Distractor] = []
    for value, tag in candidates:
        if value is None:
            continue
        value = sympy.sympify(value)
        if not value.is_finite or value.free_symbols:
            continue
        number = float(sympy.N(value))
        if any(abs(number - s) < 1e-9 * max(1.0, abs(s)) for s in seen):
            continue
        seen.append(number)
        pool.append(Distractor(typed(value), tag, str(value)))
    return pool[:n]


def check_all_wrong(answer_value: float, distractors, value_of) -> None:
    """Every distractor's value differs from the answer's (a generator-time proof, whatever the grader's tolerance)."""
    for d in distractors:
        check(
            abs(value_of(d) - answer_value) > 1e-9 * max(1.0, abs(answer_value)),
            f"distractor {d.text!r} equals the answer",
        )


# -- parts -------------------------------------------------------------------------------------------------------


def fraction_part(prompt, value, marks, working, distractors, *, follow_through=None, sensible_range=None) -> dict:
    value = frac(value)
    text = ratio_text(value)
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(Rational(value.numerator, value.denominator)),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def number_part(prompt, value, dp, marks, working, distractors, *, follow_through=None, sensible_range=None) -> dict:
    rounded = grid(value, dp)
    text = stated_precision_string(rounded, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(rounded),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def expression_part(prompt, expr, marks, working, distractors, *, follow_through=None) -> dict:
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=str(expr),
        answer_text=typed(expr),
        answer_latex=latex_answer(expr),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def choice_part(prompt, answer_text, distractors, marks, working) -> dict:
    """A multiple-choice part: the answer and its wrong options are all named in the prompt's option list."""
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        marks=marks,
        working=working,
        distractors=distractors,
    )


def positive(value) -> bool:
    return value > 0


class Redraw(Exception):
    """Raised by a tier builder when the draw cannot make a question to the standard (too few distractors, say)."""


def build_with_redraws(rng, build, what: str, tries: int = 200):
    """`build(rng)` until it returns without raising `Redraw`; the budget is bounded and consumes `rng` as drawn."""
    for _ in range(tries):
        try:
            return build(rng)
        except Redraw:
            continue
    raise VerificationError(f"no acceptable draw for {what} in {tries} tries")


def need_distractors(distractors, n: int = 3) -> None:
    if len(distractors) < n:
        raise Redraw
