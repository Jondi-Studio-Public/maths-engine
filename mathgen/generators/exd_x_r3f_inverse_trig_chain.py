"""Generates `exd_x_r3f_inverse_trig_chain` questions: an exam chain on a shifted inverse-trigonometric integral.

A curve y = f(x) has an expanded quadratic under it, either

    f(x) = k / (x^2 + 2hx + c)         ->  (k/a) [arctan((x + h)/a)]
    f(x) = k / sqrt(-x^2 - 2hx + c)    ->  k [arcsin((x + h)/a)]

and the exam asks, in order:

(a) complete the square: the constant m (or n) in (x + p)^2 + m (or n - (x + p)^2)    number,     2 marks
(b) the exact area under the curve between two limits                                  expression, 3 marks
(c) the mean value of f over the same interval                                         expression, 2 marks

Part (b) follows through from (a): a learner whose constant is wrong still earns (b) for integrating correctly
with their own constant. Part (c) follows through from (b).

## Construction: draw the shifted form first

The constants are drawn as (k, a, h), and the limits as x = -h + a t so that the shifted variable (x + h)/a takes
the values t_lo and t_hi, whose inverse trigonometric values are rational multiples of pi. The area is rebuilt
independently with `sympy.integrate`, so the closed form is checked, not assumed. The arcsin family keeps
t within (-1/2, 1/2) so the integrand is bounded on the whole interval and "area" is an honest description.
"""

from __future__ import annotations

import math
import random

from sympy import Rational, asin, atan

from ..answers import Distractor, exact_value
from ..core import (
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    pick_distractors,
)
from ._exd_distinct_parts import distinct_part_answers

_ARCTAN_LIMITS = [(-1, 1), (0, 1), (-1, 0)]
_ARCSIN_LIMITS = [(Rational(-1, 2), Rational(1, 2)), (0, Rational(1, 2)), (Rational(-1, 2), 0)]


def _simpson(f, lo: int, hi: int, steps: int = 2000) -> float:
    """Composite Simpson's rule: an independent numerical check on the closed-form area."""
    step = (hi - lo) / steps
    total = f(lo) + f(hi)
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * f(lo + i * step)
    return total * step / 3


def _typed(expr) -> str:
    return str(expr).replace("**", "^")


def _quadratic(lead: int, b: int, c: int) -> str:
    return join_terms([(lead, "x^2"), (b, "x"), (c, "")])


def _expr_distractors(answer, candidates: list) -> list:
    """Tagged distractors for an expression part: (expr, tag) pairs that differ from `answer`."""
    answer_text = _typed(answer)
    cands = []
    for value, tag in candidates:
        if value != answer:  # rational multiples of pi are canonical, so == is exact
            cands.append(Distractor(_typed(value), tag, _typed(value)))
    return pick_distractors(answer_text, cands, 3)


def _int_distractors(answer: int, candidates: list) -> list:
    cands = [Distractor(str(v), tag, exact_value(v)) for v, tag in candidates if v != answer and v > 0]
    return pick_distractors(str(answer), cands, 3)


@generator("exd_x_r3f_inverse_trig_chain", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_inverse_trig_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Complete the square, integrate to an exact area, then take the mean value of the function."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    arctan_family = rng.random() < 0.5
    k = rng.randint(1, 6)
    for _ in attempts(100, "an interval of length at least 2"):
        if arctan_family:
            a = rng.randint(2, 8)
            h = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            t_lo, t_hi = rng.choice(_ARCTAN_LIMITS)
        else:
            a = rng.choice([2, 4, 6, 8])
            h = rng.choice([v for v in (-3, -2, -1, 1, 2, 3) if abs(v) < a])
            t_lo, t_hi = rng.choice(_ARCSIN_LIMITS)
        if a * (t_hi - t_lo) >= 2:
            break
    check(a > abs(h) or arctan_family, "the radius must exceed the shift")
    lo, hi = int(-h + a * t_lo), int(-h + a * t_hi)
    width = hi - lo

    m = a * a
    if arctan_family:
        constant = h * h + a * a
        denominator = _quadratic(1, 2 * h, constant)
        integrand_float = lambda v: k / (v * v + 2 * h * v + constant)  # noqa: E731
        antiderivative_delta = atan(t_hi) - atan(t_lo)
        area = Rational(k, a) * antiderivative_delta
        function_text = f"f(x) = {k}/({denominator})"
        part_a_prompt = f"Write {denominator} in the form (x + p)^2 + q. Find q."
        working_a = [
            f"{denominator} = (x {'+' if h >= 0 else '-'} {abs(h)})^2 + {constant} - {h * h}, so q = {m}.",
        ]
        wrong_m = [
            (constant, "completing-square-constant-unchanged"),
            (constant + h * h, "completing-square-adjusted-wrong-way"),
            (h * h, "completing-square-reports-half-squared"),
            (a, "completing-square-reports-root-not-square"),
        ]
        follow = (
            f"{k}/sqrt(part_a)*(atan(({hi}+{paren_negative(h)})/sqrt(part_a))"
            f" - atan(({lo}+{paren_negative(h)})/sqrt(part_a)))"
        )
        wrong_area = [
            (k * antiderivative_delta, "arctan-missing-one-over-a"),
            (Rational(k, a * a) * antiderivative_delta, "arctan-divided-by-a-squared"),
            (-area, "integral-limit-sign-slip"),
            (Rational(1, a) * antiderivative_delta, "integral-dropped-numerator-constant"),
        ]
        working_b = [
            f"The antiderivative is ({k}/{a}) arctan(({join_terms([(1, 'x'), (h, '')])})/{a}).",
            f"Evaluating between x = {lo} and x = {hi} gives {_typed(area)}.",
        ]
    else:
        constant = a * a - h * h
        denominator = _quadratic(-1, -2 * h, constant)
        integrand_float = lambda v: k / math.sqrt(-v * v - 2 * h * v + constant)  # noqa: E731
        antiderivative_delta = asin(t_hi) - asin(t_lo)
        area = k * antiderivative_delta
        function_text = f"f(x) = {k}/sqrt({denominator})"
        part_a_prompt = f"Write {denominator} in the form q - (x + p)^2. Find q."
        working_a = [
            f"{denominator} = {constant} + {h * h} - (x {'+' if h >= 0 else '-'} {abs(h)})^2, so q = {m}.",
        ]
        wrong_m = [
            (constant, "completing-square-constant-unchanged"),
            (constant - h * h, "completing-square-adjusted-wrong-way"),
            (h * h, "completing-square-reports-half-squared"),
            (a, "completing-square-reports-root-not-square"),
        ]
        follow = f"{k}*(asin(({hi}+{paren_negative(h)})/sqrt(part_a)) - asin(({lo}+{paren_negative(h)})/sqrt(part_a)))"
        wrong_area = [
            (Rational(k, a) * antiderivative_delta, "arcsin-extra-division-by-a"),
            (k * a * antiderivative_delta, "arcsin-extra-multiplication-by-a"),
            (-area, "integral-limit-sign-slip"),
            (antiderivative_delta, "integral-dropped-numerator-constant"),
        ]
        working_b = [
            f"The antiderivative is {k} arcsin(({join_terms([(1, 'x'), (h, '')])})/{a}).",
            f"Evaluating between x = {lo} and x = {hi} gives {_typed(area)}.",
        ]

    check(
        abs(_simpson(integrand_float, lo, hi) - float(area)) < 1e-9,
        "the closed form disagrees with numerical integration",
    )
    mean = area / width
    check(mean != 0, "mean value must not be zero")
    wrong_mean = [
        (area, "mean-value-not-divided-by-interval-length"),
        (area / hi if hi != 0 else None, "mean-value-divided-by-upper-limit"),
        (area * width, "mean-value-multiplied-by-interval-length"),
        (2 * area / width, "mean-value-divided-by-half-interval"),
    ]
    wrong_mean = [(v, tag) for v, tag in wrong_mean if v is not None]

    area_text = _typed(area)
    mean_text = _typed(mean)
    stem = f"The function f is given by {function_text}."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=part_a_prompt,
                answer_kind="number",
                answer=m,
                answer_text=str(m),
                answer_latex=latex_answer(str(m)),
                marks=2,
                sensible_range=(0, None),
                working=working_a,
                distractors=_int_distractors(m, wrong_m),
            ),
            dict(
                prompt=(
                    f"Find the exact area of the region bounded by the curve y = f(x), the x-axis and the "
                    f"lines x = {lo} and x = {hi}. Give your answer as an exact value in terms of pi."
                ),
                answer_kind="expression",
                answer=area_text,
                answer_text=area_text,
                answer_latex=latex_answer(area),
                marks=3,
                follow_through=follow,
                working=working_b,
                distractors=_expr_distractors(area, wrong_area),
            ),
            dict(
                prompt=(
                    f"Hence find the mean value of f(x) over the interval {lo} ≤ x ≤ {hi}. "
                    "Give your answer as an exact value in terms of pi."
                ),
                answer_kind="expression",
                answer=mean_text,
                answer_text=mean_text,
                answer_latex=latex_answer(mean),
                marks=2,
                follow_through=f"part_b/({hi}-({lo}))",
                working=[f"Mean value = (1/{width}) × {area_text} = {mean_text}."],
                distractors=_expr_distractors(mean, wrong_mean),
            ),
        ],
    )
