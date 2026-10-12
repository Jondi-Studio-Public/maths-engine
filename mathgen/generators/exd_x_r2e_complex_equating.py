"""Exam source for `senior-review-cartesian-form`: a four-part chain that solves ``z^2 = w`` by equating parts.

The stem gives ``w = a + bi`` and the side condition ``Re(z) > 0`` (or ``< 0``) that picks one of the two square
roots of ``w``. Writing ``z = x + yi`` and equating real and imaginary parts gives ``x^2 - y^2 = a`` and
``2xy = b``.

- (a) ``x^2``, from the quadratic in ``x^2`` that those two equations produce (3 marks);
- (b) ``x``, using the side condition to choose the sign (1 mark, ``follow_through`` from (a));
- (c) ``y`` (2 marks, ``follow_through`` from (b));
- (d) the modulus of ``z`` or its principal argument, to 2 decimal places, varied by seed (2 marks,
  ``follow_through`` from (b) and (c)).

## Construction

``z`` is drawn first as integers ``x`` and ``y`` with ``|x| != |y|`` (so neither part of ``w`` is zero) and ``w``
is built from it, so every answer is exact before any rounding. The decimal answers of (d) are kept clear of a
rounding boundary. The argument follow-through uses ``arg(part_b + part_c*I)``, which puts the angle in the right
quadrant whatever signs a learner's own ``x`` and ``y`` have.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_MARGIN = 2e-4
_ABS = Fraction(5, 1000)


def _cx(re: int, im: int) -> str:
    mag = "" if abs(im) == 1 else str(abs(im))
    return f"{re} {'+' if im > 0 else '-'} {mag}i"


def _two_dp(x: float) -> Fraction:
    return round_half_up(Fraction(x), 2)


def _clear(x: float) -> bool:
    scaled = x * 100
    return abs((scaled - math.floor(scaled)) - 0.5) > _MARGIN * 100


def _int(value: int, tag: str) -> Distractor:
    return Distractor(str(value), tag, value)


def _dec(x: float, tag: str) -> Distractor:
    d = _two_dp(x)
    return Distractor(stated_precision_string(d, 2), tag, d)


@generator("exd_x_r2e_complex_equating", level=12, presets={"exam": dict(tier="exam")})
def complex_equating(rng: random.Random, tier: str = "exam") -> dict:
    """Solve z^2 = w for z by equating real and imaginary parts, then read the modulus or argument of z."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    want_modulus = rng.random() < 0.5
    for _ in attempts(4000, "a root clear of rounding boundaries"):
        x_abs = rng.randint(1, 10)
        y = rng.choice([v for v in range(-10, 11) if v and abs(v) != x_abs])
        positive = rng.random() < 0.7
        x = x_abs if positive else -x_abs
        wr, wi = x * x - y * y, 2 * x * y
        modulus, arg = math.hypot(x, y), math.atan2(y, x)
        if not _clear(modulus if want_modulus else arg):
            continue
        break

    z, w = x + y * sympy.I, wr + wi * sympy.I
    check(sympy.expand(z * z) == w, "z must square to w")
    # Independent route: the quadratic in x^2 from x^2 - y^2 = a and 2xy = b.
    disc = sympy.sqrt(wr * wr + wi * wi)
    check(sympy.simplify((wr + disc) / 2 - x * x) == 0, "the positive root of the quadratic in x^2 is x^2")

    side = "Re(z) > 0" if positive else "Re(z) < 0"
    stem = (
        f"Let z = x + yi, where x and y are real numbers, satisfy z^2 = {_cx(wr, wi)} and {side}. "
        f"Equating the real and imaginary parts of z^2 gives two equations in x and y."
    )
    x2 = x * x
    a_cands = [
        _int(-y * y, "takes-the-wrong-root-of-the-quadratic"),
        _int(2 * x2, "forgets-to-halve"),
        _int(wr, "takes-a-given-part-as-x-squared"),
        _int(abs(wi) // 2, "takes-a-given-part-as-x-squared"),
    ]
    b_cands = [
        _int(-x, "ignores-the-sign-condition"),
        _int(x2, "forgets-square-root"),
        _int(wr, "takes-a-given-part-as-x-squared"),
    ]
    c_cands = [
        _int(-y, "sign-error-in-second-equation"),
        _int(2 * y, "forgets-the-factor-2"),
        _int(x * y, "forgets-to-divide-by-x"),
        _int(x, "swaps-x-and-y"),
    ]
    if want_modulus:
        d_cands = [
            _dec(math.sqrt(wr * wr + wi * wi), "gives-the-modulus-of-w"),
            _dec(abs(x) + abs(y), "adds-components-for-modulus"),
            _dec(float(x * x + y * y), "forgets-square-root"),
            _dec(abs(x * x - y * y) ** 0.5, "subtracts-squares-in-modulus"),
        ]
        d_prompt = "Find the modulus of z, correct to 2 decimal places."
        d_value, d_ft = modulus, "sqrt(part_b**2 + part_c**2)"
    else:
        d_cands = [
            _dec(math.atan2(wi, wr), "gives-the-argument-of-w"),
            _dec(math.atan(y / x), "ignores-the-quadrant"),
            _dec(math.atan(x / y), "swaps-real-and-imaginary-parts"),
            _dec(math.degrees(arg), "gives-degrees-not-radians"),
        ]
        d_prompt = "Find the principal argument of z, in radians correct to 2 decimal places."
        d_value, d_ft = arg, "arg(part_b + part_c*I)"
    d_text = stated_precision_string(_two_dp(d_value), 2)
    picks = [
        pick_distractors(str(x2), a_cands, 3),
        pick_distractors(str(x), b_cands, 3),
        pick_distractors(str(y), c_cands, 3),
        pick_distractors(d_text, d_cands, 3, tolerance=_ABS),
    ]
    check(all(len(p) >= 2 for p in picks), f"a part has too few distractors: {[len(p) for p in picks]}")
    ft_b = "sqrt(part_a)" if positive else "-sqrt(part_a)"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of x^2.",
                answer_kind="number",
                answer=x2,
                answer_text=str(x2),
                answer_latex=latex_answer(str(x2)),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"x^2 - y^2 = {wr} and 2xy = {wi}, so y = {wi}/(2x) and x^4 - {wr}x^2 - {wi * wi // 4} = 0.",
                    f"x^2 = ({wr} + sqrt({wr * wr + wi * wi}))/2 = {x2}, taking the positive root.",
                ],
                distractors=picks[0],
            ),
            dict(
                prompt="Hence find x.",
                answer_kind="number",
                answer=x,
                answer_text=str(x),
                answer_latex=latex_answer(str(x)),
                marks=1,
                follow_through=ft_b,
                working=[f"x = {'+' if positive else '-'}sqrt({x2}) = {x}, since {side}."],
                distractors=picks[1],
            ),
            dict(
                prompt="Find y.",
                answer_kind="number",
                answer=y,
                answer_text=str(y),
                answer_latex=latex_answer(str(y)),
                marks=2,
                follow_through=f"{wi}/(2*part_b)",
                working=[f"y = {wi}/(2 x {x}) = {y}."],
                distractors=picks[2],
            ),
            dict(
                prompt=d_prompt,
                answer_kind="number",
                answer=numeric_answer(_two_dp(d_value)),
                answer_text=d_text,
                answer_latex=latex_answer(d_text),
                marks=2,
                sensible_range=(0, None) if want_modulus else None,
                follow_through=d_ft,
                working=[f"z = {_cx(x, y)}, so the {'modulus' if want_modulus else 'argument'} is {d_text}."],
                distractors=picks[3],
            ),
        ],
    )
