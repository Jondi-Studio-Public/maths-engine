"""Exam-tier source for `simplify-surds`: a rectangle whose sides are surds with different square-free parts.

The easy, medium and hard tiers simplify surds that are handed over as an expression. Here the learner builds the
expression from a setting and has to combine surds that cannot be added:

    (a) the area, a product of two different surds, left as a single surd in simplest form,
    (b) the square of the diagonal, from Pythagoras' theorem (an integer, because each side is squared),
    (c) the diagonal itself, the square root of that integer, in simplest surd form.

Built backwards from the two sides, a root `k` and a square-free `s` so that the squared diagonal is exactly
`k*k*s` with `k >= 2`: the diagonal always needs simplifying. Part (c) follows through from the learner's (b).
"""

from __future__ import annotations

import math
import re

import sympy
from sympy import Integer, sqrt

from mathgen.answers import Distractor
from mathgen.core import attempts, check, check_equivalent, generator, latex_answer, pick_distractors

_RADICANDS = (2, 3, 5, 6, 7, 10, 11, 13)
_SETTINGS = (
    "A rectangular courtyard in Canberra",
    "A rectangular sail in Fremantle",
    "A rectangular vegetable patch in Launceston",
    "A rectangular solar panel on a roof in Alice Springs",
    "A rectangular sheet of marine ply in Hobart",
)
_SIDE_PHRASES = (
    "{setting} has side lengths {side1} m and {side2} m.",
    "The sides of {setting_lower} measure {side1} m and {side2} m.",
)


def _side(coefficient: int, radicand: int) -> str:
    return f"√{radicand}" if coefficient == 1 else f"{coefficient}√{radicand}"


def _largest_square_root_factor(n: int) -> int:
    """The largest k with k*k dividing n, by trial division."""
    return max(k for k in range(1, math.isqrt(n) + 1) if n % (k * k) == 0)


def _surd_text(value) -> str:
    return str(value)


def _expression_distractor(value, tag: str) -> Distractor:
    return Distractor(_surd_text(value), tag, value)


def _is_irrational_surd(value) -> bool:
    return not value.is_Rational


@generator("exd_x_n4_surd_rectangle", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_surd_rectangle(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two surd sides whose diagonal needs simplifying"):
        m, n = rng.sample(_RADICANDS, 2)
        a, c = rng.randint(1, 5), rng.randint(1, 5)
        diagonal_squared = a * a * m + c * c * n
        k = _largest_square_root_factor(diagonal_squared)
        if k < 2 or diagonal_squared == k * k:
            continue
        side1, side2 = _side(a, m), _side(c, n)
        setting = rng.choice(_SETTINGS)
        stem = rng.choice(_SIDE_PHRASES).format(
            setting=setting, setting_lower=setting[0].lower() + setting[1:], side1=side1, side2=side2
        )
        prompts = (
            "Find the area, in m², as a single surd in simplest form.",
            "Find the square of the length of the diagonal, in m².",
            "Hence find the length of the diagonal in simplest surd form, in the form a√b.",
        )
        tokens = {int(t) for text in (stem, *prompts) for t in re.findall(r"\d+", text)}
        if diagonal_squared in tokens:
            continue

        area = Integer(a * c) * sqrt(m * n)
        diagonal = sqrt(diagonal_squared)
        coefficient_unsquared = a * m + c * n
        cands_a = []
        added = Integer(a * c) * sqrt(m + n)
        if _is_irrational_surd(added):
            cands_a.append(_expression_distractor(added, "adds-radicands-when-multiplying"))
        cands_a.append(_expression_distractor(Integer(a + c) * sqrt(m * n), "adds-coefficients-when-multiplying"))
        cands_b = [
            Distractor(str(coefficient_unsquared), "coefficient-not-squared", coefficient_unsquared),
            Distractor(str(a * a * m * c * c * n), "squares-multiplied-not-added", a * a * m * c * c * n),
        ]
        k_unsquared = sqrt(coefficient_unsquared)
        cands_c = [
            _expression_distractor(
                Integer(k * k) * sqrt(diagonal_squared // (k * k)), "surd-extracted-without-rooting"
            ),
            _expression_distractor(Integer(k), "surd-dropped-radical"),
        ]
        if _is_irrational_surd(k_unsquared):
            cands_c.append(_expression_distractor(k_unsquared, "coefficient-not-squared"))
        a_text, b_text, c_text = _surd_text(area), str(diagonal_squared), _surd_text(diagonal)
        d_a = pick_distractors(a_text, cands_a, 3)
        d_b = pick_distractors(b_text, cands_b, 3)
        d_c = pick_distractors(c_text, cands_c, 3)
        if min(len(d_a), len(d_b), len(d_c)) >= 2:
            break

    # Independent routes: build the sides as sympy surds and multiply, then apply Pythagoras to them directly.
    side_x, side_y = a * sqrt(m), c * sqrt(n)
    check_equivalent(area, side_x * side_y, "area disagrees with the product of the sides")
    check_equivalent(diagonal, sqrt(side_x**2 + side_y**2), "diagonal disagrees with Pythagoras on the sides")
    check_equivalent(diagonal**2, diagonal_squared, "diagonal squared is off")
    for answer, options in ((area, d_a), (diagonal, d_c)):
        for d in options:
            check(abs(sympy.N(answer - d.value, 30)) > 1e-9, "a distractor equals the answer")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="expression",
                answer=area,
                answer_text=a_text,
                answer_latex=latex_answer(a_text, evaluate=False),
                answer_form="simplified_surd",
                marks=2,
                working=[
                    f"Area = {side1} × {side2} = {a * c}√({m} × {n}) = {a * c}√{m * n}, simplified to {a_text}.",
                ],
                distractors=d_a,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=diagonal_squared,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"d² = ({side1})² + ({side2})² = {a * a * m} + {c * c * n} = {b_text}.",
                ],
                distractors=d_b,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="expression",
                answer=diagonal,
                answer_text=c_text,
                answer_latex=latex_answer(c_text, evaluate=False),
                answer_form="simplified_surd",
                marks=2,
                follow_through="sqrt(part_b)",
                working=[f"d = √{diagonal_squared} = √({k * k} × {diagonal_squared // (k * k)}) = {c_text}."],
                distractors=d_c,
            ),
        ],
    )
