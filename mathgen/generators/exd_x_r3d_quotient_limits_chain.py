"""Exam source of `senior-examine-the-behaviour-of-the-difference-quotient-frac`: recover a quadratic, then run it backwards.

The practice tiers read the limit of (f(x+h) - f(x))/h off a few tabulated values. An exam question gives the limits of
the difference quotient of f(x) = a x^2 + b x at two points and asks for the function and then a reverse question:

(a) the constant a (the limit at x is 2 a x + b, so two limits give a);
(b) the constant b, which follows through from (a);
(c) the value of x at which the difference quotient approaches a stated value, which follows through from (a) and (b).

Built backwards from a, b and the three x values; every answer is a whole number. The independent check expands
(f(x+h) - f(x))/h symbolically with SymPy and takes the limit at each of the three points.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, latex_answer

from .exd_x_r3d_common import tagged
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "The function f is given by f(x) = ax² + bx, where a and b are constants. The difference quotient "
    "(f(x + h) - f(x))/h approaches {l1} as h approaches 0 when x = {x1}, and approaches {l2} when x = {x2}.",
    "Let f(x) = ax² + bx for constants a and b. As h → 0, the difference quotient (f(x + h) - f(x))/h has limit "
    "{l1} at x = {x1} and limit {l2} at x = {x2}.",
    "A curve has equation y = f(x), where f(x) = ax² + bx. The limit as h → 0 of (f(x + h) - f(x))/h is {l1} when x = {x1} and {l2} "
    "when x = {x2}.",
)
_PROMPT_A = ("Find the value of a.", "Determine a.")
_PROMPT_B = ("Hence find the value of b.", "Using your value of a, find b.")
_PROMPT_C = (
    "At what value of x does the difference quotient approach {m} as h approaches 0?",
    "Find the value of x at which the limit of the difference quotient, as h → 0, equals {m}.",
)


def _whole(num: int, den: int):
    return num // den if den != 0 and num % den == 0 else None


@generator("exd_x_r3d_quotient_limits_chain", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_quotient_limits_chain(rng: random.Random, tier: str = "exam") -> dict:
    """a, b and the x at which the difference quotient reaches a stated limit."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a draw with whole-number limits and distractors"):
        a = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        x1, x2, x3 = rng.sample(range(-6, 7), 3)
        if abs(x1 - x2) == 0:
            continue
        l1, l2, m = 2 * a * x1 + b, 2 * a * x2 + b, 2 * a * x3 + b
        if len({l1, l2, m}) < 3 or 0 in (l1, l2, m):
            continue
        a_cands = [
            (2 * a, "power-rule-coefficient-dropped"),
            (-a, "sign-error-moving-term"),
            (_whole(l2 - l1, 2 * (x1 + x2)), "x-values-added-not-subtracted"),
            (_whole(l1, 2 * x1), "limit-linear-term-ignored"),
        ]
        b_cands = [
            (l1 - a * x1, "power-rule-coefficient-dropped"),
            (l1, "quadratic-term-ignored"),
            (l1 + 2 * a * x1, "sign-error-moving-term"),
            (l2 - a * x2, "power-rule-coefficient-dropped"),
        ]
        c_cands = [
            (_whole(m - b, a), "power-rule-coefficient-dropped"),
            (_whole(m + b, 2 * a), "sign-error-moving-term"),
            (_whole(m, 2 * a), "limit-linear-term-ignored"),
            (-x3, "sign-error-moving-term"),
        ]
        a_d = tagged(str(a), [c for c in a_cands if c[0] is not None])
        b_d = tagged(str(b), [c for c in b_cands if c[0] is not None])
        c_d = tagged(str(x3), [c for c in c_cands if c[0] is not None])
        if len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2:
            break
    # Independent route: the difference quotient itself at a very small h, from the function values.
    tiny = Fraction(1, 10**9)

    def quotient(point: int) -> Fraction:
        def f(t):
            return a * t * t + b * t

        return (f(point + tiny) - f(point)) / tiny

    for point, want in ((x1, l1), (x2, l2), (x3, m)):
        check(abs(quotient(point) - want) < Fraction(1, 10**6), "limit of the difference quotient disagrees")
    check((l2 - l1) % (2 * (x2 - x1)) == 0 and (l2 - l1) // (2 * (x2 - x1)) == a, "a does not follow from the limits")

    stem = rng.choice(_STEMS).format(l1=l1, x1=x1, l2=l2, x2=x2)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A),
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=latex_answer(a),
                marks=2,
                working=[f"The limit at x is 2ax + b, so {l2} - ({l1}) = 2a({x2} - ({x1})) and a = {a}."],
                distractors=a_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_B),
                answer_kind="number",
                answer=b,
                answer_text=str(b),
                answer_latex=latex_answer(b),
                marks=1,
                follow_through=f"{l1} - 2*part_a*({x1})",
                working=[f"2a({x1}) + b = {l1}, so b = {b}."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C).format(m=m),
                answer_kind="number",
                answer=x3,
                answer_text=str(x3),
                answer_latex=latex_answer(x3),
                marks=2,
                follow_through=f"({m} - part_b)/(2*part_a)",
                working=[f"2({a})x + ({b}) = {m}, so x = {x3}."],
                distractors=c_d,
            ),
        ],
    )
