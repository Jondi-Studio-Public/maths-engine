"""Generates `exd_x_r3f_vector_curve` questions: an exam chain on the curve r(t) = (a t + p, b/t + c).

The curve has one unknown constant b and passes through a point P. The chain:

(a) the parameter t at P                                    number, 1 mark      t = (x0 - p)/a
(b) the constant b                                          number, 2 marks     b = (y0 - c) t
(c) the constant K in (x - p)(y - c) = K                    number, 1 mark      K = a b
(d) y when x takes a second stated value                    number, 2 marks     y = c + K/(x1 - p)

Each part follows through from the one before it. The second value of x comes from a second parameter value
t1 that divides b, so the answer to (d) is a whole number.
"""

from __future__ import annotations

import random

from ..core import attempts, check, generator, join_terms, phrase
from ._exd_r3f_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "A curve has vector equation r(t) = ({a}t{p_term}, b/t{c_term}) for t \u2260 0, where b is a constant. "
    "The curve passes through the point P({x0}, {y0}).",
    "The position of a point on a curve is given by r(t) = ({a}t{p_term}, b/t{c_term}), where t \u2260 0 and b is "
    "a constant. The point P({x0}, {y0}) lies on the curve.",
    "Consider the curve with vector equation r(t) = ({a}t{p_term}, b/t{c_term}), t \u2260 0, for a constant b. "
    "It passes through P({x0}, {y0}).",
)


def _signed(value: int) -> str:
    """` + 3` or ` - 3` for appending to an expression; empty for zero."""
    return "" if value == 0 else f" {'+' if value > 0 else '-'} {abs(value)}"


def _whole(num: int, den: int):
    """num/den when it is a whole number, else None."""
    return num // den if den != 0 and num % den == 0 else None


@generator("exd_x_r3f_vector_curve", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_vector_curve(rng: random.Random, tier: str = "exam") -> dict:
    """Find the parameter, then b, then the Cartesian constant, then a second point on the curve."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a curve with a whole-number second point"):
        a = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
        p = rng.randint(-6, 6)
        c = rng.choice([v for v in range(-8, 9) if v != 0])
        t0 = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
        m = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        b = m * t0
        divisors = [t for t in range(-12, 13) if t not in (0, t0) and b % t == 0]
        if divisors:
            t1 = rng.choice(divisors)
            break
    x0, y0 = a * t0 + p, m + c
    big_k = a * b
    x1 = a * t1 + p
    y1 = c + b // t1

    # Independent check: both points lie on the curve, and the Cartesian relation holds at each.
    check(y0 == b // t0 + c and (x0 - p) * (y0 - c) == big_k, "P is not on the curve")
    check(x1 != p and (x1 - p) * (y1 - c) == big_k, "the second point is not on the curve")

    x_part = join_terms([(1, "x"), (-p, "")])
    y_part = join_terms([(1, "y"), (-c, "")])
    stem = phrase(rng, _STEMS, a=a, p_term=_signed(p), c_term=_signed(c), x0=x0, y0=y0)

    t_candidates = [
        (x0 - p, "forgot-to-divide-by-coefficient"),
        (-t0, "sign-error-moving-term"),
        (_whole(x0, a), "parameter-offset-ignored"),
        (_whole(x0 + p, a), "parameter-offset-sign-error"),
    ]
    b_candidates = [
        (y0 * t0, "parameter-offset-y-ignored"),
        ((y0 + c) * t0, "parameter-offset-sign-error"),
        (_whole(y0 - c, t0), "divided-by-parameter-instead-of-multiplied"),
        (y0 - c, "reports-b-over-t-not-b"),
    ]
    k_candidates = [
        (a + b, "coefficients-added-for-constant-product"),
        (b, "dropped-coefficient-a"),
        (-big_k, "sign-error-eliminating-parameter"),
        (x0 * y0, "relation-from-point-product"),
    ]
    y_candidates = [
        (y1 - c, "cartesian-offset-y-ignored"),
        (c - (y1 - c), "cartesian-sign-error-on-constant"),
        (y0, "second-point-reuses-first-point"),
        (_whole(big_k, x1) + c if _whole(big_k, x1) is not None else None, "cartesian-offset-x-ignored"),
        (b + c, "second-point-parameter-set-to-one"),
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the value of t at the point P.",
                t0,
                t_candidates,
                marks=1,
                working=[f"{a}t{_signed(p)} = {x0}, so t = {t0}."],
            ),
            number_part(
                "Hence find the value of b.",
                b,
                b_candidates,
                marks=2,
                follow_through=f"({y0} - ({c}))*part_a",
                working=[f"b/{t0} + ({c}) = {y0}, so b = {b}."],
            ),
            number_part(
                f"Eliminating t gives the relation ({x_part})({y_part}) = K. Find K.",
                big_k,
                k_candidates,
                marks=1,
                follow_through=f"({a})*part_b",
                working=[f"(x - ({p}))(y - ({c})) = ({a}t)(b/t) = {a}b = {big_k}."],
            ),
            number_part(
                f"Use this relation to find the value of y when x = {x1}.",
                y1,
                y_candidates,
                marks=2,
                follow_through=f"({c}) + part_c/({x1} - ({p}))",
                working=[f"y - ({c}) = {big_k}/({x1} - ({p})) = {y1 - c}, so y = {y1}."],
            ),
        ],
    )
