"""Exam-tier source for `classify-constant-coefficient-pde`: a second-order PDE with a parameter in the mixed term.

The equation is a u_xx + (m k) u_xy + c u_yy + (lower-order terms) = 0 with k a positive constant, where a and c
are positive and a c is a perfect square. Classification depends only on the sign of the discriminant
B^2 - 4AC = m^2 k^2 - 4 a c, so the lower-order terms are there to be ignored. Three connected parts:

(a) the value of k at which the equation is parabolic (the discriminant is zero);
(b) the largest whole-number k for which it is elliptic (follow-through from (a));
(c) the value of the discriminant at that k (follow-through from (b)), which comes out negative and confirms (b).

Built backwards from the perfect square: a = g p^2, c = g q^2, so 4 a c = (2 g p q)^2 and the parabolic value is
k = 2 g p q / |m|, a terminating decimal because |m| has only the prime factors 2 and 5.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
)

_M_VALUES = (1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50)


def _term(coef: int, symbol: str, first: bool = False) -> str:
    """A signed term such as ' + 5u_x' or ' - u' (or '5u_x' when first)."""
    body = f"{abs(coef) if abs(coef) != 1 else ''}{symbol}"
    if first:
        return f"-{body}" if coef < 0 else body
    return f" - {body}" if coef < 0 else f" + {body}"


def _dec(value: Fraction) -> str:
    return exact_decimal_string(value, 4)


def _dis(value: Fraction, tag: str | None) -> Distractor | None:
    """A distractor shown as a decimal; None when the value does not terminate within 4 decimal places."""
    try:
        return Distractor(_dec(value), tag, exact_value(value))
    except ValueError:
        return None


@generator("exd_x_sc2_pde_parameter", level=17, presets={"exam": dict(tier="exam")})
def exd_x_sc2_pde_parameter(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: find the parabolic value of k, the largest elliptic integer k, and the discriminant there."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a parameter with a parabolic value above 1.5"):
        g = rng.randint(1, 6)
        p, q = rng.randint(1, 6), rng.randint(1, 6)
        m = rng.choice(_M_VALUES)
        a, c = g * p * p, g * q * q
        if max(a, c) > 72:
            continue
        root = g * p * q
        k0 = Fraction(2 * root, m)
        if not (Fraction(3, 2) < k0 <= 20):
            continue
        if a == c and rng.random() < 0.5:
            continue  # keep a = c (a symmetric equation) rarer
        break

    sign = rng.choice((1, -1))
    b_coef = sign * m
    check(a * c == root * root and math.isqrt(a * c) == root, "a c is not the perfect square")

    # Independent route: the discriminant as a function of k, solved exactly.
    def disc(k: Fraction) -> Fraction:
        return Fraction(b_coef) ** 2 * k * k - 4 * a * c

    check(disc(k0) == 0, "discriminant is not zero at the parabolic value")
    biggest = math.ceil(k0) - 1  # the largest integer below k0
    check(disc(Fraction(biggest)) < 0 <= disc(Fraction(biggest + 1)), "largest elliptic integer disagrees")
    d_at = int(disc(Fraction(biggest)))

    first_order = rng.choice(("u_x", "u_y"))
    coefs = [rng.choice((-3, -2, 2, 3, 4, 5)), rng.choice((-4, -2, -1, 1, 2, 5))]
    terms = _term(coefs[0], first_order) + _term(coefs[1], "u")
    lead = f"{a if a != 1 else ''}u_xx"
    mixed = f"{'-' if b_coef < 0 else '+'} {m if m != 1 else ''}k u_xy"
    stem = (
        f"Consider the partial differential equation {lead} {mixed} + {c if c != 1 else ''}u_yy{terms} = 0, "
        "where k is a positive constant. The equation is classified as elliptic when B² − 4AC < 0, parabolic when "
        "B² − 4AC = 0 and hyperbolic when B² − 4AC > 0, where A, B and C are the coefficients of u_xx, u_xy and u_yy."
    )

    k_text = _dec(k0)
    big_text, disc_text = str(biggest), str(d_at)
    a_cands = [
        _dis(Fraction(4 * a * c, m * m), "forgot-square-root"),
        _dis(Fraction(root, m), "dropped-factor-4-in-discriminant"),
        _dis(Fraction(4 * root, m), None),
        _dis(Fraction(2 * a * c, m), None),
        _dis(k0 + 1, None),
        _dis(k0 - 1, None),
    ]
    b_cands = [
        Distractor(str(math.ceil(k0)), "included-the-parabolic-value", Fraction(math.ceil(k0))),
        _dis(k0, "gave-boundary-value-not-integer") if k0.denominator != 1 else None,
        Distractor(str(biggest - 1), None, Fraction(biggest - 1)),
        Distractor(str(round(k0)), None, Fraction(round(k0))),
        Distractor(str(biggest + 2), None, Fraction(biggest + 2)),
    ]
    c_cands = [
        Distractor(str(-d_at), "reversed-discriminant-sign", Fraction(-d_at)),
        Distractor(str(m * m - 4 * a * c), "forgot-parameter-in-discriminant", Fraction(m * m - 4 * a * c)),
        Distractor(
            str(biggest * biggest - 4 * a * c), "forgot-to-square-coefficient", Fraction(biggest * biggest - 4 * a * c)
        ),
        Distractor(str(m * biggest - 4 * a * c), None, Fraction(m * biggest - 4 * a * c)),
        Distractor(str(d_at - m * m), None, Fraction(d_at - m * m)),
    ]
    a_dis = pick_distractors(k_text, [x for x in a_cands if x], 3, tolerance=GRADER_TOLERANCE)
    b_dis = pick_distractors(big_text, [x for x in b_cands if x], 3, tolerance=GRADER_TOLERANCE)
    c_dis = pick_distractors(disc_text, [x for x in c_cands if x], 3, tolerance=GRADER_TOLERANCE)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of k for which the equation is parabolic. Give k as a decimal.",
                answer_kind="number",
                answer=numeric_answer(k0),
                answer_text=k_text,
                answer_latex=latex_answer(k_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"A = {a}, B = {b_coef}k, C = {c}, so B² − 4AC = {m * m}k² − {4 * a * c}.",
                    f"Parabolic: {m * m}k² = {4 * a * c}, so k² = {Fraction(4 * a * c, m * m)} and k = {k_text} (k > 0).",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt="Hence find the largest whole-number value of k for which the equation is elliptic.",
                answer_kind="number",
                answer=biggest,
                answer_text=big_text,
                answer_latex=latex_answer(big_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="ceiling(part_a) - 1",
                working=[
                    f"B² − 4AC < 0 when {m * m}k² < {4 * a * c}, that is for k < {k_text}.",
                    f"The largest whole number below {k_text} is {biggest}.",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt="Find the value of B² − 4AC when k is equal to your answer to (b).",
                answer_kind="number",
                answer=d_at,
                answer_text=disc_text,
                answer_latex=latex_answer(disc_text),
                marks=2,
                follow_through=f"{m * m}*part_b**2 - {4 * a * c}",
                working=[f"{m * m}({biggest})² − {4 * a * c} = {m * m * biggest * biggest} − {4 * a * c} = {d_at}."],
                distractors=c_dis,
            ),
        ],
    )
