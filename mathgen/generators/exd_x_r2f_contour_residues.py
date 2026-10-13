"""Exam source for "evaluate a residue-based contour integral": a four-part chain with two real poles.

f(z) = (a z + b) / (z^2 + B z + C) has two real simple poles p and q. A circle C = {|z - c0| = r}, traversed once
anticlockwise, encloses p but not q. The learner works through:

(a) which pole lies inside C                                   p                       number,   2 marks
(b) the residue of f at that pole                              (a p + b) / (p - q)      fraction, 2 marks, follows from (a)
(c) the integral of f around C                                 2 pi i * residue         expression, 1 mark, follows from (b)
(d) the integral around a larger circle enclosing both poles   (c) + 2 pi i * Res(f, q)  expression, 2 marks, follows from (a), (c)

(d) is the sum of both residues, which for a numerator of degree one is 2 pi i a: the residue at q is worked out from
the learner's own pole in (a), so a wrong (a) carries through. Poles are integers and the circle's radius is a half
integer, so which pole is inside is clear by inspection of distances, never borderline.
"""

from __future__ import annotations

import random

import sympy
from sympy import I, Rational, pi

from ..core import attempts, check, generator, join_terms, paren_negative, rational_text
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r2f_common import expr_part, frac_part


def _circle(c0: int, r: Rational) -> str:
    centre = "z" if c0 == 0 else (f"z - {c0}" if c0 > 0 else f"z + {-c0}")
    return f"|{centre}| = {rational_text(r)}"


@generator(
    "exd_x_r2f_contour_residues",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@distinct_part_answers
def exd_x_r2f_contour_residues(rng: random.Random, tier: str = "exam") -> dict:
    """Which pole is inside the circle, its residue, the integral around the circle, and around a larger circle."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a rational function with one pole inside a circle"):
        p, q = rng.randint(-6, 6), rng.randint(-6, 6)
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        b = rng.choice((-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
        c0 = p + rng.choice((-1, 0, 0, 1))
        r = Rational(2 * abs(p - c0) + 1, 2)
        if 0 in (p, q) or p == -q or abs(p - q) < 3 or abs(q - c0) <= r + 1:
            continue
        res_p, res_q = Rational(a * p + b, p - q), Rational(a * q + b, q - p)
        if res_p != 0 and res_q != 0 and res_p != res_q and abs(res_p) != abs(res_q):
            break
    big = abs(q - c0) + abs(p - c0) + 1  # a circle that encloses both poles
    big_text = _circle(c0, Rational(big))

    B, C = -(p + q), p * q
    num_text = join_terms([(a, "z"), (b, "")])
    den_text = join_terms([(1, "z²"), (B, "z"), (C, "")])
    circle = _circle(c0, r)
    stem = (
        f"Let f(z) = ({num_text}) / ({den_text}). The contour C is the circle {circle}, traversed once anticlockwise."
    )

    # Independent check: the poles are roots of the displayed quadratic, and each residue is N(pole) / Q'(pole).
    check(p * p + B * p + C == 0 and q * q + B * q + C == 0, "poles disagree")
    check(res_p == Rational(a * p + b, 2 * p + B) and res_q == Rational(a * q + b, 2 * q + B), "residues disagree")
    check(res_p + res_q == a, "sum of residues is not the leading coefficient")
    integral_c = sympy.expand(2 * pi * I * res_p)
    integral_big = sympy.expand(2 * pi * I * a)

    a_part = frac_part(
        "Find the poles of f. State the pole that lies inside C.",
        p,
        2,
        [
            f"{den_text} = (z - {p})(z - {q}), so the poles are z = {p} and z = {q}.".replace("- -", "+ "),
            f"|{p} - {paren_negative(c0)}| = {abs(p - c0)} < {rational_text(r)} but |{q} - {paren_negative(c0)}| = {abs(q - c0)} > {rational_text(r)}, "
            f"so z = {p} is inside C.",
        ],
        [
            ("pole-outside-the-contour", q),
            ("sign-flip-both-roots", -p),
            ("sign-flip-both-roots", -q),
        ],
    )
    b_part = frac_part(
        "Find the residue of f at this pole.",
        res_p,
        2,
        [
            f"Res = ({a}·{paren_negative(p)} {'+' if b >= 0 else '-'} {abs(b)}) / ({p} - {paren_negative(q)}) = {rational_text(res_p)}."
        ],
        [
            ("residue-at-wrong-pole", res_q),
            ("forgot-to-divide-by-other-factor", Rational(a * p + b)),
            ("residue-sign-error", -res_p),
            ("origin-substitution", Rational(b, p - q)),
        ],
        follow_through=f"({a}*part_a + {b})/(part_a - ({q}))",
    )
    c_part = expr_part(
        "Hence evaluate the contour integral of f around C.",
        integral_c,
        1,
        [f"Integral = 2πi × Res = 2πi × {rational_text(res_p)} = {sympy.sstr(integral_c)}."],
        [
            ("forgets-2-pi-i", res_p),
            ("forgets-i-in-residue-theorem", sympy.expand(2 * pi * res_p)),
            ("integral-sign-error", -integral_c),
            ("uses-pi-i-not-2-pi-i", sympy.expand(pi * I * res_p)),
        ],
        follow_through="2*pi*I*part_b",
    )
    d_part = expr_part(
        f"Hence evaluate the contour integral of f around the larger circle {big_text}, traversed once anticlockwise.",
        integral_big,
        2,
        [
            f"This circle encloses both poles, so the integral is 2πi × (Res at z = {p} + Res at z = {q}).",
            f"Res at z = {q} is {rational_text(res_q)}, so the integral is 2πi × {a} = {sympy.sstr(integral_big)}.",
        ],
        [
            ("ignores-pole-enclosed-by-larger-contour", integral_c),
            ("residues-subtracted", sympy.expand(2 * pi * I * (res_p - res_q))),
            ("assumes-zero-outside", sympy.Integer(0)),
            ("residue-at-wrong-pole", sympy.expand(2 * pi * I * res_q)),
        ],
        follow_through=f"part_c + 2*pi*I*({a}*({q}) + {b})/(({q}) - part_a)",
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
