"""Exam chain on a quotient of radicals of monomials: simplify to k a^p b^q, then use the result.

The stem gives cbrt(s m1^3 a^(3 x1) b^(3 x2)) / sqrt(m2^2 a^(2 y1) b^(2 y2)) for positive a and b, where s is +1 or
-1. The simplified form is k a^p b^q with k = s m1 / m2, p = x1 - y1 and q = x2 - y2 (either exponent may be
negative, so a variable can end up under the fraction bar). Four parts:

(a) the coefficient k, as a fraction;
(b) the exponent p;
(c) the exponent q;
(d) the value of the expression at stated whole values of a and b, as a fraction.

Built backwards from k, p and q; the radicands are raised from them, and the answer is re-derived from the radicands
with SymPy using positive symbols and a real cube root.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd

import sympy

from mathgen.core import check, draw_until, generator, phrase

from .exd_x_r2j_support import fraction_part, number_part

_A = sympy.Symbol("a", positive=True)
_B = sympy.Symbol("b", positive=True)

_STEM = [
    "Simplify {expr} to the form k a^p b^q, where a and b are positive.",
    "For positive a and b, the expression {expr} simplifies to the form k a^p b^q.",
    "The expression {expr}, in which a and b are positive, can be written as k a^p b^q.",
]


def _monomial(coef: int, ea: int, eb: int) -> str:
    def var(name: str, e: int) -> str:
        return "" if e == 0 else (name if e == 1 else f"{name}^{e}")

    body = f"{var('a', ea)}{var('b', eb)}"
    return body if coef == 1 and body else f"{coef}{body}"


def _draw(rng: random.Random):
    sign = rng.choice([1, 1, -1])
    m1, m2 = rng.randint(1, 6), rng.randint(2, 7)
    p = rng.choice([-3, -2, -1, 1, 2, 3])
    q = rng.choice([-3, -2, -1, 1, 2, 3])
    ua, ub = rng.randint(0, 2), rng.randint(0, 2)
    a0, b0 = rng.randint(2, 4), rng.randint(2, 5)
    return sign, m1, m2, p, q, ua, ub, a0, b0


def _accept(draw) -> bool:
    sign, m1, m2, p, q, ua, ub, a0, b0 = draw
    return gcd(m1, m2) == 1 and not (m1 == 1 and m2 == 1) and a0 != b0 and (p > 0 or q > 0 or ua + ub > 0)


@generator("exd_x_r2j_radical_monomial_chain", level=10, presets={"exam": dict(tier="exam")})
def exd_x_r2j_radical_monomial_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Coefficient and exponents of a simplified quotient of radicals, then its value at a point."""
    check(tier == "exam", "exd_x_r2j_radical_monomial_chain: only the exam tier exists")
    sign, m1, m2, p, q, ua, ub, a0, b0 = draw_until(rng, _draw, _accept, what="a radical-monomial chain")
    x1, y1 = max(p, 0) + ua, max(-p, 0) + ua
    x2, y2 = max(q, 0) + ub, max(-q, 0) + ub
    coef = Fraction(sign * m1, m2)
    cube_text = _monomial(sign * m1**3, 3 * x1, 3 * x2)
    square_text = _monomial(m2**2, 2 * y1, 2 * y2)
    expr = f"∛({cube_text}) ÷ √({square_text})"

    # independent route: positive symbols, a real cube root of the signed radicand
    radical = sympy.real_root(sign * m1**3 * _A ** (3 * x1) * _B ** (3 * x2), 3) / sympy.sqrt(
        m2**2 * _A ** (2 * y1) * _B ** (2 * y2)
    )
    target = sympy.Rational(coef.numerator, coef.denominator) * _A**p * _B**q
    check(sympy.simplify(radical - target) == 0, "the radicals do not simplify to k a^p b^q")
    value = coef * Fraction(a0) ** p * Fraction(b0) ** q
    check(sympy.Rational(value.numerator, value.denominator) == target.subs({_A: a0, _B: b0}), "value disagrees")

    stem = phrase(rng, _STEM, expr=expr)
    k_wrong = [
        (Fraction(m2, m1) * sign, "inverted-ratio"),
        (Fraction(sign * m1 * m2), "multiplies-instead-of-dividing"),
        (Fraction(sign * m1, m2 * m2), "radical-coefficient-not-rooted"),
        (Fraction(sign * m1), "ignored-the-square-root"),
        (-coef, "dropped-sign-of-cube-root"),
    ]
    p_wrong = [
        (x1 + y1, "added-instead-of-subtracted"),
        (x1, "ignored-the-square-root"),
        (-p, "variable-on-wrong-side-of-bar"),
        (3 * x1 - 2 * y1, "forgot-root-index"),
        *([(3 * x1 // 2 - 2 * y1 // 3, "wrong-root-index")] if (3 * x1) % 2 == 0 and (2 * y1) % 3 == 0 else []),
    ]
    q_wrong = [
        (x2 + y2, "added-instead-of-subtracted"),
        (x2, "ignored-the-square-root"),
        (-q, "variable-on-wrong-side-of-bar"),
        (3 * x2 - 2 * y2, "forgot-root-index"),
        *([(3 * x2 // 2 - 2 * y2 // 3, "wrong-root-index")] if (3 * x2) % 2 == 0 and (2 * y2) % 3 == 0 else []),
    ]
    value_wrong = [
        (coef * Fraction(a0) ** abs(p) * Fraction(b0) ** abs(q), "variable-on-wrong-side-of-bar"),
        (coef * Fraction(b0) ** p * Fraction(a0) ** q, "swapped-values"),
        (Fraction(a0) ** p * Fraction(b0) ** q, "dropped-coefficient"),
        (coef * (Fraction(a0) * Fraction(b0)) ** (p + q), "same-power-on-both"),
    ]
    ft_value = f"part_a*{a0}**part_b*{b0}**part_c"
    parts = [
        fraction_part(
            prompt="Find the coefficient k, as a fraction in simplest form.",
            exact=coef,
            marks=2,
            working=[f"∛({sign * m1**3}) = {sign * m1} and √({m2**2}) = {m2}, so k = {sign * m1}/{m2}."],
            wrong=k_wrong,
        ),
        number_part(
            prompt="Find the exponent p.",
            exact=p,
            places=0,
            marks=1,
            working=[f"The exponent of a is {3 * x1}/3 - {2 * y1}/2 = {x1} - {y1} = {p}."],
            wrong=[(int(v), t) for v, t in p_wrong],
        ),
        number_part(
            prompt="Find the exponent q.",
            exact=q,
            places=0,
            marks=1,
            working=[f"The exponent of b is {3 * x2}/3 - {2 * y2}/2 = {x2} - {y2} = {q}."],
            wrong=[(int(v), t) for v, t in q_wrong],
        ),
        fraction_part(
            prompt=f"Using your values of k, p and q, find the value of the expression when a = {a0} and b = {b0}, "
            "as a fraction in simplest form.",
            exact=value,
            marks=2,
            working=[f"k a^p b^q = ({coef}) × {a0}^({p}) × {b0}^({q}) = {value}."],
            wrong=value_wrong,
            follow_through=ft_value,
        ),
    ]
    return dict(text=stem, parts=parts)
