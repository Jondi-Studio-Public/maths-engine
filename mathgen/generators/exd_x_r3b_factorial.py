"""Generates `exd_x_r3b_factorial`: a multi-part exam question on factorial notation.

The stem defines ``S(n)`` as a sum or difference of two ratios of factorials of consecutive integers, such as
``(n + 3)!/(n + 1)! - (n + 1)!/(n - 1)!``, for integers ``n >= 2``. Each ratio is a product of consecutive factors,
so S(n) is a quadratic with positive leading coefficient once the factorials are cancelled.

(a) S(n) as a polynomial in n, expanded                         expression, 2 marks
(b) hence S(n0) for a stated n0                                 number,     1 mark,  Subs(part_a, (n,), (n0,))
(c) the integer n >= 2 for which S(n) = K                        number,     2 marks, the quadratic formula on part_a

K = S(n1) for a chosen n1, and the other root of S(n) = K is below 2, so n1 is the only solution in the domain. Every
key is checked against `math.factorial` on the stated expression, with no algebra.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from sympy import Poly, Symbol

from ..core import attempts, check, generator, phrase, value_safe_str
from .exd_x_r3b_common import expression_part, number_part
from ._exd_distinct_parts import distinct_part_answers

N = Symbol("n")

_STEMS = (
    "For integers n ≥ 2, let S(n) = {s}.",
    "A sequence has terms S(n) = {s}, for integers n ≥ 2.",
    "Let S(n) = {s}, where n is an integer and n ≥ 2.",
)
_QUADRATIC_PAIRS = ((2, 1), (1, 2), (2, 2))


def _poly(expr) -> Poly:
    return Poly(expr, N, domain="ZZ")


def _shift(k: int) -> str:
    return "n" if k == 0 else f"n + {k}" if k > 0 else f"n - {-k}"


def _fact(k: int) -> str:
    return "n!" if k == 0 else f"({_shift(k)})!"


def _ratio_text(a: int, b: int) -> str:
    return f"{_fact(a)}/{_fact(b)}"


def _factors(a: int, b: int) -> list[Poly]:
    """The consecutive factors (n + b + 1) ... (n + a) of (n + a)!/(n + b)!."""
    return [_poly(N + i) for i in range(b + 1, a + 1)]


def _product(factors) -> Poly:
    out = _poly(1)
    for f in factors:
        out = out * f
    return out


def _ratio_value(n: int, a: int, b: int) -> Fraction:
    return Fraction(math.factorial(n + a), math.factorial(n + b))


def _evaluate(poly: Poly, n: int) -> int:
    return int(poly.eval(n))


def _first_above(poly: Poly, bound: int) -> int | None:
    """The smallest integer n >= 2 with poly(n) > bound, for a polynomial that is eventually increasing."""
    if poly.degree() < 1 or poly.LC() <= 0:
        return None
    for n in range(2, 400):
        if poly.eval(n) > bound:
            return n
    return None


def _build(rng: random.Random) -> dict:
    d1, d2 = rng.choice(_QUADRATIC_PAIRS)
    sign = 1 if (d1, d2) == (2, 2) else rng.choice([1, -1])
    if sign == -1 and d1 == 1:
        sign = 1  # keep the leading coefficient positive: the quadratic term is never the one subtracted
    for _ in attempts(500, "two different ratios"):
        b1, b2 = rng.randint(-2, 1), rng.randint(-2, 1)
        a1, a2 = b1 + d1, b2 + d2
        if (a1, b1) != (a2, b2):
            break
    t1, t2 = _product(_factors(a1, b1)), _product(_factors(a2, b2))
    s_poly = t1 + t2 if sign == 1 else t1 - t2
    check(s_poly.degree() == 2 and s_poly.LC() > 0, "S is not a quadratic with positive leading coefficient")

    # Independent route: evaluate the stated factorial expression with math.factorial at several n.
    for n in range(2, 12):
        direct = _ratio_value(n, a1, b1) + sign * _ratio_value(n, a2, b2)
        check(direct == s_poly.eval(n), "S(n) disagrees with the factorial expression")

    n1 = rng.randint(5, 22)
    bound = _evaluate(s_poly, n1)
    roots = [n for n in range(2, 400) if s_poly.eval(n) == bound]
    check(roots == [n1], "S(n) = K has more than one solution with n >= 2")
    check(
        math.factorial(n1 + a1) // math.factorial(n1 + b1) + sign * (math.factorial(n1 + a2) // math.factorial(n1 + b2))
        == bound,
        "K disagrees with the factorial expression at n1",
    )
    n0 = rng.choice([v for v in range(6, 16) if v != n1])

    s_text = _ratio_text(a1, b1) + (" + " if sign == 1 else " - ") + _ratio_text(a2, b2)
    key_a = value_safe_str(s_poly.as_expr())

    f1, f2 = _factors(a1, b1), _factors(a2, b2)
    drop_first = _product(f1[1:]) if len(f1) > 1 else _poly(1)
    extra_factor = t1 * _poly(N + a1 + 1)
    summed = sum((f for f in f1), _poly(0)) if len(f1) > 1 else None
    wrong_polys = [
        ("forgets-one-factorial", drop_first + sign * t2),
        ("adds-extra-factor", extra_factor + sign * t2),
        ("summed-factors-not-multiplied", None if summed is None else summed + sign * t2),
        ("forgets-one-factorial", t1 + sign * (_product(f2[1:]) if len(f2) > 1 else _poly(1))),
        ("sign-error-combining-terms", t1 - sign * t2),
    ]
    wrong_polys = [(mid, p) for mid, p in wrong_polys if p is not None and p != s_poly and p.degree() >= 1]

    part_a = expression_part(
        "Simplify S(n), giving your answer as a polynomial in n in expanded form.",
        key_a,
        _latex(s_poly),
        2,
        [
            f"{_ratio_text(a1, b1)} = {_product_text(f1)}.",
            f"{_ratio_text(a2, b2)} = {_product_text(f2)}.",
            f"S(n) = {key_a}.",
        ],
        [(mid, value_safe_str(p.as_expr()), True) for mid, p in wrong_polys],
    )
    part_a["answer_form"] = "expanded"
    value_b = _evaluate(s_poly, n0)
    part_b = number_part(
        f"Hence find the value of S({n0}).",
        Fraction(value_b),
        0,
        1,
        [f"S({n0}) = {value_b}."],
        [(mid, Fraction(_evaluate(p, n0))) for mid, p in wrong_polys],
        follow_through=f"Subs(part_a, (n,), ({n0},))",
    )
    lead = int(s_poly.LC())
    u, v = "Subs(part_a, (n,), (0,))", "Subs(part_a, (n,), (1,))"
    slope = f"({v} - {u} - {lead})"  # the coefficient of n, given the known leading coefficient
    root = f"(-{slope} + sqrt({slope}**2 + {4 * lead}*({bound} - {u})))/{2 * lead}"
    coeff_b = int(s_poly.coeffs()[1]) if s_poly.degree() == 2 and len(s_poly.coeffs()) > 1 else 0
    other = Fraction(-coeff_b, lead) - n1
    check(other < 2, "the other root of S(n) = K is in the domain")
    c_cands = [("gave-extraneous-negative-root", other if other.denominator == 1 else None)]
    c_cands.append(("gave-value-not-n", Fraction(bound)))
    c_cands.append(("inequality-linear-term-ignored", Fraction(math.isqrt(bound // lead))))
    for mid, p in wrong_polys:
        nearest = min(range(2, 60), key=lambda m: (abs(_evaluate(p, m) - bound), m))
        c_cands.append((mid, Fraction(nearest)))
    part_c = number_part(
        f"Find the value of n for which S(n) = {bound}.",
        Fraction(n1),
        0,
        2,
        [
            f"S(n) = {key_a}, so {key_a} = {bound}.",
            f"n = {n1} (the other solution, n = {other}, is less than 2).",
        ],
        c_cands,
        follow_through=root,
    )
    return dict(text=phrase(rng, _STEMS, s=s_text), parts=[part_a, part_b, part_c])


def _product_text(factors) -> str:
    return "".join(f"({f.as_expr()})" for f in factors)


def _latex(poly: Poly) -> str:
    from sympy import latex

    return latex(poly.as_expr())


@generator("exd_x_r3b_factorial", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_factorial(rng: random.Random, tier: str = "exam") -> dict:
    """Cancel two ratios of factorials, evaluate the result, then find where it first passes a bound."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
