"""Generates `exd_x_r3b_second_derivative`: a multi-part exam question on the second derivative of a product.

The stem gives ``f(x) = c x^m (ax + b)^n`` in one of three forms whose second derivative factorises over the
rationals, so the point of inflection can be found by hand:

* ``c x (ax + b)^3``: ``f'' = 6ac (ax + b)(2ax + b)``, two points of inflection;
* ``c x (ax + b)^4``: ``f'' = 4ac (ax + b)^2 (5ax + 2b)``, one point of inflection (the repeated root ``-b/a`` makes
  ``f''`` zero without changing its sign, which is the trap);
* ``c x^3 (ax + b)``: ``f'' = 6c x (2ax + b)``, two points of inflection.

(a) f''(x), expanded                                   expression, 2 marks
(b) hence f''(x0)                                      number,     1 mark,  Subs(part_a, (x,), (x0,))
(c) the x-coordinate of a point of inflection          fraction,   2 marks

The key is differentiated twice as a polynomial and compared with the closed-form factorisation above; every
inflection point is checked to be a root of f'' at which the sign changes.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd

from sympy import Poly, Rational

from ..core import attempts, check, generator, phrase, value_safe_str
from .exd_x_c1_common import X, lin_text, powered, tex, x_pow
from .exd_x_r3b_common import expression_part, fraction_part, number_part
from ._exd_distinct_parts import distinct_part_answers

# (m, n): f = c x^m (ax + b)^n
_FAMILIES = ((1, 3), (1, 4), (3, 1))

_STEMS = (
    "The function f is defined by f(x) = {f}.",
    "Let f(x) = {f}.",
    "A curve has equation y = f(x), where f(x) = {f}.",
)


def _poly(expr) -> Poly:
    return Poly(expr, X, domain="ZZ")


def _frac(value) -> Fraction:
    value = Rational(value)
    return Fraction(int(value.p), int(value.q))


def _closed_second_derivative(m, n, a, b, c) -> Poly:
    """The factorised f'' for the three families, built without differentiating anything."""
    lin = _poly(a * X + b)
    x = _poly(X)
    if (m, n) == (1, 3):
        return _poly(6 * a * c) * lin * _poly(2 * a * X + b)
    if (m, n) == (1, 4):
        return _poly(4 * a * c) * lin * lin * _poly(5 * a * X + 2 * b)
    return _poly(6 * c) * x * _poly(2 * a * X + b)


def _roots(m, n, a, b):
    """Rational points of inflection, sorted: the odd-multiplicity roots of the closed form."""
    if (m, n) == (1, 3):
        return sorted([Rational(-b, a), Rational(-b, 2 * a)])
    if (m, n) == (1, 4):
        return [Rational(-2 * b, 5 * a)]
    return sorted([Rational(0), Rational(-b, 2 * a)])


def _text(poly: Poly) -> str:
    return value_safe_str(poly.as_expr())


def _build(rng: random.Random) -> dict:
    m, n = rng.choice(_FAMILIES)
    for _ in attempts(500, "a function and evaluation point"):
        a = rng.choice([1, 2, 3])
        b = rng.choice([s for s in range(-8, 9) if s != 0 and gcd(a, abs(s)) == 1])
        c = rng.choice([1, 1, 2, -1])
        x0 = rng.choice([-2, -1, 1, 2, 3])
        u, v = _poly(c * X**m), _poly((a * X + b) ** n)
        f2 = u * v
        f2 = f2.diff(X).diff(X)
        if f2.eval(x0) != 0:
            break
    check(f2 == _closed_second_derivative(m, n, a, b, c), "f'' disagrees with its factorised form")
    roots = _roots(m, n, a, b)
    for r in roots:
        check(f2.eval(r) == 0, "an inflection point is not a root of f''")
    greater = roots[-1]
    two_points = len(roots) == 2

    u1, u2, v1, v2 = u.diff(X), u.diff(X).diff(X), v.diff(X), v.diff(X).diff(X)
    f1 = (u * v).diff(X)
    inner = _poly((a * X + b) ** (n - 1)) * n
    a_cands = [
        ("differentiated-once", f1),
        ("product-rule-cross-term-dropped", u2 * v + u * v2),
        (
            "chain-rule-inner-derivative-dropped",
            u2 * v + _poly(2) * u1 * inner + u * _poly(n * (n - 1) * (a * X + b) ** (n - 2)),
        ),
    ]
    if not u2.is_zero and not v2.is_zero:
        a_cands.append(("second-derivatives-multiplied", u2 * v2))
    f_at = (u * v).eval(x0)
    b_cands = [("differentiated-once", f1.eval(x0)), ("evaluated-original-function", f_at)]
    b_cands += [(mid, poly.eval(x0)) for mid, poly in a_cands[1:]]

    c_cands = []
    if two_points:
        c_cands.append(("gave-other-root-of-two", _frac(roots[0])))
        c_cands.append(("sign-flip-both-roots", _frac(-roots[0])))  # both signs read wrongly, then the greater taken
        c_cands.append(("sign-flip-one-root", _frac(-greater)))
    else:
        c_cands.append(("repeated-root-taken-as-inflection", _frac(Rational(-b, a))))
        c_cands.append(("sign-flip-both-roots", _frac(-greater)))
    if greater.q != 1:  # the numerator of the root, left undivided by the coefficient of x
        c_cands.append(("forgot-to-divide-by-coefficient", Fraction(int(greater.p))))

    lead = {1: "", -1: "-"}.get(c, str(c))
    f_text = f"{lead}{x_pow(m)}{powered(lin_text(a, b), n)}"
    f2_text = _text(f2)
    stem = phrase(rng, _STEMS, f=f_text)
    c_prompt = "The graph of y = f(x) has a point of inflection wherever f''(x) = 0 and changes sign. " + (
        "Find the greater x-coordinate of the two points of inflection."
        if two_points
        else "Find the x-coordinate of the point of inflection."
    )
    part_a = expression_part(
        "Find f''(x), giving your answer in expanded form.",
        f2_text,
        tex(f2.as_expr()),
        2,
        [
            f"Write f as a product u v with u = {lead}{x_pow(m)} and v = {powered(lin_text(a, b), n)}.",
            f"f''(x) = u''v + 2u'v' + uv'' = {f2_text}.",
        ],
        [(mid, _text(p), p != f2) for mid, p in a_cands],
    )
    part_a["answer_form"] = "expanded"
    value_b = int(f2.eval(x0))
    part_b = number_part(
        f"Hence find the value of f''({x0}).",
        Fraction(value_b),
        0,
        1,
        [f"f''({x0}) = {value_b}."],
        [(mid, Fraction(int(val))) for mid, val in b_cands],
        follow_through=f"Subs(part_a, (x,), ({x0},))",
    )
    part_c = fraction_part(
        c_prompt,
        _frac(greater),
        2,
        [f"f''(x) = 0 at {', '.join(f'x = {r}' for r in roots)}.", f"f'' changes sign at x = {greater}."],
        c_cands,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_r3b_second_derivative", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_second_derivative(rng: random.Random, tier: str = "exam") -> dict:
    """f'' of a product, its value at a point, then a point of inflection."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
