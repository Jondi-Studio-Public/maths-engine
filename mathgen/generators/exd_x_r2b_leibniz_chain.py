"""Generates `exd_x_r2b_leibniz_chain` questions: a multi-part exam chain on Leibniz notation and the chain rule.

Skill: senior-use-the-leibniz-notation-for-the-derivative-frac (exam tier).

A quantity `y = u^n` where `u` is a polynomial in `x`, and `x` is changing at a constant rate:

(a) dy/du                                                         expression, 1 mark
(b) du/dx                                                         expression, 1 mark
(c) dy/dx = dy/du x du/dx, in terms of x                          expression, 2 marks, from (a) and (b)
(d) dy/dt when x has a stated value and dx/dt is given            number,     2 marks, from (c)

## Construction

`n`, the polynomial `u(x)`, the instant `x1` and the rate `dx/dt` are drawn, with `u(x1)` nonzero (so a negative
power is defined) and `dy/dt` nonzero. The derivative is re-derived by SymPy's `diff` of the composite `u(x)^n`
and compared with the product of the two derivatives the parts ask for, so the chain rule is checked, not assumed.
The answer to (d) is a whole number, so every follow-through target is stated to the same places.
"""

from __future__ import annotations

import sympy
from sympy import Rational, diff, symbols

from ..core import check, generator, join_terms, phrase
from .exd_x_r2b_support import Redraw, draw, exact_number_part, expression_part, same_function, typed

x, u = symbols("x u")
_N = (2, 3, 4, -1, -2)
_STEMS = (
    "A quantity y is given by y = {y}, where u = {ux}.",
    "Let y = {y}, where u = {ux}.",
    "The variable y depends on u by y = {y}, and u depends on x by u = {ux}.",
)
_RATE = (
    "Given that x is increasing at a constant rate of {r} {units} per second, find dy/dt, the rate of change of y "
    "with respect to time, at the instant x = {x1}.",
    "The value of x increases at a constant {r} {units} per second. Find the rate of change of y with respect to "
    "time, dy/dt, at the instant when x = {x1}.",
)


def _inner(rng):
    """u as a polynomial in x: a x^2 + c or x^3 + c (kept to two terms so the composite stays readable)."""
    c = rng.choice([v for v in range(-6, 7) if v])
    if rng.random() < 0.65:
        a = rng.randint(1, 3)
        return a * x**2 + c, join_terms([(a, "x^2"), (c, "")])
    return x**3 + c, join_terms([(1, "x^3"), (c, "")])


def _power_text(n: int) -> str:
    return {-1: "1/u", -2: "1/u^2"}.get(n, f"u^{n}")


def _build(rng):
    n = rng.choice(_N)
    inner, inner_text = _inner(rng)
    x1 = rng.choice([1, 2, 3, -1, -2])
    r = rng.randint(1, 4)
    d_inner = diff(inner, x)
    u1 = inner.subs(x, x1)
    if u1 == 0:
        raise Redraw("u vanishes at the instant, so a negative power is undefined")
    slope1 = n * u1 ** (n - 1) * d_inner.subs(x, x1)  # dy/dx at the instant, from the two factors
    rate = slope1 * r
    if rate == 0 or not sympy.Rational(rate).is_integer or abs(rate) > 5000:
        raise Redraw("a rate that is zero, not whole or too large")
    outer = u**n
    d_outer = diff(outer, u)
    d_total = d_outer.subs(u, inner) * d_inner
    # Independent route: differentiate the composite directly, and read the rate off it.
    check(same_function(diff(inner**n, x), d_total), "the chain rule product disagrees with diff")
    check(diff(inner**n, x).subs(x, x1) * r == rate, "dy/dt disagrees with the composite derivative")
    stem = phrase(rng, _STEMS, y=_power_text(n), ux=inner_text)

    terms_inner = sympy.Poly(inner, x).terms()
    a_wrong = [
        (n * u**n, "exponent-not-lowered"),
        (u ** (n - 1), "forgot-power-multiplier"),
        (u ** (n + 1) / (n + 1) if n != -1 else None, "integrated-instead-of-differentiated"),
    ]
    part_a = expression_part(
        "Find dy/du.",
        d_outer,
        1,
        [f"y = {_power_text(n)}, so dy/du = {typed(d_outer)}."],
        [w for w in a_wrong if w[0] is not None],
    )

    const = sympy.Poly(inner, x).coeff_monomial(1)
    b_wrong = [
        (sympy.integrate(inner, x), "integrated-instead-of-differentiated"),
        (d_inner + const, "derivative-of-constant-not-zero"),
        (sum(c * x ** (m[0] - 1) for m, c in terms_inner if m[0] > 0), "forgot-power-multiplier"),
    ]
    part_b = expression_part("Find du/dx.", d_inner, 1, [f"u = {inner_text}, so du/dx = {typed(d_inner)}."], b_wrong)

    c_wrong = [
        (d_outer.subs(u, inner), "forgot-inner-derivative"),
        (d_outer.subs(u, inner) + d_inner, "chain-rule-terms-added"),
        (d_outer.subs(u, inner) / d_inner, "inverted-ratio"),
    ]
    part_c = expression_part(
        "Hence find dy/dx in terms of x.",
        d_total,
        2,
        [f"dy/dx = ({typed(d_outer)}) × ({typed(d_inner)}) with u = {inner_text}.", f"dy/dx = {typed(d_total)}."],
        [(e, t) for e, t in c_wrong],
        follow_through=f"Subs(part_a, (u,), ({inner},))*part_b",
    )

    d_wrong = [
        (slope1, "dx-dt-factor-omitted"),
        (slope1 / r, "inverted-ratio"),
        (d_outer.subs(u, u1) * r, "forgot-inner-derivative"),
        (inner.subs(x, x1) ** n * r, "value-instead-of-rate"),
    ]
    part_d = exact_number_part(
        phrase(rng, _RATE, r=r, units="unit" if r == 1 else "units", x1=x1),
        Rational(rate),
        2,
        [f"dy/dt = dy/dx × dx/dt = {slope1} × {r} = {rate}."],
        [(Rational(v), t) for v, t in d_wrong],
        follow_through=f"Subs(part_c, (x,), ({x1},))*{r}",
        suffix="",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_leibniz_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_leibniz_chain(rng, tier: str = "exam"):
    """Leibniz notation in a chain: dy/du, du/dx, dy/dx by the chain rule, and dy/dt for a given dx/dt."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
