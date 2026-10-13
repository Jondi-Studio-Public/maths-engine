"""Generates `exd_x_r2b_antiderivative_chain` questions: a multi-part exam chain on recovering a curve from its gradient.

Skill: integrate-polynomial (exam tier).

A curve `y = f(x)` has gradient function `f'(x) = 3a x^2 + 2b x + k`, where `k` is an unknown constant. The stem gives
the gradient of the tangent at a stated x and a point the curve passes through:

(a) find k from the stated gradient                               fraction,   2 marks
(b) integrate to find f(x), the constant fixed by the point       expression, 2 marks, from (a)
(c) the value of f at a stated x                                  fraction,   1 mark,  from (b)
(d) the tangent at another stated x, as y = mx + c                expression, 2 marks, from (a) and (b)

## Construction: choose f first

`a`, `b`, `k` and the constant `C` of `f(x) = a x^3 + b x^2 + k x + C` are drawn, then the tangent gradient at `x = p`
and the point `(q, f(q))` are read off. Every key is re-derived with SymPy (`diff` and `integrate`, the constant
fixed by the point), never from the construction.
"""

from __future__ import annotations

import sympy
from sympy import Rational, diff, expand, symbols

from ..core import check, generator, join_terms, paren_negative, phrase
from .exd_x_r2b_support import Redraw, draw, expression_part, fraction_part, typed

x = symbols("x")
_STEMS = (
    "A curve y = f(x) has gradient function f'(x) = {g}, where k is a constant. The tangent to the curve at the "
    "point where x = {p} has gradient {grad}, and the curve passes through the point ({q}, {t}).",
    "The gradient function of a curve y = f(x) is f'(x) = {g}, where k is a constant. At x = {p} the gradient of "
    "the curve is {grad}, and the point ({q}, {t}) lies on the curve.",
    "For a curve with equation y = f(x), f'(x) = {g}, where k is a constant. The gradient at x = {p} is {grad}, and "
    "the curve passes through ({q}, {t}).",
)


def _line(slope, intercept) -> str:
    return "y = " + typed(slope * x + intercept)


def _build(rng):
    a, b = rng.randint(1, 3), rng.randint(-3, 3)
    k = rng.choice([v for v in range(-9, 10) if v])
    c = rng.choice([v for v in range(-9, 10) if v])
    p = rng.choice([1, 2, 3, -1, -2])
    q = rng.choice([1, 2, -1, -2])
    x1, x2 = rng.sample([1, 2, 3, -1, -2, -3, 0], 2)
    f = a * x**3 + b * x**2 + k * x + c
    grad = int(diff(f, x).subs(x, p))
    t = int(f.subs(x, q))
    if grad == 0 or t == 0:
        raise Redraw("a zero gradient or ordinate makes follow-through trivial")

    # ---- independent derivation: integrate the stated gradient function and fix the constants ---------------------
    kk = symbols("k")
    gradient = 3 * a * x**2 + 2 * b * x + kk
    line_in_k = sympy.Poly(gradient.subs(x, p) - grad, kk)  # c1 k + c0 = 0
    check(-line_in_k.coeff_monomial(1) / line_in_k.coeff_monomial(kk) == k, "k disagrees with the stated gradient")
    anti = sympy.Poly(gradient.subs(kk, k), x).integrate().as_expr()
    f_found = expand(anti + (t - anti.subs(x, q)))
    check(expand(f_found - f) == 0, "the integrated curve is not the constructed one")
    value1, value2 = int(f.subs(x, x1)), int(f.subs(x, x2))
    slope2 = int(diff(f, x).subs(x, x2))
    tangent = expand(slope2 * (x - x2) + value2)
    if slope2 == 0 or value1 == 0:
        raise Redraw("a zero slope or value")

    g_text = join_terms([(3 * a, "x^2"), (2 * b, "x"), (1, "k")])
    stem = phrase(rng, _STEMS, g=g_text, p=p, grad=grad, q=q, t=t)

    # ---- (a) k --------------------------------------------------------------------------------------------------
    a_wrong = [
        (-(grad) + 3 * a * p * p + 2 * b * p, "sign-error-moving-term"),  # k = f' stated minus the wrong way round
        (grad - 3 * a * p * p, "left-out-a-term-when-solving-for-constant"),
        (grad - 2 * b * p, "left-out-a-term-when-solving-for-constant"),
        (grad, "left-out-a-term-when-solving-for-constant"),
    ]
    part_a = fraction_part(
        "Find the value of k.",
        k,
        2,
        [f"f'({p}) = {join_terms([(3 * a * p * p, ''), (2 * b * p, ''), (1, 'k')])} = {grad}, so k = {k}."],
        a_wrong,
    )

    # ---- (b) f(x) --------------------------------------------------------------------------------------------------
    no_constant = expand(a * x**3 + b * x**2 + k * x)
    not_divided = expand(3 * a * x**3 + 2 * b * x**2 + k * x + (t - (3 * a * q**3 + 2 * b * q**2 + k * q)))
    differentiated = expand(diff(gradient.subs(kk, k), x))
    wrong_sign_c = expand(a * x**3 + b * x**2 + k * x - c)
    b_wrong = [
        (no_constant, "forgot-initial-condition"),
        (not_divided, "integration-power-not-divided"),
        (differentiated, "differentiated-instead-of-integrated"),
        (wrong_sign_c, "sign-error-moving-term"),
    ]
    part_b = expression_part(
        "Find f(x).",
        f,
        2,
        [
            f"f(x) = {join_terms([(a, 'x^3'), (b, 'x^2'), (k, 'x'), (1, 'C')])}.",
            f"f({q}) = {t} gives C = {c}, so f(x) = {typed(f)}.",
        ],
        b_wrong,
        follow_through=(
            f"{a}*x**3 + {paren_negative(b)}*x**2 + part_a*x + ({t} - ({a}*{paren_negative(q)}**3 + "
            f"{paren_negative(b)}*{paren_negative(q)}**2 + part_a*{paren_negative(q)}))"
        ),
    )

    # ---- (c) f at x1 -------------------------------------------------------------------------------------------------
    c_wrong = [
        (int(diff(f, x).subs(x, x1)), "evaluated-wrong-function-order"),
        (value1 - c, "dropped-constant-term"),
        (int(f.subs(x, -x1)) if x1 else None, "substitution-sign-error"),
    ]
    part_c = fraction_part(
        f"Find the value of f({x1}).",
        value1,
        1,
        [f"f({x1}) = {value1}."],
        [w for w in c_wrong if w[0] is not None],
        follow_through=f"Subs(part_b, (x,), ({x1},))",
    )

    # ---- (d) where the tangent at x2 meets the y-axis ------------------------------------------------------------------
    intercept = value2 - slope2 * x2
    check(int(tangent.subs(x, 0)) == intercept, "the tangent's intercept disagrees")
    if intercept == 0 or intercept == value2:
        raise Redraw("an intercept that gives the point's value away")
    d_wrong = [
        (value2, "y-value-as-intercept"),
        (value2 + slope2 * x2, "sign-slip-point-slope"),
        (-slope2 * x2, "forgot-point-y-value"),
        (slope2, "intercept-and-gradient-confused"),
    ]
    part_d = fraction_part(
        f"The tangent to the curve at the point where x = {x2} is drawn. Find the y-coordinate of the point where this tangent crosses the y-axis.",
        intercept,
        2,
        [
            f"f'({x2}) = {slope2} and f({x2}) = {value2}.",
            f"The tangent is y - {paren_negative(value2)} = {paren_negative(slope2)}(x - {paren_negative(x2)}); "
            f"at x = 0, y = {intercept}.",
        ],
        d_wrong,
        follow_through=(
            f"Subs(part_b, (x,), ({x2},)) - ({3 * a}*{paren_negative(x2)}**2 + {2 * b}*{paren_negative(x2)} + part_a)"
            f"*{paren_negative(x2)}"
        ),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_antiderivative_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_antiderivative_chain(rng, tier: str = "exam"):
    """A curve from its gradient function with an unknown constant: k, f(x), a value, a tangent."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
