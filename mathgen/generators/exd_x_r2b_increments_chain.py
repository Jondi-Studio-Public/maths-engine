"""Generates `exd_x_r2b_increments_chain` questions: a multi-part exam chain on the increments formula.

Skill: senior-use-the-increments-formula-delta-y-cong-frac (exam tier).

A quantity `y = f(x)` (a cubic, a quartic, or a product of two factors) and a small change `dx` in `x` from `x0`:

(a) find dy/dx                                                    expression, 2 marks
(b) estimate the change, dy/dx at x0 times dx                     fraction,   1 mark,  from (a)
(c) estimate the new value of y                                   fraction,   1 mark,  y0 + (b)
(d) the exact new value, and the error of the estimate (c)        fraction,   2 marks, exact value - (c)

## Construction

The function is drawn, then `dx` is a small rational (`1/10` to `1/100`, either sign) and `x0` a small positive
integer. The error `f(x0 + dx) - (f(x0) + f'(x0) dx)` is a rational computed in exact arithmetic, and is never zero
(the cubic and quartic always have a curvature term). Every key is re-derived with SymPy: `diff` for the derivative,
`Rational` arithmetic for the estimate and the exact value.
"""

from __future__ import annotations

import sympy
from fractions import Fraction

from sympy import Rational, diff, expand, symbols

from ..core import check, generator, join_terms, phrase, rational_text
from .exd_x_r2b_support import (
    Redraw,
    draw,
    decimal_places,
    exact_number_part,
    expression_part,
    number_part,
    clear_of_boundary,
)

x = symbols("x")
_DX = (Rational(1, 10), Rational(1, 5), Rational(1, 2), Rational(1, 4))
_DX_QUARTIC = (Rational(1, 10), Rational(1, 5), Rational(1, 2))
_STEMS = (
    "A quantity y is given by y = {f}. The value of x changes from x = {x0} by a small amount δx = {dx}.",
    "Let y = {f}. Starting from x = {x0}, x is changed by δx = {dx}.",
    "The variable y depends on x through y = {f}. At x = {x0}, a small change δx = {dx} is made to x.",
)


def _cubic(rng):
    a, b, c, d = rng.randint(1, 3), rng.randint(-4, 4), rng.randint(-6, 6), rng.choice([v for v in range(-9, 10) if v])
    f = a * x**3 + b * x**2 + c * x + d
    return f, join_terms([(a, "x^3"), (b, "x^2"), (c, "x"), (d, "")]), None


def _quartic(rng):
    a, b, c = rng.randint(1, 2), rng.choice([v for v in range(-5, 6) if v]), rng.choice([v for v in range(-9, 10) if v])
    return a * x**4 + b * x**2 + c, join_terms([(a, "x^4"), (b, "x^2"), (c, "")]), None


def _product(rng):
    p, q = rng.choice([v for v in range(-4, 5) if v]), rng.randint(1, 6)
    return (x + p) * (x**2 + q), f"(x {'+' if p > 0 else '-'} {abs(p)})(x^2 + {q})", (1, 2 * x)


_FAMILIES = (_cubic, _quartic, _product)


def _terms(f):
    """The `(coefficient, power)` terms of the expanded polynomial `f`."""
    poly = sympy.Poly(expand(f), x)
    return [(int(c), int(m[0])) for m, c in poly.terms()]


def _dec(value) -> str:
    """An exact terminating decimal as text (`-0.05`), or the integer when whole."""
    value = Rational(value)
    places = decimal_places(Fraction(int(value.p), int(value.q)))
    if places is None:
        raise Redraw("not a terminating decimal")
    return f"{float(value):.{places}f}" if places else str(int(value))


def _from_terms(terms):
    return sum((c * x**n for c, n in terms), sympy.Integer(0))


def _build(rng):
    f, f_text, factor_derivatives = rng.choice(_FAMILIES)(rng)
    x0 = rng.randint(2, 6)
    quartic = f.has(x**4)
    dx = rng.choice(_DX_QUARTIC if quartic else _DX) * rng.choice([1, -1])

    # ---- independent derivation ---------------------------------------------------------------------------------
    deriv = expand(diff(f, x))
    y0 = f.subs(x, x0)
    slope = deriv.subs(x, x0)
    est_change = slope * dx
    est_new = y0 + est_change
    exact_new = f.subs(x, x0 + dx)
    error = exact_new - est_new
    taylor = sum(diff(f, x, n).subs(x, x0) / sympy.factorial(n) * dx**n for n in range(2, 6))
    check(error == taylor, "the error is not the sum of the higher Taylor terms")
    if abs(error) < Rational(1, 100) or est_change == 0 or slope == 0:
        raise Redraw("degenerate draw")
    if not clear_of_boundary(Fraction(int(error.p), int(error.q)), 3):
        raise Redraw("the error is too close to a rounding tie")
    terms = _terms(f)

    stem = phrase(rng, _STEMS, f=f_text, x0=x0, dx=_dec(dx))

    # ---- (a) dy/dx -----------------------------------------------------------------------------------------------
    a_wrong = [
        (_from_terms([(c * n, n) for c, n in terms if n]), "exponent-not-lowered"),
        (_from_terms([(c, n - 1) if n else (c, 0) for c, n in terms]), "forgot-power-multiplier"),
        (_from_terms([(Rational(c, n + 1), n + 1) for c, n in terms]), "integrated-instead-of-differentiated"),
        (deriv + terms_constant(terms), "derivative-of-constant-not-zero"),
    ]
    if factor_derivatives:  # a product: the derivatives of the two factors multiplied together
        a_wrong.insert(0, (factor_derivatives[0] * factor_derivatives[1], "product-rule-multiplied-derivatives"))
    part_a = expression_part(
        "Find dy/dx.",
        deriv,
        2,
        [f"dy/dx = {sympy.sstr(deriv, order='lex').replace('**', '^')}."],
        [(expand(e), t) for e, t in a_wrong],
    )

    # ---- (b) the estimated change -----------------------------------------------------------------------------------
    b_wrong = [
        (y0 * dx, "value-instead-of-rate"),
        (slope, "dx-not-multiplied"),
        (slope / dx, "inverted-ratio"),
        (deriv.subs(x, x0 + dx) * dx, "derivative-at-new-point"),
    ]
    part_b = exact_number_part(
        f"Using the rate of change of y at x = {x0}, estimate the change in y.",
        est_change,
        1,
        [f"At x = {x0}: dy/dx = {slope}, so δy ≅ {slope} × {_dec(dx)} = {_dec(est_change)}."],
        b_wrong,
        follow_through=f"Subs(part_a, (x,), ({x0},))*Rational({dx.p}, {dx.q})",
        min_places=2,
    )

    # ---- (c) the estimated new value ------------------------------------------------------------------------------------
    c_wrong = [
        (est_change, "change-instead-of-new-value"),
        (y0 - est_change, "sign-reversed-result"),
        (y0 + slope, "dx-not-multiplied"),
    ]
    part_c = exact_number_part(
        "Hence estimate the new value of y.",
        est_new,
        1,
        [f"y ≅ {y0} + {_dec(est_change)} = {_dec(est_new)}."],
        c_wrong,
        follow_through=f"{y0} + part_b",
        min_places=2,
    )

    # ---- (d) the exact value and the error --------------------------------------------------------------------------------
    d_wrong = [
        (-error, "sign-reversed-result"),
        (exact_new, "exact-value-instead-of-error"),
        (exact_new - y0, "exact-value-instead-of-error"),
    ]
    part_d = number_part(
        "Find the exact new value of y and hence the error in your estimate in (c), that is, the exact value "
        "minus your estimate.",
        Fraction(int(error.p), int(error.q)),
        3,
        2,
        [
            f"Exact: y = f({_dec(x0 + dx)}) = {_dec(exact_new)}.",
            f"Error = {_dec(exact_new)} - {_dec(est_new)} = {_dec(error)}.",
        ],
        d_wrong,
        follow_through=f"Rational({exact_new.p}, {exact_new.q}) - part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


def terms_constant(terms):
    """The constant term of the polynomial, as an expression."""
    return _from_terms([(c, 0) for c, n in terms if n == 0])


@generator("exd_x_r2b_increments_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_increments_chain(rng, tier: str = "exam"):
    """The increments formula in a chain: dy/dx, the estimated change, the estimated value, the exact error."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
