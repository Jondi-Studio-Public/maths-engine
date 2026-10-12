"""Exam tier for a simple Fourier coefficient: a quadratic-plus-linear function and two integrations by parts.

The practice tiers integrate x, x^2 or a constant against one cosine or sine. Here the function is
f(x) = a x^2 + b x on [0, pi], so the sine integral needs integration by parts twice for the x^2 term and
once for the x term, and the answer depends on the parity of n through (-1)^n. The question is a short chain,
as an exam sets it: evaluate the integral exactly, then use it for the coefficient b_n = (2/pi) times the
integral. The second part carries `follow_through`, so a slip in the integral costs only the first part.

## Construction

For integer n,
    integral of x sin(nx) over [0, pi] = -pi (-1)^n / n
    integral of x^2 sin(nx) over [0, pi] = -pi^2 (-1)^n / n + 2((-1)^n - 1) / n^3
which are re-derived here with SymPy's integrator and with a numerical quadrature before the question is
returned.
"""

from __future__ import annotations

import random
from fractions import Fraction

import mpmath
import sympy

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    check_equivalent,
    generator,
    join_terms,
    pick_distractors,
)

_X = sympy.Symbol("x", real=True)
_COEFFS = [c for c in range(-6, 7) if c]

#: A value of the form c2 pi^2 + c1 pi + c0 + c_1 / pi, held as {power of pi: Fraction}.
_Series = dict[int, Fraction]


def _add(*terms: _Series) -> _Series:
    out: _Series = {}
    for term in terms:
        for power, coeff in term.items():
            out[power] = out.get(power, Fraction(0)) + coeff
    return {power: coeff for power, coeff in out.items() if coeff}


def _scale(term: _Series, factor: Fraction, shift: int = 0) -> _Series:
    return {power + shift: coeff * factor for power, coeff in term.items()}


def _expr(term: _Series):
    return sum((sympy.Rational(c.numerator, c.denominator) * sympy.pi**p for p, c in term.items()), sympy.Integer(0))


def _text(term: _Series) -> str:
    """A grader-friendly exact expression: `**` as `^`, `pi` spelled out."""
    return str(sympy.expand(_expr(term))).replace("**", "^")


def _integral(a: int, b: int, n: int, parity: int) -> _Series:
    """Integral of (a x^2 + b x) sin(n x) over [0, pi], with cos(n pi) taken as (-1)^parity."""
    s = (-1) ** parity
    return _add(
        {2: Fraction(-a * s, n)},
        {0: Fraction(2 * a * (s - 1), n**3)},
        {1: Fraction(-b * s, n)},
    )


def _tagged(key: _Series, candidates: list[tuple[str, _Series]]) -> list:
    out = []
    for ident, value in candidates:
        if value == key or not value:
            continue
        out.append(Distractor(_text(value), ident, sympy.expand(_expr(value))))
    return pick_distractors(_text(key), out, 3)


def _poly(a: int, b: int) -> str:
    return join_terms([(a, "x^2"), (b, "x")])


_STEMS = (
    "On [0, π], let f(x) = {poly}. The Fourier sine coefficient of f is b_n = (2/π) ∫₀^π f(x) sin(nx) dx.",
    "The function f(x) = {poly} is defined on [0, π]. Its Fourier sine coefficient is b_n = (2/π) ∫₀^π f(x) sin(nx) dx.",
    "A function is given on [0, π] by f(x) = {poly}. Its Fourier sine coefficients are b_n = (2/π) ∫₀^π f(x) sin(nx) dx.",
)


@generator("exd_x_u2_fourier_by_parts", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u2_fourier_by_parts(rng: random.Random, tier: str = "exam") -> dict:
    """Exact sine-integral of a x^2 + b x for a given n, then the coefficient b_n. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(100, "coefficients with at least two distractors per part"):
        a, b = rng.choice(_COEFFS), rng.choice(_COEFFS)
        n = rng.randint(1, 8)
        parity = n % 2
        integral = _integral(a, b, n, parity)

        # Independent routes: an antiderivative checked by differentiation, and a numerical quadrature.
        f = a * _X**2 + b * _X
        anti = (
            -f * sympy.cos(n * _X) / n + (2 * a * _X + b) * sympy.sin(n * _X) / n**2 + 2 * a * sympy.cos(n * _X) / n**3
        )
        check(sympy.expand(sympy.diff(anti, _X) - f * sympy.sin(n * _X)) == 0, "antiderivative check failed")
        numeric = mpmath.quad(lambda t: (a * t * t + b * t) * mpmath.sin(n * t), [0, mpmath.pi])
        check(abs(float(numeric) - float(sympy.N(_expr(integral), 30))) < 1e-9, "numerical integral mismatch")

        coefficient = _scale(integral, Fraction(2), -1)
        other = _integral(a, b, n, 1 - parity)  # cos(n pi) taken with the wrong parity
        no_second_step = _add(
            {2: Fraction(-a * (-1) ** parity, n)},
            {1: Fraction(-b * (-1) ** parity, n)},
        )
        i_distractors = _tagged(
            integral,
            [
                ("sign-lost-at-n-pi", other),
                ("integration-by-parts-incomplete", no_second_step),
                ("sign-error-in-by-parts", _scale(integral, Fraction(-1))),
            ],
        )
        c_distractors = _tagged(
            coefficient,
            [
                ("missing-normalisation-factor", integral),
                ("missing-normalisation-factor", _scale(integral, Fraction(2))),
                ("missing-normalisation-factor", _scale(integral, Fraction(1), -1)),
                ("sign-lost-at-n-pi", _scale(other, Fraction(2), -1)),
            ],
        )
        if len(i_distractors) < 2 or len(c_distractors) < 2:
            continue

        stem = rng.choice(_STEMS).format(poly=_poly(a, b))
        i_text, c_text = _text(integral), _text(coefficient)
        i_expr, c_expr = sympy.expand(_expr(integral)), sympy.expand(_expr(coefficient))
        return dict(
            text=stem,
            parts=[
                dict(
                    prompt=f"Use integration by parts to find the exact value of ∫₀^π f(x) sin({n if n > 1 else ''}x) dx.",
                    answer_kind="expression",
                    answer=i_expr,
                    answer_text=i_text,
                    answer_latex=sympy.latex(i_expr),
                    marks=3,
                    working=[
                        f"∫₀^π x sin({n if n > 1 else ''}x) dx = -π(-1)^{n}/{n}, and ∫₀^π x² sin({n if n > 1 else ''}x) dx = "
                        f"-π²(-1)^{n}/{n} + 2((-1)^{n} - 1)/{n}³ (two integrations by parts).",
                        f"With f(x) = {_poly(a, b)} the integral is {i_text}.",
                    ],
                    distractors=i_distractors,
                ),
                dict(
                    prompt=f"Hence find b_{n} exactly.",
                    answer_kind="expression",
                    answer=c_expr,
                    answer_text=c_text,
                    answer_latex=sympy.latex(c_expr),
                    marks=1,
                    follow_through="2*part_a/pi",
                    working=[f"b_{n} = (2/π) × ({i_text}) = {c_text}."],
                    distractors=c_distractors,
                ),
            ],
        )
