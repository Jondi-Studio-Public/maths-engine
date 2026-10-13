"""Generates `exd_x_r2a_first_principles_tangent` questions: a multi-part exam chain starting from the definition.

Skill: senior-define-the-derivative-f-left-x-right-as (exam tier). Part (a) asks for f'(x) from the limit definition
for a function whose difference quotient needs a method (a common denominator, the conjugate, or an expanded cube).
The rest of the question uses the result:

(a) find f'(x) from first principles                              expression, 2 marks
(b) hence the gradient of the graph at x = x0                     fraction,   1 mark,  Subs(part_a, x, x0)
(c) hence the equation of the tangent at x = x0                   expression, 2 marks, part_b*(x - x0) + f(x0)
(d) hence where that tangent meets the x-axis                     fraction,   1 mark,  from the learner's line

## Functions

``a/(x + b)``, ``a/x^2``, ``sqrt(cx + b)`` and ``ax^3 + bx``. For the square root, x0 is chosen so that cx0 + b is
a perfect square, which keeps f(x0) and f'(x0) rational. f'(x) is known by the power and chain rules and is checked
against the limit of the difference quotient at a sample point with SymPy (an independent route).
"""

from __future__ import annotations

import random

import mpmath
from sympy import Rational, lambdify, latex, sqrt

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    YF,
    X,
    Redraw,
    build_with_redraw,
    expr_part,
    prose,
    tangent_tail,
    x_equals,
)

_PATTERNS = (
    "Let {fx}. The tangent to the graph of {yf} at the point where {xv} is required.",
    "The function f is given by {fx}. A tangent is drawn to the graph of {yf} at the point where {xv}.",
    "Consider {fx}. The graph of {yf} has a tangent at {xv}.",
)


def _reciprocal(rng):
    a = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
    b = rng.randint(1, 5)
    x0 = rng.choice([m for m in range(-3, 4) if m + b != 0])
    f = a / (X + b)
    fprime = -a / (X + b) ** 2
    wrong = [
        ("denominator-not-squared", -a / (X + b)),
        ("sign-reversed-result", a / (X + b) ** 2),
        ("shift-dropped-in-denominator", -a / X**2),
        ("exponent-moved-wrong-way", -a / (X + b) ** 3),
    ]
    return (
        f,
        fprime,
        x0,
        f"{a}/(x + {b})",
        wrong,
        "Combine the fractions over the common denominator (x + h + b)(x + b), then let h -> 0.",
    )


def _reciprocal_square(rng):
    a = rng.choice([-4, -3, -2, 1, 2, 3, 4, 5])
    x0 = rng.choice([-3, -2, -1, 1, 2, 3])
    f = a / X**2
    fprime = -2 * a / X**3
    wrong = [
        ("forgot-power-multiplier", -a / X**3),
        ("sign-reversed-result", 2 * a / X**3),
        ("exponent-not-lowered", -2 * a / X**2),
        ("wrong-power-in-reciprocal", -a / X),
    ]
    return (
        f,
        fprime,
        x0,
        f"{a}/x^2",
        wrong,
        "Use a common denominator and the difference of squares: (x+h)^2 - x^2 = h(2x + h).",
    )


def _root(rng):
    c = rng.choice([2, 3, 4, 5, 6])
    b = rng.randint(1, 9)
    s = rng.randint(1, 6)
    x0 = Rational(s * s - b, c)
    if x0.q != 1 or x0 == 0 or abs(x0) > 12:
        raise Redraw
    f = sqrt(c * X + b)
    fprime = c / (2 * sqrt(c * X + b))
    wrong = [
        ("forgot-inner-derivative", 1 / (2 * sqrt(c * X + b))),
        ("forgot-factor-two-from-conjugate", c / sqrt(c * X + b)),
        ("shift-dropped-in-denominator", c / (2 * sqrt(X))),
        ("root-left-in-numerator", c * sqrt(c * X + b) / 2),
    ]
    return (
        f,
        fprime,
        int(x0),
        f"√({c}x + {b})",
        wrong,
        "Multiply the numerator and denominator by the conjugate, then let h -> 0.",
    )


def _cubic(rng):
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.choice([m for m in range(-6, 7) if m != 0])
    x0 = rng.choice([-2, -1, 1, 2])
    f = a * X**3 + b * X
    fprime = 3 * a * X**2 + b
    wrong = [
        ("dropped-term-derivative", 3 * a * X**2),
        ("forgot-power-multiplier", a * X**2 + b),
        ("exponent-not-lowered", 3 * a * X**3 + b),
        ("dropped-term-derivative", 3 * a * X**2 + b * X),
    ]
    return (
        f,
        fprime,
        x0,
        join_terms([(a, "x^3"), (b, "x")]),
        wrong,
        "Expand (x + h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3 and cancel h.",
    )


_FAMILIES = (_reciprocal, _reciprocal_square, _root, _cubic)


def _quotient_tends_to(f, fprime, point) -> bool:
    """Whether [f(point + h) - f(point)]/h, with h = 10^-25 and 60-digit arithmetic, matches `fprime(point)`."""
    with mpmath.workdps(60):
        fn, gn = lambdify(X, f, "mpmath"), lambdify(X, fprime, "mpmath")
        at = mpmath.mpf(point.p) / point.q
        h = mpmath.mpf(10) ** -25
        return abs((fn(at + h) - fn(at)) / h - gn(at)) < mpmath.mpf(10) ** -15


def _build(rng: random.Random):
    f, fprime, x0, f_text, wrong_a, hint = rng.choice(_FAMILIES)(rng)
    # Independent route: the difference quotient of the definition, at a sample point with a tiny h, to 15 digits.
    point = Rational(3, 2) if x0 != Rational(3, 2) else Rational(5, 2)
    check(_quotient_tends_to(f, fprime, point), "the difference quotient disagrees with the derivative")
    f0, m = f.subs(X, x0), fprime.subs(X, x0)
    if f0 == 0 or m == 0 or not (f0.is_Rational and m.is_Rational):
        raise Redraw
    rng.shuffle(wrong_a)
    part_a = expr_part(
        dict(
            prompt="Find f'(x) from first principles.",
            prompt_latex=r"\text{Find }f'(x)\text{ from first principles.}",
        ),
        fprime,
        2,
        [hint, "Simplify the difference quotient until h cancels, then take the limit as h tends to 0."],
        wrong_a,
    )
    parts = [part_a, *tangent_tail(f0, m, x0, variant="tangent_x", m_at_neg_x0=fprime.subs(X, -x0))]
    stem, stem_latex = prose(rng, _PATTERNS, fx=(f"f(x) = {f_text}", f"f(x) = {latex(f)}"), yf=YF, xv=x_equals(x0))
    return dict(text=stem, latex=stem_latex, parts=parts)


@generator("exd_x_r2a_first_principles_tangent", level=16, presets={"exam": {}})
def exd_x_r2a_first_principles_tangent(rng: random.Random):
    """From the definition: f'(x), the gradient at x0, the tangent, where it meets the x-axis."""
    return build_with_redraw(rng, _build)
