"""Exam-tier source for ``integration-by-substitution``: the substitution is not given.

At the lower tiers the prompt names the substitution (``using the substitution u = ...``).  Here the learner
must spot it, and one family also needs a second step after substituting.  Four families:

- ``k x sqrt(x + d)``: substitute ``u = x + d``, then rewrite the leftover ``x`` as ``u - d`` before integrating;
- ``k cos(x) sin^n(x)``: the derivative of the inner function sits beside it as a factor;
- ``k sin(x) / cos^n(x)``: the same, with a reciprocal power and a sign from ``d(cos)/dx``;
- ``k x / sqrt(x^2 + d)``: a non-linear inner function with a half-integer power.

Built backwards from the antiderivative ``F``; ``F' == integrand`` is re-checked numerically with
sympy at fixed points.

Distractors, per family:

- ``x sqrt(x + d)``: forgetting to rewrite the leftover ``x`` (treating it as a constant), or substituting
  ``x = u`` without the ``- d`` shift;
- ``cos(x) sin^n(x)``: not dividing by the new power, dividing by ``n`` instead of ``n + 1``, or choosing the wrong
  trigonometric family;
- ``sin(x) / cos^n(x)``: losing the minus sign from ``du = -sin(x) dx``, not dividing by the reduced power, or
  adding one to the exponent instead of taking it away;
- ``x / sqrt(x^2 + d)``: forgetting the factor of one half from ``du = 2x dx``, halving twice, or raising the
  power of the reciprocal root the wrong way.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, cos, diff, sin, sqrt

from ..core import check, check_distinct, fmt_signed, generator, join_terms, phrase, pick_distractors

_X = Symbol("x")

_PATTERNS = (
    "Find ∫ {integrand} dx.",
    "Evaluate the indefinite integral ∫ {integrand} dx.",
    "Determine ∫ {integrand} dx.",
)


def _display(expr) -> str:
    """Learner-facing text: ``^`` for powers, never Python's ``**``."""
    return str(expr).replace("**", "^")


_PROBES = (Rational(3, 7), Rational(11, 5), Rational(17, 4), Rational(13, 3))


def _agree(left, right) -> bool:
    return all(abs((left - right).subs(_X, t).evalf(40)) < 1e-20 for t in _PROBES)


def _shifted_root(rng):
    k = rng.choice([3, 6, 15])
    d = rng.choice([1, 2, 3, 4, 5, -1])
    u = _X + d
    integrand = k * _X * sqrt(u)
    answer = k * (Rational(2, 5) * u ** Rational(5, 2) - Rational(2, 3) * d * u ** Rational(3, 2))
    inner = join_terms([(1, "x"), (d, "")])
    text = f"{fmt_signed(k, 'x')}√({inner})"
    wrong = [
        k * _X * Rational(2, 3) * u ** Rational(3, 2),
        k * Rational(2, 5) * u ** Rational(5, 2),
        k * (Rational(2, 5) * u ** Rational(5, 2) + Rational(2, 3) * d * u ** Rational(3, 2)),
    ]
    return text, integrand, answer, wrong


def _sine_power(rng):
    k = rng.randint(2, 7)
    n = rng.randint(2, 5)
    integrand = k * cos(_X) * sin(_X) ** n
    answer = k * sin(_X) ** (n + 1) / (n + 1)
    text = f"{fmt_signed(k, 'cos(x)')}sin^{n}(x)"
    wrong = [k * sin(_X) ** (n + 1), k * sin(_X) ** (n + 1) / n, k * cos(_X) ** (n + 1) / (n + 1)]
    return text, integrand, answer, wrong


def _cosine_reciprocal(rng):
    k = rng.randint(2, 7)
    n = rng.choice([3, 4, 5])
    integrand = k * sin(_X) / cos(_X) ** n
    answer = k / ((n - 1) * cos(_X) ** (n - 1))
    text = f"{k}sin(x)/cos^{n}(x)"
    wrong = [-answer, k / cos(_X) ** (n - 1), k / ((n + 1) * cos(_X) ** (n + 1))]
    return text, integrand, answer, wrong


def _quadratic_root(rng):
    k = rng.randint(2, 9)
    d = rng.randint(1, 8)
    root = sqrt(_X**2 + d)
    integrand = k * _X / root
    answer = k * root
    text = f"{k}x/√(x^2 + {d})"
    wrong = [2 * k * root, k * root / 2, k * (_X**2 + d) ** Rational(3, 2) / 3]
    return text, integrand, answer, wrong


_FAMILIES = (_shifted_root, _sine_power, _cosine_reciprocal, _quadratic_root)


def _exam(rng: random.Random) -> dict:
    integrand_text, integrand, answer, wrong = rng.choice(_FAMILIES)(rng)
    check(_agree(diff(answer, _X), integrand), "derivative of the antiderivative is not the integrand")
    check_distinct(answer, *wrong)
    text = phrase(rng, _PATTERNS, integrand=integrand_text)
    answer_text = f"{_display(answer)} + C"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=str(answer),
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=pick_distractors(answer_text, [f"{_display(w)} + C" for w in wrong]),
    )


@generator("exd_x_c2_integration_by_substitution", level=15, presets={"exam": dict(tier="exam")})
def exd_x_c2_integration_by_substitution(rng: random.Random, tier: str = "exam") -> dict:
    """Integrate a composite form where the learner must find the substitution."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
