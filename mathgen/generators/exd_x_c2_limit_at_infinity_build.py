"""Exam-tier source for ``limit-at-infinity``: limits involving square roots, where dividing through
by the highest power of ``x`` is not enough.

Three families, each with an exact rational answer:

- a difference of two square roots with the same leading term, ``sqrt(a^2 x^2 + b x + c) - sqrt(a^2 x^2 + e x + f)``
  as ``x -> infinity`` (multiply by the conjugate; the limit is ``(b - e) / (2a)``);
- a square root minus a linear function, ``sqrt(a^2 x^2 + b x + c) - (a x + h)`` (the limit is ``b / (2a) - h``);
- a linear numerator over a square root as ``x -> -infinity``, ``(p x + q) / sqrt(a^2 x^2 + b x + c)``, where
  ``sqrt(x^2) = |x| = -x`` turns the limit into ``-p / a`` rather than ``p / a``.

Built backwards from the coefficients, so the key is exact by construction.  As a second route the expression
is evaluated at ``x = +/- 10^8`` with 50-digit arithmetic and must lie within ``10^-6`` of the key.

Distractors:

- difference of roots: answering 0 because "the two roots are about equal"; forgetting the factor 2 in the
  conjugate denominator; inverting the ratio;
- root minus linear: answering 0, forgetting the 2, or applying the shift ``h`` with the wrong sign;
- negative infinity: ignoring the sign change (``p / a``), or inverting the ratio, with or without the sign.
"""

from __future__ import annotations

import random

from mpmath import mp, mpf, sqrt as mp_sqrt
from sympy import Rational

from ..core import check, generator, join_terms, phrase, pick_distractors, rational_text

_PATTERNS = (
    "Find the limit as x → {target} of {expr}.",
    "Evaluate lim x→{target} of {expr}.",
    "Determine the limit of {expr} as x → {target}.",
)


def _radicand(a: int, b: int, c: int) -> str:
    return "√(" + join_terms([(a * a, "x^2"), (b, "x"), (c, "")]) + ")"


def _numeric_limit(fn, x_value: int) -> mpf:
    mp.dps = 50
    return fn(mpf(x_value))


def _difference_of_roots(rng):
    a = rng.randint(1, 4)
    b, e = rng.randint(-9, 9), rng.randint(-9, 9)
    while e == b or (b - e) ** 2 == 4 * a * a:  # keep the inverted-ratio distractor distinct from the key
        e = rng.randint(-9, 9)
    c, f = rng.randint(-6, 6), rng.randint(-6, 6)
    expr = f"{_radicand(a, b, c)} - {_radicand(a, e, f)}"
    key = Rational(b - e, 2 * a)
    fn = lambda x: mp_sqrt(a * a * x * x + b * x + c) - mp_sqrt(a * a * x * x + e * x + f)  # noqa: E731
    wrong = [Rational(0), Rational(b - e, a), Rational(2 * a, b - e)]
    return expr, "∞", key, fn, 10**8, wrong


def _root_minus_linear(rng):
    a = rng.randint(1, 4)
    b = rng.choice([v for v in range(-9, 10) if v != 0])
    c = rng.randint(-6, 6)
    h = rng.choice(
        [v for v in range(-4, 5) if v != 0 and v not in (Rational(b, 2 * a), Rational(b, a), -Rational(b, 2 * a))]
    )  # no zero or duplicate options
    linear = join_terms([(a, "x"), (h, "")])
    expr = f"{_radicand(a, b, c)} - ({linear})"
    key = Rational(b, 2 * a) - h
    fn = lambda x: mp_sqrt(a * a * x * x + b * x + c) - (a * x + h)  # noqa: E731
    wrong = [Rational(0), Rational(b, a) - h, Rational(b, 2 * a) + h]
    return expr, "∞", key, fn, 10**8, wrong


def _towards_minus_infinity(rng):
    a = rng.randint(1, 4)
    p = rng.choice([v for v in range(-6, 7) if v != 0 and abs(v) != a])
    q = rng.randint(-6, 6)
    b = rng.randint(-9, 9)
    c = rng.randint(-6, 6)
    numerator = join_terms([(p, "x"), (q, "")])
    expr = f"({numerator})/{_radicand(a, b, c)}"
    key = Rational(-p, a)
    fn = lambda x: (p * x + q) / mp_sqrt(a * a * x * x + b * x + c)  # noqa: E731
    wrong = [Rational(p, a), Rational(a, p), Rational(-a, p)]
    return expr, "-∞", key, fn, -(10**8), wrong


_FAMILIES = (_difference_of_roots, _root_minus_linear, _towards_minus_infinity)


def _exam(rng: random.Random) -> dict:
    expr, target, key, fn, probe, wrong = rng.choice(_FAMILIES)(rng)
    approx = _numeric_limit(fn, probe)
    check(abs(approx - mpf(key.p) / mpf(key.q)) < mpf("1e-6"), "numeric evaluation disagrees with the exact limit")
    text = phrase(rng, _PATTERNS, expr=expr, target=target)
    answer_text = rational_text(key)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": int(key.p), "q": int(key.q)},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, [rational_text(w) for w in wrong]),
    )


@generator("exd_x_c2_limit_at_infinity", level=16, presets={"exam": dict(tier="exam")})
def exd_x_c2_limit_at_infinity(rng: random.Random, tier: str = "exam") -> dict:
    """Limits at +/- infinity of expressions with square roots."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
