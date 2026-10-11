"""Exam-tier source for ``limit-removable``: a 0/0 limit that factoring alone cannot resolve.

Direct substitution gives 0/0 in every question, but cancelling a common linear factor is not the
route.  Four families, each needing a different idea:

- a square root in the numerator, ``(sqrt(p x + q) - r) / (x - c)`` or its negative: multiply by the
  conjugate, limit ``p / (2r)``;
- a ratio of sines, ``sin(a x) / sin(b x)`` as ``x -> 0``: limit ``a / b``;
- ``(1 - cos(k x)) / (m x sin(n x))`` as ``x -> 0``, which composes two standard limits: ``k^2 / (2 m n)``;
- a parametric numerator, ``(x^2 + p x + k) / (x - a)`` or ``(x^3 + p x + k) / (x - a)``, where the constant ``k`` is
  hidden and the learner is told only that the limit exists and is finite.  The numerator must vanish at
  ``x = a``, which fixes ``k``, and the limit is then the derivative of the numerator at ``a``.

Built backwards from the parameters; the exact value is the key.  The second route evaluates the quotient at
``c +/- 10^-12`` with 60-digit arithmetic and requires both sides to be within ``10^-8`` of the key.

Distractors:

- square-root family: 0, the value with the inner coefficient ``p`` forgotten, the reciprocal;
- sine and cosine families: the reciprocal ratio, the product of coefficients, and (cosine family) the value
  without the factor of two from ``1 - cos t ~ t^2 / 2``;
- parametric family: ``2a`` or ``3a^2`` alone (the ``p`` forgotten), ``p`` alone, and "undefined".
"""

from __future__ import annotations

import random

from mpmath import cos as mp_cos, mp, mpf, sin as mp_sin, sqrt as mp_sqrt
from sympy import Rational

from ..core import check, fmt_signed, generator, join_terms, phrase, pick_distractors, rational_text

_PATTERNS = (
    "Find the limit as x → {point} of {expr}.",
    "Evaluate lim x→{point} of {expr}.",
    "Determine the limit of {expr} as x → {point}.",
)

_PARAMETRIC_PATTERNS = (
    "The limit as x → {point} of {expr} exists and is finite, where k is a constant. Find the value of the limit.",
    "For some constant k, lim x→{point} of {expr} exists and is finite. Find its value.",
)


def _root_family(rng):
    p = rng.randint(1, 4)
    r = rng.randint(2, 5)
    c = rng.choice([v for v in range(-4, 7) if v != 0])
    q = r * r - p * c
    flipped = rng.random() < 0.5
    root = "√(" + join_terms([(p, "x"), (q, "")]) + ")"
    denominator = join_terms([(1, "x"), (-c, "")])
    numerator = f"{r} - {root}" if flipped else f"{root} - {r}"
    key = Rational(p, 2 * r) * (-1 if flipped else 1)
    sign = -1 if flipped else 1

    def fn(x):
        return (sign * (mp_sqrt(p * x + q) - r)) / (x - c)

    wrong = [Rational(0), sign * Rational(1, 2 * r), -key, sign * Rational(2 * r, p)]
    return f"({numerator}) / ({denominator})", c, key, fn, wrong, _PATTERNS


def _sine_ratio_family(rng):
    a, b = rng.sample(range(1, 8), 2)
    key = Rational(a, b)

    def fn(x):
        return mp_sin(a * x) / mp_sin(b * x)

    wrong = [Rational(b, a), Rational(a * b), Rational(a - b), Rational(a * a, b * b)]
    return f"sin({a}x) / sin({b}x)", 0, key, fn, wrong, _PATTERNS


def _cosine_family(rng):
    k = rng.randint(2, 6)
    m = rng.randint(1, 4)
    n = rng.randint(1, 4)
    key = Rational(k * k, 2 * m * n)

    def fn(x):
        return (1 - mp_cos(k * x)) / (m * x * mp_sin(n * x))

    wrong = [Rational(k * k, m * n), Rational(k, 2 * m * n), Rational(2 * m * n, k * k)]
    denominator = fmt_signed(m, "x") + f" sin({n}x)"
    return f"(1 - cos({k}x)) / ({denominator})", 0, key, fn, wrong, _PATTERNS


def _parametric_family(rng):
    a = rng.choice([v for v in range(-5, 6) if v != 0])
    p = rng.choice([v for v in range(-6, 7) if v != 0])
    cubic = rng.random() < 0.5
    if cubic:
        k = -(a**3 + p * a)
        key = Rational(3 * a * a + p)
        numerator = join_terms([(1, "x^3"), (p, "x")]) + " + k"
        wrong = [Rational(3 * a * a), Rational(p), Rational(a**3 + p * a), Rational(6 * a + p)]
    else:
        k = -(a * a + p * a)
        key = Rational(2 * a + p)
        numerator = join_terms([(1, "x^2"), (p, "x")]) + " + k"
        wrong = [Rational(2 * a), Rational(p), Rational(a * a + p * a), Rational(a + p)]

    def fn(x):
        top = x**3 + p * x + k if cubic else x**2 + p * x + k
        return top / (x - a)

    denominator = join_terms([(1, "x"), (-a, "")])
    return f"({numerator}) / ({denominator})", a, key, fn, wrong, _PARAMETRIC_PATTERNS


_FAMILIES = (_root_family, _sine_ratio_family, _cosine_family, _parametric_family)


def _exam(rng: random.Random) -> dict:
    expr, point, key, fn, wrong, patterns = rng.choice(_FAMILIES)(rng)
    with mp.workdps(60):  # not `mp.dps = 60`: mpmath's precision is global and would leak to every later caller
        target = mpf(key.p) / mpf(key.q)
        for side in (mpf(10) ** -12, -(mpf(10) ** -12)):
            near = abs(fn(mpf(point) + side) - target) < mpf("1e-8")
            check(near, "numeric evaluation disagrees with the exact limit")
    text = phrase(rng, patterns, expr=expr, point=point)
    answer_text = rational_text(key)
    answer = int(key) if key.q == 1 else {"p": int(key.p), "q": int(key.q)}
    candidates = [rational_text(w) for w in wrong] + ["undefined"]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator("exd_x_c2_limit_removable", level=16, presets={"exam": dict(tier="exam")})
def exd_x_c2_limit_removable(rng: random.Random, tier: str = "exam") -> dict:
    """0/0 limits resolved by a conjugate, standard limits, or a hidden parameter."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
