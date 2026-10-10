"""Exam-tier source for ``one-sided-limits``: asymptotes and a parameter that makes the two sides agree.

Two families:

- a rational function with a vertical asymptote, ``(p x + q) / (x - a)^n`` with ``n`` 1, 2 or 3, approached from
  the left, from the right, or (for even ``n``) from both sides.  The answer is ``∞`` or ``-∞``; the sign comes
  from the sign of the numerator at ``a`` and the sign of ``(x - a)^n`` on the stated side.
- a piecewise function ``f(x) = k x + m`` for ``x < a``, ``f(x) = s x^2 + n - k`` for ``x >= a``: find the value of the
  constant ``k`` for which the two-sided limit at ``x = a`` exists.  The left and right limits are equated and
  the resulting equation is solved for ``k``.

Built backwards: the asymptote's numerator value and the integer ``k`` are chosen first.  For the asymptote
family the second route evaluates the quotient at ``a +/- 10^-6`` with ``Fraction`` and reads off the sign; for
the piecewise family it evaluates both pieces' one-sided limits at ``a`` and requires them to be equal.

Distractors:

- asymptote: the answer from the other side, "does not exist", 0, the opposite infinity;
- parameter: solving with ``+k`` in place of ``-k`` on the right, forgetting ``k`` on the right, and reporting
  the common value of the limit instead of ``k``.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, join_terms, phrase, pick_distractors, rational_text

_ASYMPTOTE_PATTERNS = (
    "Find lim x→{point} of {expr}. Give your answer as a number, ∞ or -∞.",
    "Evaluate the limit as x → {point} of {expr}. Give your answer as a number, ∞ or -∞.",
)

_PARAMETER_PATTERNS = (
    "f(x) = {left} for x < {a}, f(x) = {right} for x ≥ {a}. Find the value of k for which the limit of f(x) as x → {a} exists.",
    "A function is defined by f(x) = {left} when x < {a} and f(x) = {right} when x ≥ {a}. "
    "For what value of k does lim x→{a} f(x) exist?",
)


def _sign_text(value: int) -> str:
    return "∞" if value > 0 else "-∞"


def _asymptote(rng: random.Random) -> dict:
    a = rng.choice([v for v in range(-5, 6) if v != 0])
    n = rng.choice([1, 2, 3])
    p = rng.choice([v for v in range(-5, 6) if v != 0])
    value = rng.choice([v for v in range(-6, 7) if v != 0])  # numerator value at x = a
    q = value - p * a
    side = rng.choice(["+", "-", "both"]) if n % 2 == 0 else rng.choice(["+", "-"])
    base = join_terms([(1, "x"), (-a, "")])
    denominator = f"({base})" if n == 1 else f"({base})^{n}"
    numerator = join_terms([(p, "x"), (q, "")])
    expr = f"({numerator}) / {denominator}"
    point = f"{a}" if side == "both" else f"{a}^{side}"

    right_sign = 1 if value > 0 else -1
    left_sign = right_sign * (-1) ** n
    key = _sign_text(right_sign if side == "+" else left_sign if side == "-" else right_sign)
    other = _sign_text(left_sign if side == "+" else right_sign)

    # Independent route: the exact quotient just beside the asymptote.
    eps = Fraction(1, 10**6)
    probes = {"+": [a + eps], "-": [a - eps], "both": [a + eps, a - eps]}[side]
    for x_value in probes:
        quotient = (p * x_value + q) / (x_value - a) ** n
        check(abs(quotient) > 1000 and (quotient > 0) == (key == "∞"), "side or sign disagrees with the quotient")

    opposite = "-∞" if key == "∞" else "∞"
    wrong = [other, "does not exist", "0", opposite]
    text = phrase(rng, _ASYMPTOTE_PATTERNS, point=point, expr=expr)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=key,
        answer_latex=key,
        distractors=pick_distractors(key, wrong),
    )


def _parameter(rng: random.Random) -> dict:
    a = rng.choice([v for v in range(-4, 5) if v not in (-1, 0, 1)])
    k = rng.choice([v for v in range(-6, 7) if v != 0])
    m = rng.randint(-6, 6)
    s = rng.choice([1, 2, -1])
    n = k * (a + 1) + m - s * a * a  # left limit k a + m equals right limit s a^2 + n - k
    left_limit = k * a + m
    right_limit = s * a * a + n - k
    check(left_limit == right_limit, "one-sided limits do not agree at the constructed k")

    left = "kx" if m == 0 else f"kx {'+' if m > 0 else '-'} {abs(m)}"
    right = f"{join_terms([(s, 'x^2'), (n, '')])} - k"
    text = phrase(rng, _PARAMETER_PATTERNS, left=left, right=right, a=a)
    wrong = [
        Fraction(s * a * a + n - m, a - 1),  # sign of k wrong on the right
        Fraction(s * a * a + n - m, a),  # k forgotten on the right
        Fraction(left_limit),  # reported the common value instead of k
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=k,
        answer_text=str(k),
        answer_latex=str(k),
        distractors=pick_distractors(str(k), [rational_text(w) for w in wrong]),
    )


def _exam(rng: random.Random) -> dict:
    return _asymptote(rng) if rng.random() < 0.5 else _parameter(rng)


@generator("exd_x_c2_one_sided_limits", level=16, presets={"exam": dict(tier="exam")})
def exd_x_c2_one_sided_limits(rng: random.Random, tier: str = "exam") -> dict:
    """One-sided limits at an asymptote, or the parameter that makes the two sides agree."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
