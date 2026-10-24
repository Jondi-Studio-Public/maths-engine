"""Exam chain on an exponential density, worked with numerical integration: a cumulative probability, the
probability of an interval, the tail beyond it, and a conditional probability.

For f(t) = lambda e^(-lambda t), t >= 0, the stem names lambda, a bound b and a width s. Four parts:

(a) P(T <= b), by numerical integration over [0, b], to 4 decimal places;
(b) P(b < T <= b + s), by numerical integration over [b, b + s], to 4 decimal places;
(c) hence P(T > b + s) = 1 - (a) - (b), to 4 decimal places;
(d) hence P(T <= b + s | T > b) = (b) / (1 - (a)), to 3 decimal places.

Every key is computed from the exact exponentials and rounded once. The draw is kept only when working through
the chain on the rounded earlier answers lands on the same displayed answer, so a learner who rounds as asked
and carries on agrees with the key.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import check, draw_until, generator, phrase

from .exd_x_r2j_support import clear_of_boundary, number_part, shown, text_of

_STEM = [
    "A continuous random variable T has probability density function f(t) = {lam}e^(-{lam}t) for t ≥ 0, and "
    "f(t) = 0 otherwise.",
    "Let f(t) = {lam}e^(-{lam}t), t ≥ 0, be the probability density function of an exponential random variable "
    "T with parameter λ = {lam}.",
    "The random variable T has the exponential probability density function f(t) = {lam}e^(-{lam}t) for t ≥ 0.",
]


def _exp(x: Fraction) -> Fraction:
    return Fraction(math.exp(float(x)))


def _draw(rng: random.Random):
    lam = Fraction(rng.choice([k for k in range(3, 31) if k != 10]), 10)
    b = Fraction(rng.randint(5, 40), 10)
    s = Fraction(rng.randint(2, 20), 10)
    return lam, b, s


def _values(lam: Fraction, b: Fraction, s: Fraction):
    ea, eb = _exp(-lam * b), _exp(-lam * (b + s))
    a = 1 - ea
    mid = ea - eb
    tail = eb
    cond = mid / ea
    return a, mid, tail, cond


def _accept(draw) -> bool:
    lam, b, s = draw
    a, mid, tail, cond = _values(lam, b, s)
    if not (Fraction(15, 100) < a < Fraction(9, 10) and tail > Fraction(3, 100) and mid > Fraction(3, 100)):
        return False
    if not all(clear_of_boundary(v, p) for v, p in ((a, 4), (mid, 4), (tail, 4), (cond, 3))):
        return False
    a4, m4 = shown(a, 4), shown(mid, 4)
    # carrying the rounded answers forward must land on the same displayed answers
    return shown(1 - a4 - m4, 4) == shown(tail, 4) and shown(m4 / (1 - a4), 3) == shown(cond, 3)


def _fmt(x: Fraction) -> str:
    return f"{float(x):g}"


@generator("exd_x_r2j_exponential_pdf_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_r2j_exponential_pdf_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Cumulative, interval, tail and conditional probabilities of an exponential density by numerical integration."""
    check(tier == "exam", "exd_x_r2j_exponential_pdf_chain: only the exam tier exists")
    lam, b, s = draw_until(rng, _draw, _accept, what="an exponential chain whose rounded route is stable")
    a, mid, tail, cond = _values(lam, b, s)
    hi = b + s
    stem = phrase(rng, _STEM, lam=_fmt(lam))
    lam_s, b_s, hi_s = _fmt(lam), _fmt(b), _fmt(hi)

    # independent re-derivation of the three integrals by Simpson's rule
    def simpson(lo: float, up: float, n: int = 2000) -> float:
        h = (up - lo) / n
        f = lambda t: float(lam) * math.exp(-float(lam) * t)  # noqa: E731
        return h / 3 * (f(lo) + f(up) + sum((4 if i % 2 else 2) * f(lo + i * h) for i in range(1, n)))

    check(abs(simpson(0, float(b)) - float(a)) < 1e-9, "P(T <= b) disagrees with quadrature")
    check(abs(simpson(float(b), float(hi)) - float(mid)) < 1e-9, "interval probability disagrees with quadrature")
    check(
        abs(simpson(float(hi), float(hi) + 40 / float(lam), 4000) - float(tail)) < 1e-6,
        "tail disagrees with quadrature",
    )

    ea = 1 - a

    def probabilities(pairs):
        """Keep only candidates that are possible probabilities, which a learner would not offer otherwise."""
        return [(v, tag) for v, tag in pairs if 0 < v < 1]

    a_part = number_part(
        prompt=f"Use numerical integration to find P(T ≤ {b_s}), correct to 4 decimal places.",
        exact=a,
        places=4,
        marks=2,
        working=[f"P(T ≤ {b_s}) = ∫ from 0 to {b_s} of f(t) dt = {text_of(a, 4)} (4 d.p.)."],
        wrong=probabilities(
            [
                (ea, "cdf-vs-survival-confused"),
                ((1 - ea) / lam, "forgot-density-factor"),
                (_exp(lam * b) - 1, "positive-exponent-in-density"),
                (a * lam, "forgot-density-factor"),
            ]
        ),
        sensible_range=(0, 1),
    )
    b_part = number_part(
        prompt=f"Use numerical integration to find P({b_s} < T ≤ {hi_s}), correct to 4 decimal places.",
        exact=mid,
        places=4,
        marks=2,
        working=[f"P({b_s} < T ≤ {hi_s}) = ∫ from {b_s} to {hi_s} of f(t) dt = {text_of(mid, 4)} (4 d.p.)."],
        wrong=probabilities(
            [
                (1 - tail, "evaluated-one-limit-only"),
                (1 - _exp(-lam * s), "wrong-limits-lower-bound-zero"),
                (ea, "evaluated-one-limit-only"),
                (tail, "evaluated-one-limit-only"),
            ]
        ),
        sensible_range=(0, 1),
    )
    c_part = number_part(
        prompt=f"Hence find P(T > {hi_s}), correct to 4 decimal places.",
        exact=tail,
        places=4,
        marks=1,
        working=["P(T > b + s) = 1 - P(T ≤ b) - P(b < T ≤ b + s).", f"= 1 - {text_of(a, 4)} - {text_of(mid, 4)}."],
        wrong=[
            (a + mid, "cdf-vs-survival-confused"),
            (1 - mid, "omitted-first-interval"),
            (1 - a, "omitted-first-interval"),
        ],
        follow_through="1 - part_a - part_b",
        sensible_range=(0, 1),
    )
    d_part = number_part(
        prompt=f"Given that T > {b_s}, find P(T ≤ {hi_s} | T > {b_s}), correct to 3 decimal places.",
        exact=cond,
        places=3,
        marks=2,
        working=[
            f"P(T ≤ {hi_s} | T > {b_s}) = P({b_s} < T ≤ {hi_s}) / P(T > {b_s}) = {text_of(mid, 4)} / (1 - {text_of(a, 4)}).",
            f"= {text_of(cond, 3)} (3 d.p.).",
        ],
        wrong=[
            (a + mid, "conditional-ignored"),
            (mid / a, "conditioned-on-the-wrong-event"),
            (mid, "conditional-ignored"),
            (1 - cond, "cdf-vs-survival-confused"),
        ],
        follow_through="part_b/(1 - part_a)",
        sensible_range=(0, 1),
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
