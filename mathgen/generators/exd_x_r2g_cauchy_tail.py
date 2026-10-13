"""Exam question: tail spread and an epsilon-N threshold for a Cauchy sequence (`ucv1-completeness-cauchy-check`).

The sequence is a_n = (alpha*n + beta)/(gamma*n) = L + c/n, with L = alpha/gamma and c = beta/gamma. It converges,
so it is Cauchy; the exam asks for the numbers behind the definition. The parts chain:

* (a) the limit L (fraction);
* (b) the least upper bound of |a_m - a_n| over all m, n >= N0, which is c / N0 (fraction);
* (c) the smallest N such that |a_m - a_n| < epsilon for every m, n >= N: the spread is c/N, so N is the least
  integer above c/epsilon. Follow-through from (b), which supplies c = N0 * (b);
* (d) |a_N - L| for that N, which is c/N (follow-through from (b) and (c)).

Everything is exact (fractions, and an integer N). Draws where c/epsilon is a whole number are skipped: the spread
c/N is a least upper bound that is never attained, so "less than epsilon" would then be a matter of reading, not
of method.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import floor

from ..core import check, generator
from .exd_x_r2g_support import build_part, redraws


def _eps_text(eps: Fraction) -> str:
    # epsilon is always a terminating decimal here
    scaled = eps * 1000
    check(scaled.denominator == 1, "epsilon must be a terminating decimal")
    return f"{float(eps):g}"


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        alpha = rng.choice([-1, 1]) * rng.randint(2, 8)
        gamma = rng.randint(2, 6)
        beta = rng.choice([-1, 1]) * rng.randint(1, 9)
        c = Fraction(abs(beta), gamma)
        n0 = rng.randint(2, 9)
        eps = Fraction(rng.randint(1, 9), rng.choice([10, 20, 50, 100]))
        threshold = c / eps
        n = floor(threshold) + 1
        if not 4 <= n <= 90 or n <= n0 or Fraction(alpha, gamma) == 0:
            continue
        if threshold.denominator == 1:
            continue  # the spread c/N is never attained, so a whole-number threshold would make "< epsilon" subtle
        return dict(alpha=alpha, beta=beta, gamma=gamma, c=c, n0=n0, eps=eps, n=n, limit=Fraction(alpha, gamma))
    check(False, "no acceptable sequence draw")


def _num(alpha: int, beta: int) -> str:
    return f"{alpha}n {'+' if beta > 0 else '-'} {abs(beta)}"


_STEMS = [
    "The sequence a_n = ({num})/({gamma}n), n = 1, 2, 3, ..., converges, so it is Cauchy.",
    "Consider the sequence defined by a_n = ({num})/({gamma}n) for n = 1, 2, 3, .... It is convergent and "
    "therefore Cauchy.",
    "A sequence has terms a_n = ({num})/({gamma}n) for n = 1, 2, 3, ...; being convergent, it is a Cauchy sequence.",
]


@generator("exd_x_r2g_cauchy_tail", level=15, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_cauchy_tail(rng: random.Random, tier: str = "exam") -> dict:
    """Limit, tail spread and the least N for a Cauchy sequence of the form L + c/n."""
    check(tier == "exam", "exd_x_r2g_cauchy_tail: only the exam tier exists")
    d = _draw(rng)
    alpha, beta, gamma, c, n0, eps, n, limit = (
        d[k] for k in ("alpha", "beta", "gamma", "c", "n0", "eps", "n", "limit")
    )
    spread0 = c / n0
    gap = c / n
    eps_s = _eps_text(eps)
    stem = rng.choice(_STEMS).format(num=_num(alpha, beta), gamma=gamma)
    first_term = Fraction(alpha + beta, gamma)
    eps_ft = f"({eps.numerator}/{eps.denominator})"

    parts = [
        build_part(
            "Find the limit L of the sequence, as a fraction in lowest terms.",
            limit,
            kind="fraction",
            marks=1,
            working=[f"a_n = {alpha}/{gamma} + ({beta}/{gamma})/n, and the second term tends to 0.", f"L = {limit}."],
            wrong=[
                (Fraction(alpha), "limit-numerator-coefficient-only"),
                (Fraction(beta, gamma), "limit-taken-as-constant-term"),
                (first_term, "first-term-as-limit"),
                (limit - Fraction(beta, gamma), "limit-sign-error-on-constant"),
                (Fraction(gamma, alpha), "limit-inverted"),
            ],
        ),
        build_part(
            f"Find the least upper bound of |a_m - a_n| over all m, n >= {n0}.",
            spread0,
            kind="fraction",
            marks=2,
            sensible_range=(0, None),
            working=[
                f"For m, n >= {n0} the terms lie between a_{n0} and L, a distance {c}/{n0} apart at most.",
                f"The least upper bound is {spread0}.",
            ],
            wrong=[
                (c, "ignores-the-index-in-tail-bound"),
                (c / (n0 + 1), "tail-bound-from-next-index"),
                (2 * spread0, "tail-spread-doubled"),
                (c / (n0 * (n0 + 1)), "consecutive-difference-as-tail-spread"),
            ],
        ),
        build_part(
            f"Hence find the smallest integer N such that |a_m - a_n| < {eps_s} for all m, n >= N.",
            n,
            kind="number",
            marks=2,
            sensible_range=(0, None),
            follow_through=f"floor({n0}*part_b/{eps_ft}) + 1",
            working=[
                f"The spread for m, n >= N is {c}/N, so we need {c}/N < {eps_s}, that is N > {c / eps}.",
                f"The smallest such integer is N = {n}.",
            ],
            wrong=[
                (floor(c / eps), "rounds-down-instead-of-up"),
                (n + 1, "off-by-one-count"),
                (floor(1 / eps) + 1, "ignores-the-index-in-tail-bound"),
                (floor(2 * c / eps) + 1, "tail-spread-doubled"),
            ],
        ),
        build_part(
            "For this N, find |a_N - L|, where L is the limit.",
            gap,
            kind="fraction",
            marks=1,
            sensible_range=(0, None),
            follow_through=f"{n0}*part_b/part_c",
            working=[f"|a_N - L| = {c}/N = {c}/{n} = {gap}."],
            wrong=[
                (eps, "epsilon-taken-as-the-gap"),
                (c / (n - 1), "off-by-one-count"),
                (spread0, "reuses-earlier-part-value"),
                (c / (n + 1), "off-by-one-count"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
