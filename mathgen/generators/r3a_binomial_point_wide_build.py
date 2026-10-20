"""Wider exam tier for `binomial-point-probability`: n from 12 to 25, the centre of the distribution, then one point.

The older exam source of this skill asks for one event built from a handful of point probabilities when n is 5 to 9.
This one (topic `exd_x_r3a_binomial_point_wide`) works with a larger n, where the learner first locates the centre of
the distribution and only then evaluates a point probability:

(a) the mean np;
(b) the standard deviation sqrt(npq);
(c) the most likely value of X, the largest k with P(X = k) maximal, which is floor((n + 1)p) when (n + 1)p is not a
    whole number (the draw guarantees it is not, so the value is unique); it follows from (a);
(d) the point probability P(X = most likely value), to 4 decimal places.

Each most likely value is confirmed by comparing every point probability in the exact distribution, and each point
probability is re-derived from Pascal's recurrence, which uses no binomial coefficient.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3a_kit import BINOMIAL_CONTEXTS, build_until, clear_of_boundary, dec, number_part, pmf, sqrt_fraction


def _distribution_by_recurrence(n: int, p: Fraction) -> list[Fraction]:
    dist = [Fraction(1)]
    for _ in range(n):
        nxt = [Fraction(0)] * (len(dist) + 1)
        for k, mass in enumerate(dist):
            nxt[k] += mass * (1 - p)
            nxt[k + 1] += mass * p
        dist = nxt
    return dist


def _exam(rng: random.Random) -> dict:
    """Mean, standard deviation, most likely value, then the probability of that value (n 12 to 25)."""
    for _ in attempts(3000, "a binomial with a unique mode and clear rounding"):
        template, x_word, _y, _noun, ps = rng.choice(BINOMIAL_CONTEXTS)
        n, p = rng.randint(12, 25), Fraction(rng.randint(int(min(ps) * 100), int(max(ps) * 100)), 100)
        q = 1 - p
        if ((n + 1) * p).denominator == 1:
            continue
        mean, var = n * p, n * p * q
        sd = sqrt_fraction(var)
        mode = math.floor((n + 1) * p)
        dist = _distribution_by_recurrence(n, p)
        if not clear_of_boundary(sd, 2) or not clear_of_boundary(dist[mode], 4, Fraction(1, 5)):
            continue
        break
    check(dist.index(max(dist)) == mode and dist.count(max(dist)) == 1, "the most likely value is not unique")
    check(dist[mode] == pmf(n, p, mode), "recurrence and formula disagree")
    setting = template.format(n=n, p=dec(p))
    pq = f"({p.numerator}/{p.denominator})"
    parts = [
        number_part(
            "Find the mean of X, correct to 2 decimal places.",
            mean,
            2,
            1,
            [f"E(X) = np = {n} x {dec(p)} = {dec(mean)}."],
            [
                (n * q, "r3a-mean-uses-failure-probability"),
                (var, "r3a-mean-reported-as-variance"),
                (n + p, "r3a-mean-adds-n-and-p"),
            ],
            sensible_range=(0, n),
        ),
        number_part(
            "Find the standard deviation of X, correct to 2 decimal places.",
            sd,
            2,
            1,
            [f"SD(X) = sqrt(npq) = sqrt({dec(mean)} x {dec(q)})."],
            [
                (var, "r3a-sd-omits-square-root"),
                (sqrt_fraction(mean), "r3a-sd-omits-failure-probability"),
                (mean, "r3a-sd-reported-as-mean"),
            ],
            follow_through=f"sqrt(part_a*(1-{pq}))",
            sensible_range=(0, n),
        ),
        number_part(
            "Find the most likely value of X, the whole number k for which P(X = k) is greatest.",
            Fraction(mode),
            0,
            2,
            [
                f"P(X = k) rises while (n - k + 1)p > kq, that is while k < (n + 1)p = {float((n + 1) * p):.2f}.",
                f"The most likely value is k = {mode}.",
            ],
            [
                (Fraction(round(mean)), "r3a-mode-rounds-the-mean"),
                (Fraction(math.floor(mean)), "r3a-mode-floors-the-mean"),
                (Fraction(mode + 1), "r3a-mode-one-too-high"),
                (Fraction(mode - 1), "r3a-mode-one-too-low"),
                (Fraction(math.ceil(mean)), "r3a-mode-ceils-the-mean"),
            ],
            follow_through=f"floor(part_a+{pq})",
            sensible_range=(0, n),
        ),
        number_part(
            "Find the probability that X takes its most likely value, correct to 4 decimal places.",
            dist[mode],
            4,
            3,
            [f"P(X = {mode}) = C({n}, {mode}) x {dec(p)}^{mode} x {dec(q)}^{n - mode} = {float(dist[mode]):.4f}."],
            [
                (dist[round(mean)], "r3a-point-taken-at-the-rounded-mean"),
                (dist[min(mode + 1, n)], "r3a-point-taken-one-above-the-mode"),
                (dist[max(mode - 1, 0)], "r3a-point-taken-one-below-the-mode"),
                (p**mode * q ** (n - mode), "r3a-point-omits-the-coefficient"),
            ],
            sensible_range=(0, 1),
            within=(Fraction(0), Fraction(1)),
        ),
    ]
    return dict(text=f"{setting} Let X be {x_word}.", parts=parts)


@generator("exd_x_r3a_binomial_point_wide", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3a_binomial_point_wide(rng: random.Random, tier: str = "exam") -> dict:
    """Mean, standard deviation, most likely value, then the probability of that value (n 12 to 25)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
