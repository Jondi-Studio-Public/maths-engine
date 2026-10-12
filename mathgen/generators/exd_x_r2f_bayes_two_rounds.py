"""Exam source for "the conjugate normal-normal posterior": a four-part chain over two rounds of data.

An unknown mean mu has the prior N(mu0, tau0^2). Measurements have known variance sigma^2. Round one gives n1 values
with mean xbar1 and round two gives n2 values with mean xbar2. With precision = 1 / variance:

(a) the posterior mean after round one      (mu0 / tau0^2 + n1 xbar1 / sigma^2) / (1/tau0^2 + n1/sigma^2)          2 marks
(b) the posterior variance after round one  1 / (1/tau0^2 + n1/sigma^2)                                             1 mark
(c) the posterior mean after round two      the round-one posterior is the prior for round two,
                                            (a/b + n2 xbar2/sigma^2) / (1/b + n2/sigma^2)                            3 marks, from (a), (b)
(d) the predictive variance of one new measurement   1 / (1/b + n2/sigma^2) + sigma^2                                  2 marks, from (b)

Every value is an exact fraction. (c) and (d) follow through from (b) (and (a)), so a slip in round one is carried
through round two. The prior and noise variances are small whole numbers so no denominator grows beyond a few digits.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational

from ..core import attempts, check, generator, rational_text
from .exd_x_r2f_common import frac_part

_VARIANCES = (1, 2, 3, 4, 5, 6)
_SETTINGS = (
    "The unknown mean error μ of a pressure sensor, in millivolts, has prior N({mu0}, {tau0}).",
    "The unknown mean daily change μ in a share price, in cents, has prior N({mu0}, {tau0}).",
    "The unknown mean offset μ of a clock, in seconds per day, has prior N({mu0}, {tau0}).",
)


def _update(mu, var, n, xbar, noise):
    """The conjugate update of N(mu, var) by n values of mean xbar and variance `noise`."""
    precision = Fraction(1) / var + Fraction(n) / noise
    return (mu / var + Fraction(n) * xbar / noise) / precision, 1 / precision


@generator(
    "exd_x_r2f_bayes_two_rounds",
    level=19,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_bayes_two_rounds(rng: random.Random, tier: str = "exam") -> dict:
    """Round-one posterior (mean, variance), the round-two posterior mean, and the predictive variance."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw with distinct, non-zero answers"):
        mu0, x1, x2 = rng.randint(-9, 9), rng.randint(-9, 9), rng.randint(-9, 9)
        tau0, noise = rng.choice(_VARIANCES), rng.choice(_VARIANCES)
        n1, n2 = rng.randint(2, 6), rng.randint(2, 6)
        mu1, var1 = _update(Fraction(mu0), Fraction(tau0), n1, Fraction(x1), Fraction(noise))
        mu2, var2 = _update(mu1, var1, n2, Fraction(x2), Fraction(noise))
        predictive = var2 + noise
        values = (mu1, var1, mu2, predictive)
        if (
            tau0 != noise
            and len(set(values)) == 4
            and 0 not in values
            and len({x1, x2, mu0}) == 3
            and max(v.denominator for v in values) <= 200
            and max(abs(mu1), abs(mu2)) < 20
        ):
            break

    # Independent route: pool everything in one go. With precisions added, the posterior after both rounds is
    # N((mu0/tau0^2 + (n1 xbar1 + n2 xbar2)/sigma^2) / P, 1/P) with P = 1/tau0^2 + (n1 + n2)/sigma^2.
    total = Fraction(1, tau0) + Fraction(n1 + n2, noise)
    pooled_mean = (Fraction(mu0, tau0) + Fraction(n1 * x1 + n2 * x2, noise)) / total
    check(pooled_mean == mu2 and 1 / total == var2, "the two-round update disagrees with the pooled update")
    check(var1 < tau0 and var2 < var1, "the variance did not shrink")

    stem = (
        f"{rng.choice(_SETTINGS).format(mu0=mu0, tau0=tau0)} Measurements have known variance σ² = {noise}. "
        f"Round one gives n₁ = {n1} measurements with sample mean {x1}; "
        f"round two gives n₂ = {n2} measurements with sample mean {x2}."
    )
    prior_prec, data1, data2 = Fraction(1, tau0), Fraction(n1, noise), Fraction(n2, noise)

    a_part = frac_part(
        "Find the posterior mean of μ after round one.",
        mu1,
        2,
        [
            f"Precision = 1/{tau0} + {n1}/{noise} = {rational_text(prior_prec + data1)}.",
            f"Mean = ({mu0}/{tau0} + {n1}×{x1}/{noise}) ÷ precision = {rational_text(mu1)}.",
        ],
        [
            ("unweighted-average", Rational(mu0 + x1, 2)),
            (
                "posterior-ignores-sample-size",
                (Fraction(mu0, tau0) + Fraction(x1, noise)) / (Fraction(1, tau0) + Fraction(1, noise)),
            ),
            ("precision-weights-swapped", (Fraction(mu0 * n1, noise) + Fraction(x1, tau0)) / (prior_prec + data1)),
            ("ignores-the-prior", Rational(x1)),
        ],
    )
    b_part = frac_part(
        "Find the posterior variance of μ after round one.",
        var1,
        1,
        [f"Variance = 1 ÷ precision = {rational_text(var1)}."],
        [
            ("doesnt-shrink", Rational(tau0)),
            ("reports-precision-not-variance", Rational(prior_prec + data1)),
            ("posterior-ignores-sample-size", 1 / (Fraction(1, tau0) + Fraction(1, noise))),
            ("ignores-the-prior", Rational(noise, n1)),
        ],
        sensible_range=(0, None),
    )
    c_part = frac_part(
        "Round one's posterior is now the prior for round two. Find the posterior mean of μ after round two.",
        mu2,
        3,
        [
            f"Prior precision = 1 ÷ {rational_text(var1)} = {rational_text(1 / var1)}; data precision = {n2}/{noise}.",
            f"Mean = ({rational_text(mu1)}×{rational_text(1 / var1)} + {n2}×{x2}/{noise}) ÷ ({rational_text(1 / var1 + data2)}) = {rational_text(mu2)}.",
        ],
        [
            ("discards-round-one", (Fraction(mu0, tau0) + Fraction(n2 * x2, noise)) / (prior_prec + data2)),
            ("unweighted-average", (mu1 + x2) / 2),
            ("posterior-ignores-sample-size", (mu1 / var1 + Fraction(x2, noise)) / (1 / var1 + Fraction(1, noise))),
            ("ignores-the-prior", Rational(n1 * x1 + n2 * x2, n1 + n2)),
        ],
        follow_through=f"(part_a/part_b + {n2}*({x2})/{noise})/(1/part_b + {n2}/{noise})",
    )
    d_part = frac_part(
        "Find the variance of the predictive distribution for one new measurement, after both rounds.",
        predictive,
        2,
        [
            f"Posterior variance after round two = 1 ÷ ({rational_text(1 / var1)} + {n2}/{noise}) = {rational_text(var2)}.",
            f"A new measurement adds the noise variance: {rational_text(var2)} + {noise} = {rational_text(predictive)}.",
        ],
        [
            ("forgets-observation-noise", Rational(var2)),
            ("discards-round-two", Rational(var1 + noise)),
            ("uses-variance-of-the-mean", Rational(var2 + Fraction(noise, n2))),
            ("reports-precision-not-variance", Rational(1 / var2 + noise)),
        ],
        follow_through=f"1/(1/part_b + {n2}/{noise}) + {noise}",
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
