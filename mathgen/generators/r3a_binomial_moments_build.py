"""Mean, variance and standard deviation of a binomial random variable: `r3a_binomial_moments` and its exam chain.

Skill `binomial-mean-variance-sd`. X counts successes in n independent trials with the same probability p.

Tiers (topic `r3a_binomial_moments`), each with a different piece of reasoning:

* easy   -- the mean, the variance or the standard deviation of X from n and p (n 10 to 30);
* medium -- a quantity built on those: an end of the interval two standard deviations either side of the mean; the
  mean or standard deviation of the count of FAILURES; or E(X^2) = Var(X) + (E X)^2;
* hard   -- two independent binomial counts combined: the standard deviation of their total when the success
  probabilities differ, of their difference, or of the total when the probability is shared. Variances add, never
  standard deviations.

The exam chain (topic `exd_x_r3a_binomial_moments`) is (a) the mean, (b) the variance from the mean, (c) the
standard deviation from the variance, (d) the probability that X lies within one standard deviation of its mean.

Every key is held as an exact Fraction (the standard deviation to about 15 digits) and re-derived from the
distribution itself: the mean and variance are summed over the probability mass function, which uses no formula
for the moments. A draw is rejected when a rounded answer sits close to a rounding boundary.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3a_kit import (
    BINOMIAL_CONTEXTS,
    build_until,
    clear_of_boundary,
    dec,
    number_part,
    pmf,
    single,
    span,
    sqrt_fraction,
)


def _moments_by_sum(n: int, p: Fraction) -> tuple[Fraction, Fraction]:
    """Mean and variance of Bin(n, p) summed over the mass function, with no moment formula."""
    mean = sum((k * pmf(n, p, k) for k in range(n + 1)), Fraction(0))
    second = sum((k * k * pmf(n, p, k) for k in range(n + 1)), Fraction(0))
    return mean, second - mean * mean


def _draw(rng: random.Random):
    n = rng.randint(10, 30)
    setting, x_word, y_word, noun, ps = rng.choice(BINOMIAL_CONTEXTS)
    p = rng.choice(ps)
    return n, p, setting.format(n=n, p=dec(p)), x_word, y_word, noun


def _verified(n: int, p: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    mean, var = n * p, n * p * (1 - p)
    by_sum = _moments_by_sum(n, p)
    check(by_sum == (mean, var), "the formulas disagree with the sum over the distribution")
    return mean, var, sqrt_fraction(var)


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    for _ in attempts(500, "a variance and standard deviation clear of a rounding boundary"):
        n, p, setting, x_word, _y, _noun = _draw(rng)
        mean, var, sd = _verified(n, p)
        q = 1 - p
        if clear_of_boundary(var, 2) and clear_of_boundary(sd, 2):
            break
    ask = rng.choice(("mean", "variance", "sd"))
    stem_x = f"Let X be {x_word}."
    if ask == "mean":
        part = number_part(
            "Find the mean of X, correct to 2 decimal places.",
            mean,
            2,
            1,
            [f"E(X) = np = {n} x {dec(p)} = {mean}."],
            [
                (n * q, "r3a-mean-uses-failure-probability"),
                (var, "r3a-mean-reported-as-variance"),
                (n + p, "r3a-mean-adds-n-and-p"),
            ],
        )
    elif ask == "variance":
        part = number_part(
            "Find the variance of X, correct to 2 decimal places.",
            var,
            2,
            1,
            [f"Var(X) = npq = {n} x {dec(p)} x {dec(q)}."],
            [
                (mean, "r3a-variance-omits-failure-probability"),
                (sd, "r3a-variance-reported-as-sd"),
                (n * p * p, "r3a-variance-squares-p"),
            ],
        )
    else:
        part = number_part(
            "Find the standard deviation of X, correct to 2 decimal places.",
            sd,
            2,
            1,
            [f"SD(X) = sqrt(npq) = sqrt({n} x {dec(p)} x {dec(q)})."],
            [
                (var, "r3a-sd-omits-square-root"),
                (sqrt_fraction(mean), "r3a-sd-omits-failure-probability"),
                (mean, "r3a-sd-reported-as-mean"),
            ],
        )
    return single(f"{setting} {stem_x} {part['prompt']}", part)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium(rng: random.Random) -> dict:
    variant = rng.choice(("interval", "failures", "second_moment"))
    for _ in attempts(500, "a draw clear of every rounding boundary"):
        n, p, setting, x_word, y_word, _noun = _draw(rng)
        mean, var, sd = _verified(n, p)
        q = 1 - p
        end = rng.choice(("upper", "lower"))
        second = var + mean * mean
        if not (clear_of_boundary(var, 2) and clear_of_boundary(sd, 2) and clear_of_boundary(second, 2)):
            continue
        if variant == "interval" and end == "lower" and mean - 2 * sd <= 0:
            continue
        break
    if variant == "interval":
        value = mean + 2 * sd if end == "upper" else mean - 2 * sd
        wrong_end = mean - 2 * sd if end == "upper" else mean + 2 * sd
        sign = "+" if end == "upper" else "-"
        part = number_part(
            f"Find the {end} end of the interval that runs two standard deviations either side of the mean of X, "
            "correct to 2 decimal places.",
            value,
            2,
            1,
            [f"mean = {mean}, SD = {float(sd):.4f}, so {end} end = {mean} {sign} 2 x {float(sd):.4f}."],
            [
                (wrong_end, "r3a-interval-wrong-end"),
                (mean + (2 * var if end == "upper" else -2 * var), "r3a-interval-uses-variance-for-sd"),
                (mean + (sd if end == "upper" else -sd), "r3a-interval-one-sd-for-two"),
            ],
        )
        stem = f"{setting} Let X be {x_word}."
    elif variant == "failures":
        which = rng.choice(("mean", "sd"))
        stem = f"{setting} Let Y be {y_word}."
        if which == "mean":
            part = number_part(
                "Find the mean of Y, correct to 2 decimal places.",
                n * q,
                2,
                1,
                [f"Y counts failures, which have probability {dec(q)}: E(Y) = {n} x {dec(q)}."],
                [
                    (mean, "r3a-failures-mean-uses-success-probability"),
                    (var, "r3a-mean-reported-as-variance"),
                    (n * q * q, "r3a-mean-squares-probability"),
                ],
            )
        else:
            part = number_part(
                "Find the standard deviation of Y, correct to 2 decimal places.",
                sd,
                2,
                1,
                [f"Y = {n} - X, so Var(Y) = Var(X) = npq = {var} and SD(Y) = sqrt({var})."],
                [
                    (sqrt_fraction(n * q), "r3a-sd-omits-failure-probability"),
                    (var, "r3a-sd-omits-square-root"),
                    (n - sd, "r3a-sd-of-complement-subtracted-from-n"),
                ],
            )
    else:
        stem = f"{setting} Let X be {x_word}."
        part = number_part(
            "Find E(X^2), correct to 2 decimal places.",
            second,
            2,
            1,
            [f"E(X^2) = Var(X) + (E(X))^2 = {var} + {mean}^2."],
            [
                (mean * mean, "r3a-second-moment-squares-the-mean"),
                (var, "r3a-second-moment-is-the-variance"),
                (mean * mean + mean, "r3a-second-moment-adds-the-mean"),
            ],
        )
    return single(f"{stem} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------

#: (different-probability setting, shared-probability setting, the total T, the difference D, plausible p values)
_PAIRS = (
    (
        "Machine A makes {n1} components and Machine B makes {n2}. Each component from A is independently faulty "
        "with probability {p1}, and each from B with probability {p2}.",
        "Machine A makes {n1} components and Machine B makes {n2}. Each component, whichever machine made it, is "
        "independently faulty with probability {p1}.",
        "the total number of faulty components",
        "the number of faulty components from A minus the number from B",
        tuple(Fraction(k, 20) for k in range(1, 7)),
    ),
    (
        "Clinic A sees {n1} patients in a day and Clinic B sees {n2}. Each patient at A is independently late with "
        "probability {p1}, and each at B with probability {p2}.",
        "Clinic A sees {n1} patients in a day and Clinic B sees {n2}. Each patient, at either clinic, is "
        "independently late with probability {p1}.",
        "the total number of late patients",
        "the number of late patients at A minus the number at B",
        tuple(Fraction(k, 20) for k in range(2, 9)),
    ),
    (
        "Orchard A sends {n1} apples for inspection and Orchard B sends {n2}. Each apple from A is independently "
        "bruised with probability {p1}, and each from B with probability {p2}.",
        "Orchard A sends {n1} apples for inspection and Orchard B sends {n2}. Each apple, from either orchard, is "
        "independently bruised with probability {p1}.",
        "the total number of bruised apples",
        "the number of bruised apples from A minus the number from B",
        tuple(Fraction(k, 20) for k in range(1, 7)),
    ),
    (
        "Class A has {n1} students sit a test and Class B has {n2}. Each student in A independently passes with "
        "probability {p1}, and each in B with probability {p2}.",
        "Class A has {n1} students sit a test and Class B has {n2}. Each student, in either class, independently "
        "passes with probability {p1}.",
        "the total number of students who pass",
        "the number of students who pass in A minus the number who pass in B",
        tuple(Fraction(k, 20) for k in range(12, 19)),
    ),
)


def _hard(rng: random.Random) -> dict:
    variant = rng.choice(("total", "difference", "shared"))
    different, same, total_word, diff_word, ps = rng.choice(_PAIRS)
    for _ in attempts(1000, "two counts with clear standard deviations"):
        n1, n2 = rng.randint(8, 24), rng.randint(8, 24)
        p1 = rng.choice(ps)
        p2 = p1 if variant == "shared" else rng.choice(ps)
        if variant != "shared" and p1 == p2:
            continue
        v1, v2 = n1 * p1 * (1 - p1), n2 * p2 * (1 - p2)
        s1, s2 = sqrt_fraction(v1), sqrt_fraction(v2)
        total_var = v1 + v2
        total_sd = sqrt_fraction(total_var)
        if not (clear_of_boundary(total_sd, 2) and clear_of_boundary(total_var, 2)):
            continue
        if variant == "difference" and (v1 - v2 <= 0 or abs(v1 - v2) < Fraction(1, 2)):
            continue
        break
    check(
        math.isclose(float(total_sd) ** 2, float(v1 + v2), rel_tol=1e-9),
        "standard deviation does not square to the summed variance",
    )
    if variant == "shared":
        stem = same.format(n1=n1, n2=n2, p1=dec(p1)) + f" All outcomes are independent. Let T be {total_word}."
        pooled = sqrt_fraction((n1 + n2) * p1 * (1 - p1))
        check(pooled == total_sd, "pooled binomial disagrees with the sum of the variances")
        part = number_part(
            "Find the standard deviation of T, correct to 2 decimal places.",
            total_sd,
            2,
            1,
            [f"T ~ Bin({n1 + n2}, {dec(p1)}), so SD(T) = sqrt({n1 + n2} x {dec(p1)} x {dec(1 - p1)})."],
            [
                (s1 + s2, "r3a-adds-standard-deviations"),
                (total_var, "r3a-sd-omits-square-root"),
                (sqrt_fraction((n1 + n2) * p1), "r3a-sd-omits-failure-probability"),
            ],
        )
    else:
        stem = different.format(n1=n1, n2=n2, p1=dec(p1), p2=dec(p2)) + " All outcomes are independent."
        if variant == "total":
            stem += f" Let T be {total_word}."
            part = number_part(
                "Find the standard deviation of T, correct to 2 decimal places.",
                total_sd,
                2,
                1,
                [f"Var(T) = {v1} + {v2} = {total_var}, so SD(T) = sqrt({total_var})."],
                [
                    (s1 + s2, "r3a-adds-standard-deviations"),
                    (total_var, "r3a-sd-omits-square-root"),
                    (
                        sqrt_fraction((n1 + n2) * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2)),
                        "r3a-pools-unequal-probabilities",
                    ),
                ],
            )
        else:
            stem += f" Let D be {diff_word}."
            part = number_part(
                "Find the standard deviation of D, correct to 2 decimal places.",
                total_sd,
                2,
                1,
                [f"Var(D) = Var(A) + Var(B) = {v1} + {v2} = {total_var}, so SD(D) = sqrt({total_var})."],
                [
                    (sqrt_fraction(abs(v1 - v2)), "r3a-subtracts-variances"),
                    (abs(s1 - s2), "r3a-subtracts-standard-deviations"),
                    (total_var, "r3a-sd-omits-square-root"),
                ],
            )
    return single(f"{stem} {part['prompt']}", part)


@generator(
    "r3a_binomial_moments",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3a_binomial_moments(rng: random.Random, tier: str = "easy") -> dict:
    """Binomial moments: formula (easy), built on it (medium), independent counts combined (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    """Mean, variance, standard deviation, then the probability of falling within one standard deviation."""
    for _ in attempts(2000, "a binomial whose one-standard-deviation interval is stable"):
        template, x_word, _y, _noun, ps = rng.choice(BINOMIAL_CONTEXTS)
        n, p = rng.randint(14, 30), Fraction(rng.randint(int(min(ps) * 100), int(max(ps) * 100)), 100)
        mean, var, sd = _verified(n, p)
        lo, hi = mean - sd, mean + sd
        if not (clear_of_boundary(var, 2) and clear_of_boundary(sd, 2)):
            continue
        # the interval ends must not sit near a whole number, or a rounded SD could change the integers inside
        if any(abs(e - round(e)) < Fraction(1, 20) for e in (lo, hi)):
            continue
        a, b = math.ceil(lo), math.floor(hi)
        sd_2dp = Fraction(round(sd * 100), 100)
        if (math.ceil(mean - sd_2dp), math.floor(mean + sd_2dp)) != (a, b):
            continue
        inside = span(n, p, a, b)
        if not (clear_of_boundary(inside, 4, Fraction(1, 5)) and Fraction(1, 2) < inside < 1):
            continue
        break
    # the independent route: add up the mass function over the whole numbers in the interval
    check(inside == sum(pmf(n, p, k) for k in range(n + 1) if lo <= k <= hi), "interval probability disagrees")
    q = 1 - p
    pq = f"{p.numerator}/{p.denominator}"
    parts = [
        number_part(
            "Find E(X), the mean of X, correct to 2 decimal places.",
            mean,
            2,
            1,
            [f"E(X) = np = {n} x {dec(p)} = {mean}."],
            [
                (n * q, "r3a-mean-uses-failure-probability"),
                (var, "r3a-mean-reported-as-variance"),
                (n + p, "r3a-mean-adds-n-and-p"),
            ],
            sensible_range=(0, n),
        ),
        number_part(
            "Find Var(X), the variance of X, correct to 2 decimal places.",
            var,
            2,
            1,
            [f"Var(X) = npq = {mean} x {dec(q)} = {var}."],
            [
                (mean, "r3a-variance-omits-failure-probability"),
                (n * p * p, "r3a-variance-squares-p"),
                (n * q * q, "r3a-variance-squares-q"),
            ],
            follow_through=f"part_a*(1-{pq})",
            sensible_range=(0, n),
        ),
        number_part(
            "Find the standard deviation of X, correct to 2 decimal places.",
            sd,
            2,
            1,
            [f"SD(X) = sqrt(Var(X)) = sqrt({var})."],
            [
                (var, "r3a-sd-omits-square-root"),
                (sqrt_fraction(mean), "r3a-sd-omits-failure-probability"),
                (var / 2, "r3a-sd-halves-the-variance"),
            ],
            follow_through="sqrt(part_b)",
            sensible_range=(0, n),
        ),
        number_part(
            "Find the probability that X lies within one standard deviation of its mean, that is "
            "P(mean - SD <= X <= mean + SD), correct to 4 decimal places.",
            inside,
            4,
            3,
            [
                f"The interval is {float(lo):.2f} to {float(hi):.2f}, so the whole numbers are {a} to {b}.",
                f"P({a} <= X <= {b}) = {float(inside):.4f}.",
            ],
            [
                (span(n, p, 0, b), "r3a-within-sd-ignores-lower-bound"),
                (span(n, p, a - 1, b + 1), "r3a-within-sd-rounds-outwards"),
                (span(n, p, a + 1, b - 1), "r3a-within-sd-drops-end-values"),
                (span(n, p, math.ceil(mean - 2 * sd), math.floor(mean + 2 * sd)), "r3a-within-two-sd"),
            ],
            sensible_range=(0, 1),
        ),
    ]
    setting = template.format(n=n, p=dec(p))
    return dict(text=f"{setting} Let X be {x_word}.", parts=parts)


@generator("exd_x_r3a_binomial_moments", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3a_binomial_moments(rng: random.Random, tier: str = "exam") -> dict:
    """Mean, variance, standard deviation, then the probability of falling within one standard deviation."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
