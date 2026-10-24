"""Mean and spread of the sampling distribution of a mean or a proportion: `r3a_sampling_moments` and its exam chain.

Skill `sampling-distribution-moments`. These are the numeric moments only: no tail probability is asked (that is
`exd_x_r2k_xbar_pvalue`'s ground).

Tiers (topic `r3a_sampling_moments`):

* easy   -- the standard deviation (standard error) or the variance of the sample mean from the population standard
  deviation and the sample size;
* medium -- the standard deviation of a sample proportion from p and n; or how a standard error changes when the
  sample size is multiplied (or divided) by a perfect square;
* hard   -- the least sample size for a standard error not to exceed a target (a mean, or a proportion), or the
  standard deviation of the difference between two independent sample means.

The exam chain (topic `exd_x_r3a_sampling_moments`) is (a) the standard error of the mean, (b) the standard error
when the sample size is quadrupled, from (a), (c) the least sample size for a stated target standard error, from
(a), (d) the standard error at that sample size, from (a) and (c). Parts (b) to (d) carry `follow_through`.

Each key is held as an exact Fraction (standard errors to about 15 digits); a least sample size is confirmed by
comparing squared quantities in exact arithmetic at n and at n - 1.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3a_kit import build_until, clear_of_boundary, number_part, single, sqrt_fraction

#: (the measurement, its unit, a typical mean range, a typical sigma range) -- sigma is in the same unit
_CONTEXTS = (
    ("the mass of a packet of rice", "grams", (480, 520), (4, 12)),
    ("the time a customer waits on hold", "seconds", (90, 180), (20, 45)),
    ("the lifetime of a battery", "hours", (18, 30), (2, 6)),
    ("the length of a bolt", "millimetres", (40, 80), (2, 5)),
    ("the daily water use of a household", "litres", (300, 520), (40, 90)),
    ("the score on a reaction test", "milliseconds", (240, 320), (15, 40)),
)
_PROPORTIONS = (
    ("voters who support a proposal", "voters"),
    ("households with solar panels", "households"),
    ("students who walk to school", "students"),
    ("customers who return an item", "customers"),
)


def _draw_context(rng: random.Random):
    thing, unit, m_range, s_range = rng.choice(_CONTEXTS)
    mu = rng.randint(*m_range)
    sigma = Fraction(rng.randint(s_range[0] * 2, s_range[1] * 2), 2)
    return thing, unit, mu, sigma


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    thing, unit, mu, sigma = _draw_context(rng)
    ask = rng.choice(("sd", "variance"))
    for _ in attempts(500, "a standard error clear of a rounding boundary"):
        n = rng.choice((9, 16, 20, 24, 25, 30, 36, 40, 45, 50, 60, 64, 75, 80, 100))
        se = sqrt_fraction(sigma * sigma / n)
        se_var = sigma * sigma / n
        if clear_of_boundary(se, 2) and clear_of_boundary(se_var, 2):
            break
    stem = (
        f"{thing[0].upper()}{thing[1:]} has population mean {mu} {unit} and population standard deviation "
        f"{float(sigma):g} {unit}. Random samples of {n} are taken, and the sample mean is calculated for each."
    )
    if ask == "sd":
        part = number_part(
            "Find the standard deviation of the sample mean, correct to 2 decimal places.",
            se,
            2,
            1,
            [f"SD of the sample mean = sigma / sqrt(n) = {float(sigma):g} / sqrt({n})."],
            [
                (sigma / n, "r3a-se-divides-by-n"),
                (sigma * sqrt_fraction(Fraction(n)), "r3a-se-multiplies-by-root-n"),
                (sigma * sigma / sqrt_fraction(Fraction(n)), "r3a-se-uses-the-variance-over-root-n"),
                (sigma, "r3a-se-is-the-population-sd"),
            ],
        )
    else:
        part = number_part(
            "Find the variance of the sample mean, correct to 2 decimal places.",
            se_var,
            2,
            1,
            [f"Var of the sample mean = sigma^2 / n = {float(sigma):g}^2 / {n}."],
            [
                (sigma * sigma / sqrt_fraction(Fraction(n)), "r3a-variance-of-mean-divides-by-root-n"),
                (sigma / n, "r3a-se-divides-by-n"),
                (se, "r3a-variance-of-mean-reported-as-sd"),
                (sigma * sigma, "r3a-se-is-the-population-sd"),
            ],
        )
    return single(f"{stem} {part['prompt']}", part)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium(rng: random.Random) -> dict:
    variant = rng.choice(("proportion", "scale_up", "scale_down"))
    if variant == "proportion":
        thing, who = rng.choice(_PROPORTIONS)
        for _ in attempts(500, "a clear standard deviation"):
            p = Fraction(rng.randint(2, 18), 20)
            n = rng.choice((40, 50, 60, 80, 100, 120, 150, 200, 250, 400))
            sd = sqrt_fraction(p * (1 - p) / n)
            if clear_of_boundary(sd, 4):
                break
        stem = (
            f"In a large population, the proportion of {thing} is {float(p):g}. A random sample of {n} {who} is "
            f"taken, and the sample proportion is recorded."
        )
        part = number_part(
            "Find the standard deviation of the sample proportion, correct to 4 decimal places.",
            sd,
            4,
            2,
            [f"SD = sqrt(p(1 - p)/n) = sqrt({float(p):g} × {float(1 - p):g} / {n})."],
            [
                (p * (1 - p) / n, "r3a-proportion-sd-omits-the-square-root"),
                (sqrt_fraction(p * (1 - p)), "r3a-proportion-sd-omits-n"),
                (sqrt_fraction(p / n), "r3a-proportion-sd-omits-the-failure-probability"),
                (p * (1 - p) / sqrt_fraction(Fraction(n)), "r3a-proportion-sd-roots-n-only"),
            ],
        )
        return single(f"{stem} {part['prompt']}", part)
    thing, unit, mu, sigma = _draw_context(rng)
    root = rng.choice((2, 3, 4, 5))
    factor = root * root
    n0 = rng.choice((10, 12, 15, 20, 24, 30, 36, 40, 50))
    for _ in attempts(500, "a standard error clear of a rounding boundary"):
        se0 = Fraction(rng.randint(30, 300), 100)
        if se0 != 1 and clear_of_boundary(se0 / root, 2):
            break
    if variant == "scale_up":
        n1 = n0 * factor
        value = se0 / root
        stem = (
            f"Sample means of {thing} are calculated from samples of {n0} observations, and they have standard "
            f"deviation {float(se0):g} {unit}. The sample size is now increased to {n1}."
        )
        slips = [
            (se0 / factor, "r3a-se-scales-with-n-not-root-n"),
            (se0 * root, "r3a-se-moves-the-wrong-way"),
        ]
        working = [f"The sample size is multiplied by {factor}, so the standard error is divided by {root}."]
    else:
        n1 = n0 * factor
        se_big = se0
        value = se_big * root
        stem = (
            f"Sample means of {thing} are calculated from samples of {n1} observations, and they have standard "
            f"deviation {float(se_big):g} {unit}. The sample size is cut to {n0}."
        )
        slips = [
            (se_big * factor, "r3a-se-scales-with-n-not-root-n"),
            (se_big / root, "r3a-se-moves-the-wrong-way"),
        ]
        working = [f"The sample size is divided by {factor}, so the standard error is multiplied by {root}."]
    check(sqrt_fraction(Fraction(factor)) == root, "the scale factor is not a perfect square")
    part = number_part(
        "Find the new standard deviation of the sample means, correct to 2 decimal places.", value, 2, 2, working, slips
    )
    return single(f"{stem} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------


def _least_n_mean(sigma: Fraction, target: Fraction) -> int:
    """The least n with sigma^2 / n <= target^2, found by direct search in exact arithmetic."""
    n = 1
    while sigma * sigma / n > target * target:
        n += 1
        check(n < 100000, "no n found")
    return n


def _hard(rng: random.Random) -> dict:
    variant = rng.choice(("least_n_mean", "least_n_proportion", "difference"))
    if variant == "difference":
        thing, unit, mu, s1 = _draw_context(rng)
        for _ in attempts(500, "a clear standard deviation of the difference"):
            s2 = s1 + Fraction(rng.randint(-4, 4), 2)
            n1, n2 = rng.choice((25, 30, 36, 40, 50, 64)), rng.choice((20, 25, 40, 45, 60, 80))
            if s2 <= 0 or s2 == s1:
                continue
            value = sqrt_fraction(s1 * s1 / n1 + s2 * s2 / n2)
            if clear_of_boundary(value, 2):
                break
        stem = (
            f"{thing[0].upper()}{thing[1:]} is measured at two sites. Site A has population standard deviation "
            f"{float(s1):g} {unit} and Site B has {float(s2):g} {unit}. Independent random samples of {n1} from A "
            f"and {n2} from B are taken, and the difference of the sample means (A minus B) is recorded."
        )
        part = number_part(
            "Find the standard deviation of this difference, correct to 2 decimal places.",
            value,
            2,
            3,
            [f"Var = {float(s1):g}^2/{n1} + {float(s2):g}^2/{n2}, and the standard deviation is its square root."],
            [
                (
                    abs(sqrt_fraction(s1 * s1 / n1) - sqrt_fraction(s2 * s2 / n2)),
                    "r3a-difference-subtracts-the-standard-errors",
                ),
                (sqrt_fraction(s1 * s1 / n1) + sqrt_fraction(s2 * s2 / n2), "r3a-difference-adds-the-standard-errors"),
                (sqrt_fraction(abs(s1 * s1 / n1 - s2 * s2 / n2)), "r3a-difference-subtracts-the-variances"),
                (s1 * s1 / n1 + s2 * s2 / n2, "r3a-se-omits-the-square-root"),
            ],
        )
        return single(f"{stem} {part['prompt']}", part)
    if variant == "least_n_mean":
        thing, unit, mu, sigma = _draw_context(rng)
        for _ in attempts(1000, "a target whose least n is not on a boundary"):
            target = Fraction(rng.randint(round(sigma * 10), round(sigma * 31)), 100)
            ratio = sigma * sigma / (target * target)
            if sigma / target > 3 and abs(float(ratio) - round(float(ratio))) > 0.15 and ratio < 3000:
                break
        n = _least_n_mean(sigma, target)
        check(sigma * sigma / n <= target * target < sigma * sigma / (n - 1), "n is not the least")
        stem = (
            f"{thing[0].upper()}{thing[1:]} has population standard deviation {float(sigma):g} {unit}. A researcher "
            f"wants the standard deviation of the sample mean to be at most {float(target):g} {unit}."
        )
        part = number_part(
            "Find the least sample size that achieves this.",
            Fraction(n),
            0,
            3,
            [
                f"sigma / sqrt(n) <= {float(target):g}, so n >= ({float(sigma):g} / {float(target):g})^2 = {float(ratio):.2f}.",
                f"The least whole number is {n}.",
            ],
            [
                (Fraction(math.floor(ratio)), "r3a-least-n-rounds-down"),
                (Fraction(math.ceil(sigma / target)), "r3a-least-n-omits-the-square"),
                (Fraction(math.ceil(sigma * sigma / target)), "r3a-least-n-squares-sigma-only"),
            ],
        )
        return single(f"{stem} {part['prompt']}", part)
    thing, who = rng.choice(_PROPORTIONS)
    for _ in attempts(2000, "a proportion target whose least n is clear"):
        p = Fraction(rng.randint(3, 17), 20)
        target = Fraction(rng.randint(15, 60), 1000)
        ratio = p * (1 - p) / (target * target)
        if abs(float(ratio) - round(float(ratio))) > 0.15 and 30 < ratio < 4000:
            break
    n = 1
    while p * (1 - p) / n > target * target:
        n += 1
        check(n < 100000, "no n found")
    check(p * (1 - p) / n <= target * target < p * (1 - p) / (n - 1), "n is not the least")
    stem = (
        f"In a large population the proportion of {thing} is {float(p):g}. A survey wants the standard deviation of "
        f"the sample proportion to be at most {float(target):g}."
    )
    part = number_part(
        "Find the least sample size that achieves this.",
        Fraction(n),
        0,
        3,
        [
            f"sqrt(p(1 - p)/n) <= {float(target):g}, so n >= {float(p):g} × {float(1 - p):g} / {float(target):g}^2 = {float(ratio):.2f}.",
            f"The least whole number is {n}.",
        ],
        [
            (Fraction(math.floor(ratio)), "r3a-least-n-rounds-down"),
            (Fraction(math.ceil(sqrt_fraction(p * (1 - p)) / target)), "r3a-least-n-omits-the-square"),
            (Fraction(math.ceil(p / (target * target))), "r3a-least-n-omits-the-failure-probability"),
        ],
    )
    return single(f"{stem} {part['prompt']}", part)


@generator(
    "r3a_sampling_moments",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3a_sampling_moments(rng: random.Random, tier: str = "easy") -> dict:
    """Sampling-distribution moments: a standard error (easy), a proportion or a rescaling (medium), a design (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    """Standard error, the effect of quadrupling n, the sample size for a target, and the standard error reached."""
    thing, unit, mu, _ = _draw_context(rng)
    for _ in attempts(5000, "a standard error that is exact and a target with a clear least n"):
        n0 = rng.choice((16, 25, 36, 49, 64, 81, 100, 144))
        root = int(math.isqrt(n0))
        se = Fraction(2 * rng.randint(20, 200), 100)  # an even number of hundredths, so half of it is exact to 2 places
        sigma = se * root
        if sigma == 1:
            continue
        target = Fraction(round(se * rng.randint(35, 80)), 100)
        if target < Fraction(1, 5) or target == 1:
            continue
        ratio = sigma * sigma / (target * target)
        if not (abs(float(ratio) - round(float(ratio))) > 0.15 and ratio < 2000):
            continue
        n_least = _least_n_mean(sigma, target)
        se_least = sqrt_fraction(sigma * sigma / n_least)
        if clear_of_boundary(se_least, 2) and se_least <= target and (sigma * 100).denominator == 1:
            break
    check(sigma * sigma / n_least <= target * target < sigma * sigma / (n_least - 1), "n is not the least")
    stem = (
        f"{thing[0].upper()}{thing[1:]} has population mean {mu} {unit} and population standard deviation "
        f"{float(sigma):g} {unit}. Random samples of {n0} are taken and the sample mean is calculated for each."
    )
    parts = [
        number_part(
            "Find the standard deviation of the sample mean, correct to 2 decimal places.",
            se,
            2,
            1,
            [f"sigma / sqrt(n) = {float(sigma):g} / sqrt({n0}) = {float(se):.2f}."],
            [
                (sigma / n0, "r3a-se-divides-by-n"),
                (sigma * root, "r3a-se-multiplies-by-root-n"),
                (sigma * sigma / root, "r3a-se-uses-the-variance-over-root-n"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"The sample size is quadrupled to {4 * n0}. Find the new standard deviation of the sample mean, correct to "
            "2 decimal places.",
            se / 2,
            2,
            1,
            [f"Quadrupling n halves the standard error: {float(se):.2f} / 2."],
            [
                (se / 4, "r3a-se-scales-with-n-not-root-n"),
                (se * 2, "r3a-se-moves-the-wrong-way"),
                (se / 16, "r3a-se-scales-with-n-squared"),
            ],
            follow_through="part_a/2",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the least sample size for which the standard deviation of the sample mean is at most "
            f"{float(target):g} {unit}.",
            Fraction(n_least),
            0,
            3,
            [
                f"sigma = {float(se):.2f} × sqrt({n0}) = {float(sigma):g}, and n >= ({float(sigma):g} / {float(target):g})^2 = {float(ratio):.2f}.",
                f"The least whole number is {n_least}.",
            ],
            [
                (Fraction(math.floor(ratio)), "r3a-least-n-rounds-down"),
                (Fraction(math.ceil(sigma / target)), "r3a-least-n-omits-the-square"),
                (Fraction(math.ceil(float(se) / float(target))), "r3a-least-n-uses-the-standard-error-for-sigma"),
            ],
            follow_through=f"ceiling((part_a*{root}/({target.numerator}/{target.denominator}))**2)",
            sensible_range=(0, None),
        ),
        number_part(
            "For that sample size, find the standard deviation of the sample mean, correct to 2 decimal places.",
            se_least,
            2,
            1,
            [f"sigma / sqrt({n_least}) = {float(sigma):g} / sqrt({n_least}) = {float(se_least):.2f}."],
            [
                (sigma / n_least, "r3a-se-divides-by-n"),
                (target, "r3a-se-reports-the-target"),
                (sigma / sqrt_fraction(Fraction(n_least - 1)), "r3a-se-uses-one-less-than-n"),
                (sigma / sqrt_fraction(Fraction(n_least + 1)), "r3a-se-uses-one-more-than-n"),
                (sigma, "r3a-se-is-the-population-sd"),
            ],
            follow_through=f"part_a*{root}/sqrt(part_c)",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3a_sampling_moments", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3a_sampling_moments(rng: random.Random, tier: str = "exam") -> dict:
    """Standard error, the effect of quadrupling n, the sample size for a target, and the standard error reached."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
