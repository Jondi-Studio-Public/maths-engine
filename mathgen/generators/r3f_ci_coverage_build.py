"""Confidence-interval coverage and simulated intervals: `r3f_ci_coverage` and its exam chain.

Skill `confidence-interval-simulation-coverage`. A C% confidence interval is produced by a method that captures the
population value in C% of repeated samples, so in a table of independent simulated intervals about (100 - C)% miss it.

Tiers (topic `r3f_ci_coverage`), each with a different piece of reasoning:

* easy   -- the first stage of any interval for a proportion: the margin of error (or an end of the interval) from the raw
            counts x out of n, with a stated critical z value;
* medium -- reading a table of simulated intervals: count the intervals that miss the true mean (or contain it, or miss
            it from above), the widths varying from sample to sample;
* hard   -- what the coverage rate predicts: the expected number of misses in N intervals, the probability that all N
            capture the true value, the probability that at least one misses; and how much wider the wrong interval
            "mean +/- 1.96 s" is than the correct "mean +/- 1.96 s / sqrt(n)".

The exam chain (topic `exd_x_r3f_ci_coverage`) is on 20 simulated 95% intervals for a mean (sigma known): (a) the margin
of error, (b) the number of intervals that miss the true mean, (c) the proportion that missed (follows part b), (d) a
multiple-choice reading of what the 95% means given that count.

Every count is re-derived from the table's own endpoints, and every probability by exact powers of Fractions. A draw is
rejected when an endpoint equals the true mean or when a rounded answer sits near a rounding boundary.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3f_kit import ThinOptions, build_until, clear_of_boundary, dec, mcq_part, number_part, single, sqrt_fraction

Z = {90: Fraction("1.645"), 95: Fraction("1.96"), 99: Fraction("2.576")}


def zt(level: int) -> str:
    return f"{float(Z[level]):g}"


# -- easy ------------------------------------------------------------------------------------------------------

POLLS = (
    (
        "In a poll of {n} randomly chosen adults, {x} said they use public transport at least once a week.",
        "who use public transport weekly",
    ),
    (
        "A council surveyed {n} randomly chosen households, and {x} of them have solar panels.",
        "of households with solar panels",
    ),
    (
        "In a random sample of {n} Year 10 students, {x} said they walk to school.",
        "of Year 10 students who walk to school",
    ),
    (
        "A quality inspector checked {n} randomly chosen phone cases and found {x} with a visible flaw.",
        "of cases with a visible flaw",
    ),
)


def _easy(rng: random.Random) -> dict:
    for _ in attempts(500, "a margin of error clear of a rounding boundary"):
        n = rng.choice((200, 250, 400, 500, 600, 800, 1000, 1200, 1500))
        x = round(n * rng.randint(2, 18) / 20)
        level = rng.choice((90, 95, 95, 99))
        p = Fraction(x, n)
        me = 100 * Z[level] * sqrt_fraction(p * (1 - p) / n)
        upper = 100 * p + me
        ask = rng.choice(("margin", "upper"))
        value = me if ask == "margin" else upper
        if clear_of_boundary(value, 1, Fraction(1, 5)):
            break
    setting, what = rng.choice(POLLS)
    stem = setting.format(n=n, x=x)
    if ask == "margin":
        part = number_part(
            f"Using z = {zt(level)} for a {level}% confidence interval, find the margin of error for the population "
            "proportion, in percentage points, correct to 1 decimal place.",
            value,
            1,
            2,
            [
                f"p-hat = {x}/{n} = {float(p):g}.",
                f"Margin of error = {zt(level)} x √(p-hat(1 - p-hat)/n) x 100 = {float(value):.1f}.",
            ],
            [
                (100 * sqrt_fraction(p * (1 - p) / n), "r3f-ci-margin-omits-the-critical-value"),
                (100 * Z[level] * sqrt_fraction(p / n), "r3f-ci-margin-omits-the-failure-proportion"),
                (2 * value, "r3f-ci-margin-reports-the-whole-width"),
            ],
            sensible_range=(0, 100),
        )
    else:
        part = number_part(
            f"Using z = {zt(level)} for a {level}% confidence interval, find the upper end of the interval for the population "
            "percentage, correct to 1 decimal place.",
            value,
            1,
            3,
            [
                f"p-hat = {x}/{n} = {float(p):g}.",
                f"Margin of error = {float(me):.2f} percentage points, so the upper end is {float(100 * p):g} + {float(me):.2f}.",
            ],
            [
                (100 * p - me, "r3f-ci-upper-end-reports-the-lower-end"),
                (me, "r3f-ci-upper-end-reports-the-margin"),
                (100 * p + 100 * sqrt_fraction(p * (1 - p) / n), "r3f-ci-margin-omits-the-critical-value"),
                (100 * p, "r3f-ci-upper-end-reports-the-estimate"),
            ],
            sensible_range=(0, 100),
        )
    return single(f"{stem} {part['prompt']}", part)


# -- simulated intervals -----------------------------------------------------------------------------------------


def _interval_table(rng, mu, se, half, count, varying: bool):
    """`count` intervals (lo, hi) as one-decimal Fractions from independent sample means; half is the nominal half-width."""
    rows = []
    for _ in range(count):
        centre = Fraction(round((mu + se * rng.gauss(0, 1)) * 10), 10)
        h = half * (Fraction(round(rng.uniform(0.75, 1.3) * 100), 100) if varying else 1)
        h = Fraction(round(float(h) * 10), 10) if varying else Fraction(h)
        rows.append((centre - h, centre + h))
    return rows


def _table_text(rows) -> str:
    return "; ".join(f"{i}: {float(lo):g} to {float(hi):g}" for i, (lo, hi) in enumerate(rows, start=1))


def _misses(rows, mu):
    below = sum(1 for lo, hi in rows if hi < mu)  # whole interval below mu
    above = sum(1 for lo, hi in rows if lo > mu)  # whole interval above mu
    return below, above


def _medium(rng: random.Random) -> dict:
    ask = rng.choice(("miss", "contain", "above"))
    for _ in attempts(1000, "a table with a few misses and no endpoint on the true mean"):
        mu = rng.choice((40, 50, 60, 100, 120))
        se = rng.choice((2, 3, 4, 5))
        level = rng.choice((90, 95, 95, 99))
        count = rng.choice((20, 20, 25, 30))
        half = Z[level] * se
        rows = _interval_table(rng, mu, se, half, count, True)
        below, above = _misses(rows, mu)
        miss = below + above
        if any(lo == mu or hi == mu for lo, hi in rows):
            continue
        expected = Fraction(count * (100 - level), 100)
        if 1 <= miss <= 5 and below != above and miss != expected and (ask != "above" or above >= 1):
            break
    contain = count - miss
    value = {"miss": miss, "contain": contain, "above": above}[ask]
    check(contain == sum(1 for lo, hi in rows if lo < mu < hi), "containment count disagrees")
    words = {
        "miss": "do not contain μ",
        "contain": "contain μ",
        "above": "lie entirely above μ",
    }[ask]
    slips = {
        "miss": [
            (contain, "r3f-ci-counts-the-intervals-that-contain"),
            (expected, "r3f-ci-reports-the-expected-count-not-the-observed"),
            (miss + 1, "r3f-ci-miss-count-off-by-one"),
            (below, "r3f-ci-counts-only-one-side"),
        ],
        "contain": [
            (miss, "r3f-ci-counts-the-intervals-that-miss"),
            (Fraction(count * level, 100), "r3f-ci-reports-the-expected-count-not-the-observed"),
            (contain - 1, "r3f-ci-miss-count-off-by-one"),
            (contain + 1, "r3f-ci-miss-count-off-by-one"),
        ],
        "above": [
            (miss, "r3f-ci-counts-both-sides"),
            (below, "r3f-ci-counts-the-wrong-side"),
            (contain, "r3f-ci-counts-the-intervals-that-contain"),
            (above + 1, "r3f-ci-miss-count-off-by-one"),
        ],
    }[ask]
    stem = (
        f"A researcher takes {count} independent random samples from a population whose true mean is μ = {mu}, and "
        f"from each sample builds a {level}% confidence interval for μ. The intervals are: {_table_text(rows)}."
    )
    part = number_part(
        f"How many of the {count} intervals {words}?",
        value,
        0,
        3,
        [
            "Check each interval against μ: it contains μ when its lower end is below and its upper end is above.",
            f"Count = {value}.",
        ],
        slips,
        sensible_range=(-1, count + 1),
    )
    return single(f"{stem} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------


def _expected(rng):
    count = rng.choice((20, 30, 40, 50, 60, 80, 100, 200))
    level = rng.choice((90, 95, 99))
    exp = Fraction(count * (100 - level), 100)
    part = number_part(
        f"On average, how many of the {count} intervals would you expect NOT to contain μ? Give your answer correct to 1 decimal place.",
        exp,
        1,
        2,
        [
            f"Each interval misses μ with probability {(100 - level) / 100:g}.",
            f"Expected number = {count} x {(100 - level) / 100:g} = {float(exp):g}.",
        ],
        [
            (Fraction(count * level, 100), "r3f-ci-reports-the-count-that-contain"),
            (Fraction(level, 100) * (100 - level), "r3f-ci-expected-count-uses-percent-as-count"),
            (Fraction(count, 100 - level), "r3f-ci-expected-count-divides-by-the-miss-rate"),
            (Fraction(100 - level), "r3f-ci-expected-count-ignores-the-number-of-intervals"),
        ],
        sensible_range=(0, count),
    )
    stem = (
        f"A researcher builds {count} independent {level}% confidence intervals for a population mean μ, each from its "
        "own random sample."
    )
    return single(f"{stem} {part['prompt']}", part)


def _all_contain(rng):
    for _ in attempts(300, "a coverage probability clear of a rounding boundary"):
        count = rng.choice((5, 8, 10, 12, 15, 20, 25))
        level = rng.choice((90, 95, 99))
        c = Fraction(level, 100)
        p_all = c**count
        ask = rng.choice(("all", "some"))
        value = p_all if ask == "all" else 1 - p_all
        if clear_of_boundary(value, 4, Fraction(1, 5)) and 0.03 < value < 0.97:
            break
    stem = (
        f"A researcher builds {count} independent {level}% confidence intervals for a population mean μ, each from its "
        "own random sample."
    )
    if ask == "all":
        part = number_part(
            f"Find the probability that all {count} intervals contain μ, correct to 4 decimal places.",
            value,
            4,
            4,
            [
                f"Each interval contains μ with probability {level / 100:g}, independently.",
                f"P(all contain) = {level / 100:g}^{count} = {float(value):.4f}.",
            ],
            [
                (1 - p_all, "r3f-ci-gives-the-complement"),
                (c, "r3f-ci-ignores-the-number-of-intervals"),
                (1 - (1 - c) * count if (1 - (1 - c) * count) > 0 else None, "r3f-ci-adds-the-miss-probabilities"),
                (c * (1 - c) ** (count - 1), "r3f-ci-misses-and-contains-swapped"),
            ],
            sensible_range=(0, 1),
            within=(0, 1),
        )
    else:
        part = number_part(
            f"Find the probability that at least one of the {count} intervals does not contain μ, correct to 4 decimal places.",
            value,
            4,
            4,
            [
                f"P(all contain) = {level / 100:g}^{count} = {float(p_all):.4f}.",
                f"P(at least one misses) = 1 - {float(p_all):.4f} = {float(value):.4f}.",
            ],
            [
                (p_all, "r3f-ci-reports-the-probability-that-all-contain"),
                (1 - c, "r3f-ci-ignores-the-number-of-intervals"),
                ((1 - c) * count if (1 - c) * count < 1 else None, "r3f-ci-adds-the-miss-probabilities"),
                (1 - c**2, "r3f-ci-uses-two-intervals"),
            ],
            sensible_range=(0, 1),
            within=(0, 1),
        )
    return single(f"{stem} {part['prompt']}", part)


def _too_wide(rng):
    for _ in attempts(500, "a width difference clear of a rounding boundary"):
        n = rng.choice((9, 16, 25, 36, 49, 64, 100))
        root = math.isqrt(n)
        s = rng.randint(3, 40)
        diff = 2 * Z[95] * s * (1 - Fraction(1, root))
        if clear_of_boundary(diff, 2, Fraction(1, 5)):
            break
    part = number_part(
        "Find how much wider the student's interval is than the correct interval, correct to 2 decimal places.",
        diff,
        2,
        3,
        [
            f"The correct interval has half-width 1.96 x {s}/√{n} = {float(Z[95] * s / root):.3f}; the student's has half-width 1.96 x {s} = {float(Z[95] * s):.3f}.",
            f"The difference in width is 2 x ({float(Z[95] * s):.3f} - {float(Z[95] * s / root):.3f}) = {float(diff):.2f}.",
        ],
        [
            (Z[95] * s * (1 - Fraction(1, root)), "r3f-ci-width-difference-uses-half-widths"),
            (2 * Z[95] * s, "r3f-ci-width-difference-is-the-whole-student-width"),
            (2 * Z[95] * s * (1 - Fraction(1, n)), "r3f-ci-divides-by-n-not-root-n"),
            (2 * Z[95] * s / root, "r3f-ci-width-difference-is-the-correct-width"),
        ],
        sensible_range=(0, None),
    )
    stem = (
        f"A random sample of n = {n} observations from a population has standard deviation s = {s}. A student writes the 95% "
        "confidence interval for the population mean as (sample mean) ± 1.96s, instead of (sample mean) ± 1.96s/√n."
    )
    return single(f"{stem} {part['prompt']}", part)


def _hard(rng):
    return rng.choice((_expected, _all_contain, _too_wide))(rng)


@generator(
    "r3f_ci_coverage",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3f_ci_coverage(rng: random.Random, tier: str = "easy") -> dict:
    """Margin of error (easy), counting misses in simulated intervals (medium), what coverage predicts (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    for _ in attempts(4000, "a table of intervals with a few misses"):
        mu = rng.choice((40, 50, 60, 100, 120))
        sigma = rng.randint(6, 30)
        n = rng.choice((9, 16, 25, 36, 49, 64, 81, 100))
        root = math.isqrt(n)
        count = rng.choice((20, 40, 80))
        se = Fraction(sigma, root)
        me = Z[95] * se
        if not clear_of_boundary(me, 2, Fraction(1, 5)):
            continue
        half = Fraction(round(me * 100), 100)  # the margin as printed, so every interval is its centre plus or minus it
        rows = _interval_table(rng, mu, float(se), half, count, False)
        below, above = _misses(rows, mu)
        miss = below + above
        if any(lo <= 0 or lo == mu or hi == mu for lo, hi in rows):
            continue
        if 2 <= miss <= count // 10 + 1 and miss != count // 20:
            break
    check(miss == sum(1 for lo, hi in rows if not lo < mu < hi), "miss count disagrees")
    places = {20: 2, 40: 3, 80: 4}[count]
    proportion = Fraction(miss, count)
    stem = (
        f"The mean of a population is μ = {mu} and its standard deviation is σ = {sigma}. A researcher takes {count} independent "
        f"random samples of size {n} and from each builds a 95% confidence interval for μ, using the critical value 1.96 "
        f"and σ. The intervals are: {_table_text(rows)}."
    )
    key = f"About 5% of such intervals miss μ, so about {count // 20} of {count} is expected and a count of {miss} is no sign of a fault."
    parts = [
        number_part(
            "Find the margin of error of each interval, correct to 2 decimal places.",
            me,
            2,
            2,
            [f"Margin of error = 1.96 x {sigma}/√{n} = {float(me):.2f}."],
            [
                (Z[95] * sigma, "r3f-ci-margin-omits-the-square-root-of-n"),
                (Z[95] * Fraction(sigma, n), "r3f-ci-divides-by-n-not-root-n"),
                (se, "r3f-ci-margin-omits-the-critical-value"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"How many of the {count} intervals do not contain μ?",
            miss,
            0,
            3,
            [
                "Check each interval against μ: it contains μ only when its lower end is below and its upper end is above."
            ],
            [
                (count - miss, "r3f-ci-counts-the-intervals-that-contain"),
                (count // 20, "r3f-ci-reports-the-expected-count-not-the-observed"),
                (miss + 1, "r3f-ci-miss-count-off-by-one"),
                (max(miss - 1, 0) if miss > 1 else None, "r3f-ci-miss-count-off-by-one"),
            ],
            sensible_range=(-1, count + 1),
        ),
        number_part(
            f"What proportion of the {count} intervals missed μ? Give your answer as a decimal.",
            proportion,
            places,
            1,
            [f"{miss}/{count} = {float(proportion):.{places}f}."],
            [
                (1 - proportion, "r3f-ci-reports-the-proportion-that-contain"),
                (Fraction(1, 20), "r3f-ci-reports-the-expected-count-not-the-observed"),
                (Fraction(miss, 100), "r3f-ci-proportion-divides-by-100"),
            ],
            follow_through=f"part_b/{count}",
            sensible_range=(-0.0001, 1.0001),
            within=(-1, 2),
        ),
        mcq_part(
            f"Which statement is correct, given that {miss} of the {count} intervals missed μ?",
            key,
            2,
            [
                f"A 95% confidence level describes the long-run proportion of intervals that capture μ, not a guarantee for a set of {count}."
            ],
            [
                (
                    f"Exactly {count // 20} of the {count} intervals must miss μ, so a count of {miss} proves that the method has failed.",
                    "r3f-ci-coverage-treated-as-exact",
                ),
                (
                    "Each interval has a 95% chance of containing the sample mean, so none of them should ever miss μ at all.",
                    "r3f-ci-confused-with-sample-mean",
                ),
                (
                    "A 95% interval holds 95% of the individual values in the population, so the count means nothing at all.",
                    "r3f-ci-confused-with-data-spread",
                ),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3f_ci_coverage", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3f_ci_coverage(rng: random.Random, tier: str = "exam") -> dict:
    """Margin of error, the number of misses in a table of intervals, the proportion, and what 95% means."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
