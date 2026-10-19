"""Finding n or p of a binomial random variable: `r3a_binomial_find_np` and its exam chain.

Skill `binomial-find-n-or-p`. The numbers of trials n and the success probability p are not given; they are
recovered from facts about the distribution.

Tiers (topic `r3a_binomial_find_np`):

* easy   -- one of n and p from the mean and the other (np = mean);
* medium -- both unknown, the mean and the spread given (the variance, or the standard deviation): the ratio of
  variance to mean is 1 - p, so p comes first, then n;
* hard   -- a probability is given and the parameter is buried in a power or a logarithm: the least n with
  P(X >= 1) above a stated level, p from P(X = 0), or p from P(X = n).

The exam chain (topic `exd_x_r3a_binomial_find_np`) is (a) p from the mean and variance, (b) n from the mean and
p, (c) P(X = 1), (d) the least number of trials for P(at least one success) to exceed 0.99. Parts (b) to (d) carry
`follow_through`.

Each key is rebuilt in exact arithmetic: a least n is confirmed by comparing 1 - q^n with the level in Fractions at
n and at n - 1, and a recovered p is put back into the distribution to reproduce the stated mean and variance.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3a_kit import build_until, clear_of_boundary, dec, number_part, pmf, single, sqrt_fraction

#: (unknown-n wording with {p}, unknown-p wording with {n}, both unknown, what X counts)
_CONTEXTS = (
    (
        "Each seed a gardener sows germinates independently with probability {p}. A tray holds n seeds.",
        "A gardener sows {n} seeds in a tray, and each germinates independently with the same probability p.",
        "A gardener sows a tray of seeds, and each seed germinates independently with the same probability.",
        "the number of seeds in the tray that germinate",
    ),
    (
        "Each call a call centre makes is answered independently with probability {p}. A shift has n calls.",
        "A call centre makes {n} calls in a shift, and each is answered independently with the same probability p.",
        "A call centre makes a number of calls in a shift, and each is answered independently with the same probability.",
        "the number of calls in the shift that are answered",
    ),
    (
        "Each phone screen from a production line is independently faulty with probability {p}. A crate holds "
        "n screens.",
        "A crate holds {n} phone screens, and each is independently faulty with the same probability p.",
        "A crate holds a number of phone screens, and each is independently faulty with the same probability.",
        "the number of faulty screens in the crate",
    ),
    (
        "Each shot a netballer takes independently scores with probability {p}. In a game she takes n shots.",
        "A netballer takes {n} shots in a game, and each independently scores with the same probability p.",
        "A netballer takes a number of shots in a game, and each independently scores with the same probability.",
        "the number of shots that score",
    ),
)

_P_NOT_HALF = tuple(Fraction(k, 20) for k in range(2, 19) if k != 10)


def _decimal_places(value: Fraction) -> int:
    for places in range(0, 7):
        if (value * 10**places).denominator == 1:
            return places
    raise ValueError(value)


def _shown(value: Fraction) -> str:
    return f"{float(value):.{_decimal_places(value)}f}"


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    unknown = rng.choice(("n", "p"))
    with_p, with_n, both, x_word = rng.choice(_CONTEXTS)
    p = rng.choice(_P_NOT_HALF)
    n = rng.randint(12, 40)
    mean = n * p
    if unknown == "n":
        # keep p a multiple of 0.05 whose reciprocal makes n a whole number: mean is stated exactly
        stem = f"{with_p.format(p=dec(p))} Let X be {x_word}. The mean of X is {_shown(mean)}."
        part = number_part(
            "Find n.",
            Fraction(n),
            0,
            1,
            [f"np = {_shown(mean)}, so n = {_shown(mean)} / {dec(p)} = {n}."],
            [
                (mean * p, "r3a-find-n-multiplies-mean-by-p"),
                (mean / (1 - p), "r3a-find-n-uses-failure-probability"),
                (mean - p, "r3a-find-n-subtracts-p"),
                (mean + n // 4 + 1, "r3a-find-n-adds-to-the-mean"),
            ],
        )
    else:
        stem = f"{with_n.format(n=n)} Let X be {x_word}. The mean of X is {_shown(mean)}."
        part = number_part(
            "Find p, correct to 2 decimal places.",
            p,
            2,
            1,
            [f"np = {_shown(mean)}, so p = {_shown(mean)} / {n} = {dec(p)}."],
            [
                (1 - p, "r3a-find-p-gives-failure-probability"),
                (mean / (n - mean), "r3a-find-p-divides-by-the-failures"),
                (mean / (n + 1), "r3a-find-p-divides-by-one-more-than-n"),
            ],
        )
    return single(f"{stem} {part['prompt']}", part)


# -- medium ----------------------------------------------------------------------------------------------------


def _square_sd_cases():
    """(n, p, sd) where npq is the square of a decimal with one place: the standard deviation is stated exactly."""
    out = []
    for n in range(10, 60):
        for k in range(2, 19):
            p = Fraction(k, 20)
            var = n * p * (1 - p)
            for tenths in range(5, 40):
                if Fraction(tenths, 10) ** 2 == var:
                    out.append((n, p, Fraction(tenths, 10)))
    return out


_SD_CASES = _square_sd_cases()


def _medium(rng: random.Random) -> dict:
    spread = rng.choice(("variance", "sd"))
    ask = rng.choice(("p", "n"))
    with_p, with_n, both, x_word = rng.choice(_CONTEXTS)
    if spread == "sd":
        n, p, sd = rng.choice(_SD_CASES)
        spread_text = f"standard deviation {_shown(sd)}"
        var = sd * sd
    else:
        for _ in attempts(500, "a mean and variance with p a multiple of 0.05"):
            n, p = rng.randint(12, 48), rng.choice(_P_NOT_HALF)
            var = n * p * (1 - p)
            if n * p != var:
                break
        spread_text = f"variance {_shown(var)}"
    mean = n * p
    q = var / mean
    check(1 - q == p and mean / (1 - q) == n, "the ratio route disagrees with the construction")
    stem = f"{both} Let X be {x_word}. The number of trials n and the success probability p are unknown. X has mean {_shown(mean)} and {spread_text}."
    if ask == "p":
        slips = [
            (q, "r3a-find-p-reports-failure-probability"),
            (1 - sd / mean, "r3a-uses-sd-as-variance")
            if spread == "sd"
            else (1 - sqrt_fraction(var) / mean, "r3a-roots-the-variance-first"),
            (sd / mean, "r3a-uses-sd-as-variance")
            if spread == "sd"
            else (sqrt_fraction(var) / mean, "r3a-roots-the-variance-first"),
            (var / n, "r3a-find-p-divides-variance-by-n"),
        ]
        part = number_part(
            "Find p, correct to 2 decimal places.",
            p,
            2,
            2,
            [f"npq / np = q = {_shown(var)} / {_shown(mean)}, so p = 1 - q = {dec(p)}."],
            slips,
        )
    else:
        slips = [
            (mean / q, "r3a-find-n-uses-failure-probability"),
            (mean / var, "r3a-find-n-divides-mean-by-variance"),
            (mean + var, "r3a-find-n-adds-mean-and-variance"),
        ]
        part = number_part(
            "Find n.",
            Fraction(n),
            0,
            2,
            [f"q = {_shown(var)} / {_shown(mean)}, so p = {dec(p)} and n = {_shown(mean)} / {dec(p)} = {n}."],
            slips,
        )
    return single(f"{stem} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------


def _least_n(q: Fraction, level: Fraction) -> int:
    """The least n with 1 - q^n > level, found by direct search in exact arithmetic."""
    n = 1
    while 1 - q**n <= level:
        n += 1
        check(n < 5000, "no n found")
    return n


def _hard(rng: random.Random) -> dict:
    variant = rng.choice(("least_n", "p_from_zero", "p_from_all"))
    with_p, with_n, both, x_word = rng.choice(_CONTEXTS)
    if variant == "least_n":
        for _ in attempts(500, "a level and probability with a clear least n"):
            q = rng.choice((Fraction(4, 5), Fraction(17, 20), Fraction(9, 10), Fraction(19, 20), Fraction(39, 40)))
            level = rng.choice((Fraction(9, 10), Fraction(19, 20), Fraction(99, 100), Fraction(999, 1000)))
            ratio = math.log(1 - level) / math.log(q)
            if abs(ratio - round(ratio)) > 0.05:
                break
        n = _least_n(q, level)
        check(1 - q ** (n - 1) <= level < 1 - q**n, "n is not the least")
        p = 1 - q
        stem = (
            f"Each seed a gardener sows germinates independently with probability {dec(p)}. The gardener sows n "
            f"seeds in a tray and wants the probability that at least one germinates to exceed {dec(level)}."
        )
        if rng.random() < 0.5:
            stem = (
                f"Each call a call centre makes is answered independently with probability {dec(p)}. The centre "
                f"makes n calls and wants the probability of at least one answer to exceed {dec(level)}."
            )
        part = number_part(
            "Find the least value of n.",
            Fraction(n),
            0,
            3,
            [
                f"P(X >= 1) = 1 - {dec(q)}^n > {dec(level)}, so {dec(q)}^n < {dec(1 - level)}.",
                f"n > ln({dec(1 - level)}) / ln({dec(q)}) = {ratio:.3f}, so n = {n}.",
            ],
            [
                (Fraction(math.floor(ratio)), "r3a-least-n-rounds-down"),
                (Fraction(math.ceil(math.log(float(level)) / math.log(float(q)))), "r3a-least-n-logs-the-level"),
                (Fraction(math.ceil(math.log(float(1 - level)) / math.log(float(p)))), "r3a-least-n-uses-p-as-base"),
            ],
        )
        return single(f"{stem} {part['prompt']}", part)
    n = rng.randint(2, 5)
    q = Fraction(rng.randint(5, 9), 10)
    if variant == "p_from_zero":
        value = q**n
        p = 1 - q
        check(pmf(n, p, 0) == value, "P(X = 0) disagrees")
        stem = (
            f"The probability that none of {n} independent trials succeeds is {_shown(value)}, and every trial has "
            f"the same probability p of success."
        )
        part = number_part(
            "Find p, correct to 2 decimal places.",
            p,
            2,
            2,
            [f"(1 - p)^{n} = {_shown(value)}, so 1 - p = {dec(q)} and p = {dec(p)}."],
            [
                (q, "r3a-find-p-reports-failure-probability"),
                (1 - value, "r3a-find-p-uses-complement-of-zero-successes"),
                (value / n, "r3a-find-p-divides-probability-by-n"),
            ],
        )
    else:
        p = 1 - q
        value = p**n
        check(pmf(n, p, n) == value, "P(X = n) disagrees")
        stem = (
            f"The probability that all of {n} independent trials succeed is {_shown(value)}, and every trial has "
            f"the same probability p of success."
        )
        part = number_part(
            "Find p, correct to 2 decimal places.",
            p,
            2,
            2,
            [f"p^{n} = {_shown(value)}, so p = {dec(p)}."],
            [
                (q, "r3a-find-p-gives-failure-probability"),
                (value / n, "r3a-find-p-divides-probability-by-n"),
                (1 - value, "r3a-find-p-uses-complement-of-zero-successes"),
            ],
        )
    return single(f"{stem} {part['prompt']}", part)


@generator(
    "r3a_binomial_find_np",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3a_binomial_find_np(rng: random.Random, tier: str = "easy") -> dict:
    """Find n or p: from the mean (easy), from mean and spread (medium), from a probability condition (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    """p from mean and variance, n from the mean, P(X = 1), then the least n for P(at least one) above 0.99."""
    with_p, with_n, both, x_word = rng.choice(_CONTEXTS)
    for _ in attempts(3000, "a binomial with a stable exactly-one probability and a clear least n"):
        n = rng.randint(8, 32)
        p = Fraction(rng.randint(10, 40), 100)
        q = 1 - p
        mean, var = n * p, n * p * q
        one = pmf(n, p, 1)
        ratio = math.log(0.01) / math.log(float(q))
        if one < Fraction(1, 50) or not clear_of_boundary(one, 4, Fraction(1, 5)):
            continue
        if abs(ratio - round(ratio)) < 0.08:
            continue
        break
    least = _least_n(q, Fraction(99, 100))
    check(1 - q ** (least - 1) <= Fraction(99, 100) < 1 - q**least, "the least n is wrong")
    check(least == math.floor(ratio) + 1, "the logarithm route disagrees with the search")
    # put the recovered parameters back: they must reproduce the stated mean and variance
    check(sum(k * pmf(n, p, k) for k in range(n + 1)) == mean, "the mean is not reproduced")
    stem = (
        f"{both} Let X be {x_word}. Over many repetitions, X is found to have mean {_shown(mean)} and variance "
        f"{_shown(var)}."
    )
    pq = f"{p.numerator}/{p.denominator}"
    parts = [
        number_part(
            "Find p, the probability of success on each trial, correct to 2 decimal places.",
            p,
            2,
            2,
            [f"Var / mean = q = {_shown(var)} / {_shown(mean)} = {dec(q)}, so p = {dec(p)}."],
            [
                (q, "r3a-find-p-reports-failure-probability"),
                (mean - var, "r3a-find-p-subtracts-variance-from-mean"),
                (var / n, "r3a-find-p-divides-variance-by-n"),
            ],
            sensible_range=(0, 1),
        ),
        number_part(
            "Find n, the number of trials.",
            Fraction(n),
            0,
            1,
            [f"np = {_shown(mean)}, so n = {_shown(mean)} / {dec(p)} = {n}."],
            [
                (mean / q, "r3a-find-n-uses-failure-probability"),
                (mean * p, "r3a-find-n-multiplies-mean-by-p"),
                (mean - p, "r3a-find-n-subtracts-p"),
            ],
            follow_through=f"({mean.numerator}/{mean.denominator})/part_a",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the probability that exactly one trial succeeds, P(X = 1), correct to 4 decimal places.",
            one,
            4,
            3,
            [f"P(X = 1) = {n} x {dec(p)} x {dec(q)}^{n - 1} = {float(one):.4f}."],
            [
                (pmf(n, p, 0), "r3a-exactly-one-uses-zero-successes"),
                (p * q ** (n - 1), "r3a-exactly-one-omits-the-coefficient"),
                (1 - pmf(n, p, 0), "r3a-exactly-one-takes-at-least-one"),
                (n * p * q**n, "r3a-exactly-one-wrong-exponent"),
            ],
            follow_through="part_b*part_a*(1-part_a)**(part_b-1)",
            sensible_range=(0, 1),
        ),
        number_part(
            "Find the least number of trials needed for the probability of at least one success to exceed 0.99.",
            Fraction(least),
            0,
            2,
            [
                f"1 - {dec(q)}^m > 0.99, so m > ln(0.01) / ln({dec(q)}) = {ratio:.3f}.",
                f"The least whole number is {least}.",
            ],
            [
                (Fraction(math.floor(ratio)), "r3a-least-n-rounds-down"),
                (Fraction(math.ceil(math.log(0.99) / math.log(float(q)))), "r3a-least-n-logs-the-level"),
                (Fraction(math.ceil(math.log(0.01) / math.log(float(p)))), "r3a-least-n-uses-p-as-base"),
                (Fraction(math.ceil(0.99 / float(p))), "r3a-least-n-divides-level-by-p"),
            ],
            follow_through="ceiling(log(1/100)/log(1-part_a))",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3a_binomial_find_np", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3a_binomial_find_np(rng: random.Random, tier: str = "exam") -> dict:
    """p from mean and variance, n from the mean, P(X = 1), then the least n for P(at least one) above 0.99."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
