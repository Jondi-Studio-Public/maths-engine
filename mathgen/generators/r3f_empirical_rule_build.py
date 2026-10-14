"""The 68-95-99.7 rule for normal curves: `r3f_empirical_rule` and its exam chain.

Skill `normal-curve-empirical-rule`. A normal variable has mean mu and standard deviation sigma. The rule gives the
share of values within 1, 2 and 3 standard deviations of the mean as 68%, 95% and 99.7%, so the share in each band
between neighbouring whole-number z values is 34, 13.5, 2.35 and 0.15 per cent on each side.

Tiers (topic `r3f_empirical_rule`), each a different piece of reasoning:

* easy   -- the percentage of values in a stated region: the stem gives mu, sigma and the region's end values in the
            units of the variable, so the learner first decides how many standard deviations each end is from the mean;
* medium -- a count of values expected in a sample (the percentage first, then the count); the standard deviation from
            the points of inflection of the curve, or from the interval holding 95% of the values; or which of two
            statements about two curves is true;
* hard   -- the mean or the standard deviation recovered from two tail percentages ("16% exceed A, 2.5% fall below B");
            and the share of a second normal population that falls in the interval holding the middle 95% of a first.

The exam chain (topic `exd_x_r3f_empirical_rule`) is (a) sigma from the points of inflection, (b) the percentage in a
stated asymmetric region, (c) the expected count in a large sample (follows part b), (d) the value exceeded by only
2.5% (follows part a).

The band shares are held as exact Fractions and cross-checked against the normal distribution itself: each
cumulative share must sit within 0.3 of the exact value (the rule is a rounded one), so a mis-typed band cannot survive.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3f_kit import build_until, dec, number_part, phi, single

#: cumulative share (per cent) at whole-number z, upper side of the mean
_CUM = {0: Fraction(50), 1: Fraction(84), 2: Fraction("97.5"), 3: Fraction("99.85")}

for _z, _c in _CUM.items():  # the rule against the exact distribution, to the rule's own precision
    check(abs(float(_c) - 100 * float(phi(_z))) < 0.3, f"empirical cumulative share at z = {_z} is wrong")


def cum(z: int | None) -> Fraction:
    """Per cent of values below mu + z sigma, by the rule; `None` is plus infinity."""
    if z is None:
        return Fraction(100)
    return _CUM[z] if z >= 0 else 100 - _CUM[-z]


def share(lo: int | None, hi: int | None) -> Fraction:
    """Per cent of values between mu + lo sigma and mu + hi sigma; `None` is no bound on that side."""
    return cum(hi) - (Fraction(0) if lo is None else cum(lo))


def central(k: int) -> Fraction:
    return share(-k, k)


def places_for(p: Fraction) -> int:
    return 1 if (p * 10).denominator == 1 else 2


#: (what is measured, as a subject, unit, mu choices, sigma choices, plural noun for a sample)
CONTEXTS = (
    ("The lengths of bolts made by a machine are normally distributed", "mm", range(40, 91, 5), (2, 3, 4, 5), "bolts"),
    (
        "The time Year 12 students take to complete a practice paper is normally distributed",
        "minutes",
        range(90, 151, 5),
        (4, 5, 6, 8),
        "students",
    ),
    (
        "Birth masses of babies at a regional hospital are normally distributed",
        "g",
        range(3100, 3601, 100),
        (300, 400, 500),
        "babies",
    ),
    (
        "The daily maximum temperature in Broome in July is normally distributed",
        "degrees Celsius",
        range(28, 34),
        (2, 3),
        "days",
    ),
    (
        "Scores on a state-wide reading test are normally distributed",
        "marks",
        range(480, 541, 10),
        (20, 30, 40, 50),
        "students",
    ),
    (
        "Packets of rice from a mill have masses that are normally distributed",
        "g",
        range(480, 521, 5),
        (4, 5, 6, 8),
        "packets",
    ),
)

EASY_REGIONS = (
    (None, -1),
    (None, -2),
    (-3, None),
    (2, None),
    (1, None),
    (3, None),
    (0, 1),
    (0, 2),
    (-2, 0),
    (-1, 0),
    (-1, 1),
    (-2, 2),
    (-3, 3),
    (-3, 0),
    (None, 1),
    (None, 2),
    (-1, None),
    (-2, None),
)
MEDIUM_REGIONS = ((-1, 2), (-2, 1), (-2, 3), (-3, 2), (-1, 3), (1, 3), (-3, -1), (-3, 1), (1, 2), (-2, -1), (0, 3))


def _setting(rng):
    """(subject clause, unit, mu, sigma, plural noun, full sentence, sentence with the mean only)."""
    base, unit, mus, sigmas, plural = rng.choice(CONTEXTS)
    mu, s = rng.choice(list(mus)), rng.choice(sigmas)
    full = f"{base} with mean {mu} {unit} and standard deviation {s} {unit}."
    mean_only = f"{base} with mean {mu} {unit}."
    return base, unit, mu, s, plural, full, mean_only


def _bound(end: int | None, mu: int, s: int) -> int | None:
    return None if end is None else mu + end * s


def _region_text(lo, hi, mu, s) -> str:
    a, b = _bound(lo, mu, s), _bound(hi, mu, s)
    if lo is None:
        return f"below {b}"
    if hi is None:
        return f"above {a}"
    return f"between {a} and {b}"


def _outer_band(lo, hi) -> Fraction | None:
    """The share of the next band outside a finite region, or None when the region already reaches 3 sigma."""
    if hi is not None and hi < 3:
        return cum(hi + 1) - cum(hi)
    if lo is not None and lo > -3:
        return cum(lo) - cum(lo - 1)
    return None


def _region_slips(lo, hi, p):
    """Wrong percentages a learner reaches by a named slip, in order of likelihood."""
    slips = [(100 - p, "r3f-empirical-gives-complement")]
    if lo is not None and hi is not None:
        if lo != -hi:
            slips.append((central(max(abs(lo), abs(hi))), "r3f-empirical-treats-interval-as-symmetric"))
        band = _outer_band(lo, hi)
        slips.append((None if band is None else p + band, "r3f-empirical-band-too-many"))
        slips.append((p / 2, "r3f-empirical-halves-the-middle"))
    else:
        k = abs(hi if lo is None else lo)
        slips.append((100 - central(k) if k else None, "r3f-empirical-tail-not-halved"))
        slips.append((central(k) / 2 if k else None, "r3f-empirical-tail-as-half-of-the-middle"))
    return slips


def _dp(places: int) -> str:
    return f"{places} decimal place{'s' if places > 1 else ''}"


def _shown(end: int | None) -> str:
    return "unbounded" if end is None else f"{end}"


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    _base, _unit, mu, s, _plural, full, _mean_only = _setting(rng)
    lo, hi = rng.choice(EASY_REGIONS)
    p = share(lo, hi)
    where = _region_text(lo, hi, mu, s)
    places = places_for(p)
    part = number_part(
        f"Using the 68-95-99.7 rule, approximately what percentage of values are {where}? "
        f"Give the percentage to {_dp(places)}.",
        p,
        places,
        1,
        [
            f"The ends are {_shown(lo)} and {_shown(hi)} standard deviations from the mean.",
            f"Adding the band shares gives {dec(p)}%.",
        ],
        _region_slips(lo, hi, p),
        within=(0, 100),
    )
    return single(f"{full} {part['prompt']}", part)


# -- medium ----------------------------------------------------------------------------------------------------


def _count(rng: random.Random) -> dict:
    _base, _unit, mu, s, plural, full, _mean_only = _setting(rng)
    lo, hi = rng.choice(MEDIUM_REGIONS)
    p = share(lo, hi)
    n = rng.choice((2000, 4000, 8000, 10000, 20000))
    expected = Fraction(n) * p / 100
    check(expected.denominator == 1, "the expected count is not a whole number")
    where = _region_text(lo, hi, mu, s)
    band = _outer_band(lo, hi)
    part = number_part(
        f"In a sample of {n} {plural}, approximately how many are expected to have values {where}?",
        expected,
        0,
        2,
        [f"The percentage {where} is {dec(p)}%.", f"{n} x {dec(p)}/100 = {expected}."],
        [
            (Fraction(n) * (100 - p) / 100, "r3f-empirical-count-of-the-complement"),
            (
                Fraction(n) * central(max(abs(lo), abs(hi))) / 100 if lo != -hi else None,
                "r3f-empirical-treats-interval-as-symmetric",
            ),
            (p, "r3f-empirical-percentage-reported-as-count"),
            (None if band is None else Fraction(n) * (p + band) / 100, "r3f-empirical-band-too-many"),
        ],
    )
    return single(f"{full} {part['prompt']}", part)


def _inflection(rng: random.Random) -> dict:
    base, unit, mu, s, _plural, _full, _mean_only = _setting(rng)
    a, b = mu - s, mu + s
    if rng.random() < 0.65:
        part = number_part(
            f"The graph of its probability density function has points of inflection at {a} and {b}. "
            "Find the standard deviation.",
            s,
            0,
            1,
            [f"The points of inflection are one standard deviation either side of the mean, so 2 sigma = {b} - {a}."],
            [
                (b - a, "r3f-inflection-gap-taken-as-sd"),
                (mu, "r3f-inflection-midpoint-taken-as-sd"),
                (Fraction(b - a, 4), "r3f-inflection-gap-quartered"),
                (a, "r3f-inflection-lower-point-taken-as-sd"),
            ],
        )
    else:
        part = number_part(
            f"The graph of its probability density function has points of inflection at {a} and {b}. Find the mean.",
            mu,
            0,
            1,
            [f"The mean is midway between the points of inflection: ({a} + {b})/2 = {mu}."],
            [
                (s, "r3f-inflection-sd-reported-as-mean"),
                (b - a, "r3f-inflection-gap-taken-as-mean"),
                (b, "r3f-inflection-upper-point-taken-as-mean"),
                (a, "r3f-inflection-lower-point-taken-as-mean"),
            ],
        )
    return single(f"{base}. {part['prompt']}", part)


def _interval95(rng: random.Random) -> dict:
    base, unit, mu, s, _plural, _full, mean_only = _setting(rng)
    a, b = mu - 2 * s, mu + 2 * s
    part = number_part(
        f"About 95% of the values lie between {a} and {b}. Find the standard deviation.",
        s,
        0,
        2,
        [f"95% lie within 2 standard deviations of the mean, so 4 sigma = {b} - {a}."],
        [
            (Fraction(b - a, 2), "r3f-interval95-half-width-taken-as-sd"),
            (b - a, "r3f-interval95-width-taken-as-sd"),
            (Fraction(b - a, 6), "r3f-interval95-treated-as-99-7"),
            (Fraction(b - a, 8), "r3f-interval95-width-divided-by-eight"),
        ],
    )
    return single(f"{mean_only} {part['prompt']}", part)


def _compare(rng: random.Random) -> dict:
    ka = rng.choice((2, 3, 4, 5))
    factor = rng.choice((2, 3, Fraction(3, 2), Fraction(5, 2)) if ka % 2 == 0 else (2, 3))
    kb = ka * factor
    check(Fraction(kb).denominator == 1, "second spread is not a whole number")
    kb = int(kb)
    centre = rng.choice((100, 120, 150, 200))
    unit = 5
    stem = (
        f"Two normal curves have the same mean. Curve A has points of inflection at {centre - ka * unit} and "
        f"{centre + ka * unit}; curve B has points of inflection at {centre - kb * unit} and {centre + kb * unit}. "
        "The standard deviation of curve B is how many times the standard deviation of curve A?"
    )
    ratio = Fraction(kb, ka)
    part = number_part(
        stem,
        ratio,
        1,
        2,
        [
            f"The standard deviations are half the gaps: {ka * unit} for A and {kb * unit} for B, a ratio of {dec(ratio)}."
        ],
        [
            (1 / ratio, "r3f-compare-ratio-inverted"),
            (ratio * ratio, "r3f-compare-ratio-of-variances"),
            (Fraction(kb - ka, ka), "r3f-compare-extra-taken-as-ratio"),
            (Fraction(kb * unit - ka * unit), "r3f-compare-difference-taken-as-ratio"),
        ],
    )
    return single(part["prompt"], part)


def _medium(rng: random.Random) -> dict:
    return rng.choice((_count, _inflection, _interval95, _compare))(rng)


# -- hard ------------------------------------------------------------------------------------------------------

_TAIL = {1: Fraction(16), 2: Fraction("2.5"), 3: Fraction("0.15")}


def _two_tails(rng: random.Random) -> dict:
    base, unit, mu, s, _plural, _full, _mean_only = _setting(rng)
    k1, k2 = rng.choice((1, 2)), rng.choice((1, 2, 3))
    upper, lower = mu + k1 * s, mu - k2 * s
    stem = f"{base}. {dec(_TAIL[k1])}% of the values exceed {upper} and {dec(_TAIL[k2])}% fall below {lower}."
    if rng.random() < 0.5:
        part = number_part(
            f"{stem} Find the standard deviation.",
            s,
            0,
            3,
            [
                f"{upper} is {k1} sigma above the mean and {lower} is {k2} sigma below it.",
                f"{k1 + k2} sigma = {upper} - {lower}.",
            ],
            [
                (upper - lower, "r3f-two-tails-gap-taken-as-sd"),
                (Fraction(upper - lower, 2), "r3f-two-tails-gap-halved"),
                (Fraction(upper - lower, k1 + k2 + 1), "r3f-two-tails-miscounts-sds"),
                (Fraction(upper - lower, 4), "r3f-two-tails-gap-quartered"),
            ],
        )
    else:
        part = number_part(
            f"{stem} Find the mean.",
            mu,
            0,
            3,
            [
                f"{upper} is {k1} sigma above the mean and {lower} is {k2} sigma below it, so sigma = {s}.",
                f"mean = {upper} - {k1} x {s}.",
            ],
            [
                (Fraction(upper + lower, 2), "r3f-two-tails-midpoint-taken-as-mean"),
                (upper - s, "r3f-two-tails-one-sd-assumed"),
                (lower + s, "r3f-two-tails-one-sd-assumed"),
                (upper - (k1 + k2) * s, "r3f-two-tails-wrong-multiple"),
            ],
        )
    return single(part["prompt"], part)


def _second_population(rng: random.Random) -> dict:
    for _ in attempts(500, "a second population with whole-number z limits"):
        mu_a, s_a = rng.choice(range(40, 81, 5)), rng.choice((2, 3, 4, 5))
        lo, hi = mu_a - 2 * s_a, mu_a + 2 * s_a
        s_b = rng.choice((s_a, 2 * s_a, 3 * s_a))
        zlo = rng.choice((-3, -2, -1, 0))
        mu_b = lo - zlo * s_b
        zhi = Fraction(hi - mu_b, s_b)
        if zhi.denominator == 1 and zhi > zlo and zhi in (0, 1, 2, 3) and (mu_b, s_b) != (mu_a, s_a):
            break
    zhi = int(zhi)
    p = share(zlo, zhi)
    places = places_for(p)
    stem = (
        f"Population A is normally distributed with mean {mu_a} and standard deviation {s_a}. Population B is "
        f"normally distributed with mean {mu_b} and standard deviation {s_b}. The middle 95% of population A lies "
        f"between {lo} and {hi}. What percentage of population B lies in that same interval? "
        f"Give the percentage to {_dp(places)}."
    )
    part = number_part(
        stem,
        p,
        places,
        3,
        [f"For B the ends are z = {zlo} and z = {zhi}.", f"The share between them is {dec(p)}%."],
        [
            (Fraction(95), "r3f-second-population-reports-95"),
            (central(max(abs(zlo), abs(zhi))), "r3f-empirical-treats-interval-as-symmetric"),
            (100 - p, "r3f-empirical-gives-complement"),
            (_outer_band(zlo, zhi) and p + _outer_band(zlo, zhi), "r3f-empirical-band-too-many"),
        ],
        within=(0, 100),
    )
    return single(part["prompt"], part)


def _hard(rng: random.Random) -> dict:
    return rng.choice((_two_tails, _second_population))(rng)


@generator(
    "r3f_empirical_rule",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3f_empirical_rule(rng: random.Random, tier: str = "easy") -> dict:
    """Empirical rule: a region's percentage (easy), counts and curve features (medium), recovered parameters (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    base, unit, mu, s, plural, _full, mean_only = _setting(rng)
    lo, hi = rng.choice(((-1, 2), (-2, 1), (-2, 3), (-1, 3), (-3, 2), (1, 3)))
    p = share(lo, hi)
    n = rng.choice((2000, 4000, 10000, 20000))
    count = Fraction(n) * p / 100
    check(count.denominator == 1, "exam count is not a whole number")
    a_in, b_in = mu - s, mu + s
    cutoff = mu + 2 * s
    places = places_for(p)
    where = _region_text(lo, hi, mu, s)
    band = _outer_band(lo, hi)
    parts = [
        number_part(
            f"The graph of the probability density function has points of inflection at {a_in} and {b_in}. "
            "Find the standard deviation.",
            s,
            0,
            1,
            [
                f"The points of inflection are one standard deviation either side of the mean: 2 sigma = {b_in} - {a_in}."
            ],
            [
                (b_in - a_in, "r3f-inflection-gap-taken-as-sd"),
                (mu, "r3f-inflection-midpoint-taken-as-sd"),
                (Fraction(b_in - a_in, 4), "r3f-inflection-gap-quartered"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Using the 68-95-99.7 rule, approximately what percentage of values are {where}? "
            f"Give the percentage to {_dp(places)}.",
            p,
            places,
            2,
            [f"The ends are {lo} and {hi} standard deviations from the mean.", f"The share between them is {dec(p)}%."],
            _region_slips(lo, hi, p),
            sensible_range=(0, 100),
            within=(0, 100),
        ),
        number_part(
            f"In a sample of {n} {plural}, approximately how many are expected to have values {where}?",
            count,
            0,
            1,
            [f"{n} x {dec(p)}/100 = {count}."],
            [
                (Fraction(n) * (100 - p) / 100, "r3f-empirical-count-of-the-complement"),
                (p, "r3f-empirical-percentage-reported-as-count"),
                (None if band is None else Fraction(n) * (p + band) / 100, "r3f-empirical-band-too-many"),
                (Fraction(n) * p / 200, "r3f-empirical-halves-the-middle"),
            ],
            follow_through=f"part_b*{n}/100",
            sensible_range=(0, n),
        ),
        number_part(
            "Only 2.5% of the values exceed a certain value. Find that value.",
            cutoff,
            0,
            2,
            [f"2.5% lie above 2 standard deviations over the mean: {mu} + 2 x {s} = {cutoff}."],
            [
                (mu + s, "r3f-cutoff-one-sd-above-mean"),
                (mu + 3 * s, "r3f-cutoff-three-sd-above-mean"),
                (mu + 2 * (b_in - a_in), "r3f-cutoff-uses-the-gap-as-sd"),
            ],
            follow_through=f"{mu}+2*part_a",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=mean_only, parts=parts)


@generator("exd_x_r3f_empirical_rule", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3f_empirical_rule(rng: random.Random, tier: str = "exam") -> dict:
    """Sigma from the inflection points, a region's percentage, the count in a sample, then the 2.5% cut-off."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
