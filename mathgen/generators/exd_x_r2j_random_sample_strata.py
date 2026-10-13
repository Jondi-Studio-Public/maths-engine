"""Exam chain for judging and building a random sample: a proportional stratified sample, worked through.

A population is split into three strata of known size and a stratified random sample of stated size is to be
drawn, each stratum represented in proportion to its size. Four parts, each using the one before it:

(a) the sampling fraction n / N, as a fraction in simplest form;
(b) the number to select from the first stratum (the fraction times the stratum size);
(c) the number to select from the second stratum;
(d) the number to select from the third stratum, found by taking the first two away from the sample size.

Built backwards: the sampling fraction c/d (in lowest terms) and the stratum multipliers m1, m2, m3 are drawn,
the strata have d*m_i members and the sample takes c*m_i from each, so every part is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd

from mathgen.core import check, generator, phrase

from .exd_x_r2j_support import fraction_part, number_part

_FRACTIONS = [(1, 6), (1, 8), (1, 10), (1, 12), (1, 15), (1, 20), (2, 9), (2, 15), (3, 16), (3, 20), (3, 25), (4, 25)]

#: (three strata, who is sampled, population sentence)
_SETTINGS = [
    (
        ("the Years 7 and 8 group", "the Years 9 and 10 group", "the Years 11 and 12 group"),
        "students",
        "{place} has {total} students: {n1} in {s1}, {n2} in {s2} and {n3} in {s3}.",
    ),
    (
        ("the under-30 group", "the 30-to-54 group", "the 55-and-over group"),
        "members",
        "{place} has {total} members: {n1} in {s1}, {n2} in {s2} and {n3} in {s3}.",
    ),
    (
        ("the warehouse", "the call centre", "the head office"),
        "employees",
        "{place} employs {total} people: {n1} at {s1}, {n2} at {s2} and {n3} at {s3}.",
    ),
    (
        ("the northern suburbs", "the central suburbs", "the southern suburbs"),
        "residents",
        "{place} has {total} registered voters: {n1} in {s1}, {n2} in {s2} and {n3} in {s3}.",
    ),
]

_PLACES = {
    "students": [
        "Banksia Secondary College",
        "Mount Pleasant High School",
        "Riverside Grammar",
        "Kestrel Senior College",
    ],
    "members": ["The Wattle Park Sports Club", "The Coastal Swimming Club", "Eastside Community Gym"],
    "employees": ["Southern Cross Freight", "Murray Valley Foods", "Gum Tree Logistics"],
    "residents": ["The Shire of Marlow", "The City of Hartley", "Tarrawarra Council"],
}

_STEM = [
    "{setting} A stratified random sample of {n} {who} is to be taken, with each group represented in proportion to "
    "its size.",
    "{setting} A survey will use a stratified random sample of {n} {who}, with the number taken from each group in "
    "proportion to the size of the group.",
]


def _build(rng: random.Random) -> dict:
    c, d = rng.choice(_FRACTIONS)
    check(gcd(c, d) == 1, "sampling fraction not in lowest terms")
    ms = sorted(rng.sample(range(5, 46), 3))
    rng.shuffle(ms)
    sizes = [d * m for m in ms]
    takes = [c * m for m in ms]
    total, n = sum(sizes), sum(takes)
    check(Fraction(n, total) == Fraction(c, d), "sampling fraction mismatch")

    strata, who, template = rng.choice(_SETTINGS)
    place = rng.choice(_PLACES[who])
    setting = template.format(
        place=place,
        total=f"{total:,}",
        n1=f"{sizes[0]:,}",
        n2=f"{sizes[1]:,}",
        n3=f"{sizes[2]:,}",
        s1=strata[0],
        s2=strata[1],
        s3=strata[2],
    )
    stem = phrase(rng, _STEM, setting=setting, n=n, who=who)

    frac = Fraction(n, total)
    n_of_stratum = [frac * s for s in sizes]
    check([int(v) for v in n_of_stratum] == takes and all(v.denominator == 1 for v in n_of_stratum), "allocation")
    check(takes[2] == n - takes[0] - takes[1], "third stratum by subtraction disagrees")

    a_wrong = [
        (Fraction(total, n), "inverted-ratio"),
        (Fraction(n, sizes[0]), "sampling-fraction-from-one-stratum"),
        (Fraction(n, total - n), "part-to-part-instead-of-part-to-whole"),
    ]
    part_a = fraction_part(
        prompt="Find the sampling fraction, the sample size divided by the population size, as a fraction in "
        "simplest form.",
        exact=frac,
        marks=1,
        working=[f"Sampling fraction = {n} / {total:,} = {c}/{d} in simplest form."],
        wrong=a_wrong,
        sensible_range=(0, 1),
    )

    def alloc_part(i: int, ft: str) -> dict:
        equal_share = Fraction(n, 3)
        return number_part(
            prompt=f"How many {who} should be selected from {strata[i]}?",
            exact=takes[i],
            places=0,
            marks=2 if i == 0 else 1,
            working=[f"{c}/{d} x {sizes[i]:,} = {takes[i]}."],
            wrong=[
                (round(equal_share), "equal-allocation-ignores-size"),
                (Fraction(sizes[i] * d, c), "inverted-ratio"),
                (Fraction(sizes[i] * (d - c), d), "complement-fraction-taken"),
            ],
            follow_through=ft,
            sensible_range=(0, None),
        )

    part_b = alloc_part(0, f"part_a*{sizes[0]}")
    part_c = alloc_part(1, f"part_a*{sizes[1]}")
    part_d = number_part(
        prompt=f"Hence find how many {who} should be selected from {strata[2]}, so that the sample has {n} {who} in "
        "all.",
        exact=takes[2],
        places=0,
        marks=2,
        working=[f"{n} - {takes[0]} - {takes[1]} = {takes[2]}.", f"Check: {c}/{d} x {sizes[2]:,} = {takes[2]}."],
        wrong=[
            (n - takes[0], "leftover-ignores-a-stratum"),
            (n - takes[1], "leftover-ignores-a-stratum"),
            (round(Fraction(n, 3)), "equal-allocation-ignores-size"),
            (takes[0] + takes[1], "leftover-ignores-a-stratum"),
        ],
        follow_through=f"{n} - part_b - part_c",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2j_random_sample_strata", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2j_random_sample_strata(rng: random.Random, tier: str = "exam") -> dict:
    """Sampling fraction, then the number drawn from each of three strata of a proportional stratified sample."""
    check(tier == "exam", "exd_x_r2j_random_sample_strata: only the exam tier exists")
    return _build(rng)
