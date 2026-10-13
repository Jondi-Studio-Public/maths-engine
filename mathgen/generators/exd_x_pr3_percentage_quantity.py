"""Exam source for a percentage of a quantity: a percentage of a group, then a percentage of what is left.

A cohort of students, a percentage who walk, and a percentage of the rest who catch the bus. Part (a) finds the
walkers, part (b) the others, part (c) the bus riders as a percentage of that smaller group, and part (d) writes the
bus riders as a percentage of the whole cohort. Parts (b), (c) and (d) carry `follow_through`. The cohort size is
drawn so every count is a whole number and the final percentage is a whole number.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import leaks, part

# (opening clause, group noun, what the first group does, its negation, what the second group does, the rest)
_COHORTS = [
    (
        "A school in Darwin has {N} students",
        "students",
        "walk to school",
        "do not walk to school",
        "catch the bus",
        "are driven",
    ),
    (
        "A surf life saving club on the Sunshine Coast has {N} junior members",
        "members",
        "arrive by bike",
        "do not arrive by bike",
        "come by bus",
        "are driven",
    ),
    (
        "A university campus in Perth surveys {N} first-year students",
        "students",
        "live on campus",
        "do not live on campus",
        "catch the train",
        "drive",
    ),
    (
        "A football club in Geelong has {N} members",
        "members",
        "live within walking distance",
        "do not live within walking distance",
        "catch the bus",
        "drive",
    ),
]
_P1 = (10, 20, 25, 30, 40, 60, 75)
_P2 = (20, 25, 40, 50, 60, 75)


@generator("exd_x_pr3_percentage_quantity", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_percentage_quantity(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(4000, "a cohort with whole-number counts"):
        p1, p2 = rng.choice(_P1), rng.choice(_P2)
        N = 20 * rng.randint(8, 60)
        walk = Fraction(N * p1, 100)
        rest = N - walk
        bus = rest * p2 / 100
        pct = Fraction(bus * 100, N)
        if any(v.denominator != 1 for v in (walk, rest, bus, pct)) or pct in (p1, p2):
            continue
        walk, rest, bus, pct = int(walk), int(rest), int(bus), int(pct)
        lead, noun, walkers, not_walkers, riders, others = rng.choice(_COHORTS)
        stem = f"{lead.format(N=N)}. {p1}% of them {walkers}. Of the others, {p2}% {riders} and the rest {others}. Give every answer as a whole number."
        if not leaks(stem, [walk, rest, bus, pct]):
            break
    check(walk + rest == N and bus * 100 == rest * p2 and bus * 100 == pct * N, "cohort chain disagrees")
    parts = [
        part(
            f"How many of the {N} {noun} {walkers}?",
            walk,
            marks=1,
            working=[f"{p1}% of {N} = {walk}."],
            candidates=[
                (N - walk, "gives-remainder-not-part"),
                (N * p1, "multiplies-by-whole-percent"),
                (Fraction(N, p1), "divides-by-the-percent"),
                (p1, "percent-as-absolute"),
                (N - p1, "subtracts-percentage"),
            ],
        ),
        part(
            f"Using your answer to (a), how many of the {N} {noun} {not_walkers}?",
            rest,
            marks=1,
            follow_through=f"{N}-part_a",
            working=[f"{N} − {walk} = {rest}."],
            candidates=[
                (walk, "gave-part-not-remainder"),
                (N - p1, "subtracts-percentage"),
                (N + walk, "adds-instead-of-subtracts"),
            ],
        ),
        part(
            f"Using your answer to (b), how many of the {N} {noun} {riders}?",
            bus,
            marks=2,
            follow_through=f"part_b*{p2}/100",
            working=[f"{p2}% of {rest} = {bus}."],
            candidates=[
                (Fraction(N * p2, 100), "percent-of-wrong-group"),
                (Fraction(walk * p2, 100), "percent-of-wrong-group"),
                (rest - p2, "percent-as-absolute"),
            ],
        ),
        part(
            f"What percentage of all {N} {noun} {riders}? Use your answer to (c).",
            pct,
            marks=2,
            follow_through=f"part_c*100/{N}",
            sensible_range=(0, 100),
            working=[f"{bus} ÷ {N} × 100 = {pct}%."],
            candidates=[
                (p2, "percent-of-wrong-group"),
                (bus, "count-given-as-percent"),
                (p1 + p2, "adds-percentages-of-different-groups"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
