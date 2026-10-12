"""Generates `exd_x_r3f_permutation_roles` questions: an exam chain on filling distinct roles from a group.

A group of n members, s of whom belong to a named subgroup, fills r different roles, each held by a different
member. The chain:

(a) the number of ways to fill the roles                       number,   1 mark    P(n, r)
(b) the number of ways in which no subgroup member has a role  number,   2 marks   P(n - s, r)
(c) the number of ways in which at least one does              number,   1 mark    part_a - part_b
(d) the probability of that, for roles filled at random        fraction, 2 marks   part_c / part_a

Parts (c) and (d) follow through, so a learner whose count in (a) or (b) is wrong still earns the later parts
for using it correctly. Every count is checked against `math.perm` and, for the complement, by enumerating the
roles one at a time.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_r3f_parts import fraction_part, number_part
from ._exd_r3_distinct import distinct_part_answers

_ROLE_POOLS = (
    (
        "A school council has {n} members, {s} of whom are {plural}.",
        ("president", "secretary", "treasurer", "social convenor", "publicity officer"),
        ("Year 12 students", "Year 12 student"),
    ),
    (
        "A surf life saving club has {n} members, {s} of whom are {plural}.",
        ("captain", "vice-captain", "secretary", "treasurer", "equipment officer"),
        ("senior patrol members", "senior patrol member"),
    ),
    (
        "A historical society has {n} members, {s} of whom are {plural}.",
        ("president", "secretary", "treasurer", "archivist", "newsletter editor"),
        ("founding members", "founding member"),
    ),
)


def _listed(names) -> str:
    return ", ".join(names[:-1]) + " and " + names[-1]


@generator("exd_x_r3f_permutation_roles", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_permutation_roles(rng: random.Random, tier: str = "exam") -> dict:
    """Ordered selections, the complement count, 'at least one', and the matching probability."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    template, roles, (plural, singular) = rng.choice(_ROLE_POOLS)
    for _ in attempts(1000, "a group that can fill the roles without the subgroup"):
        r = rng.randint(3, 5)
        n = rng.randint(8, 20)
        s = rng.randint(2, 6)
        if n - s >= r + 1 and s < n - s:
            break
    total = math.perm(n, r)
    none = math.perm(n - s, r)
    at_least = total - none

    # Independent route: build the complement role by role (each role has one fewer eligible member than the last).
    walk = 1
    for i in range(r):
        walk *= (n - s) - i
    check(walk == none and none < total, "the complement count disagrees")
    probability = Fraction(at_least, total)

    total_candidates = [
        (math.comb(n, r), "permutation-replaced-by-combination"),
        (n**r, "repeats-allowed-used-for-no-repeats"),
        (math.perm(n, r - 1), "one-factor-short"),
        (math.perm(n - 1, r), "removed-one-member-too-few"),
    ]
    none_candidates = [
        (math.comb(n - s, r), "permutation-replaced-by-combination"),
        ((n - s) ** r, "repeats-allowed-used-for-no-repeats"),
        (math.perm(n - s, r - 1), "one-factor-short"),
        (math.perm(n - 1, r), "removed-one-member-too-few"),
    ]
    at_least_candidates = [
        (none, "complement-reported"),
        (total + none, "complement-added-instead-of-subtracted"),
        (s * math.perm(n - 1, r - 1), "at-least-one-counted-for-first-role-only"),
        (s * r * math.perm(n - 1, r - 1), "at-least-one-overcounted-by-placing-one-then-filling-freely"),
    ]
    probability_candidates = [
        (Fraction(none, total), "complement-reported"),
        (Fraction(at_least, n**r), "probability-denominator-allows-repeats"),
        (Fraction(s, n), "subgroup-share-reported-as-probability"),
        (Fraction(s * math.perm(n - 1, r - 1), total), "at-least-one-counted-for-first-role-only"),
    ]

    stem = (
        template.format(n=n, s=s, plural=plural)
        + f" The roles of {_listed(roles[:r])} are each filled by a different member."
    )
    return dict(
        text=stem,
        parts=[
            number_part(
                "In how many ways can the roles be filled?",
                total,
                total_candidates,
                marks=1,
                sensible_range=(0, None),
                working=[f"{n} \u00d7 " + " \u00d7 ".join(str(n - i) for i in range(1, r)) + f" = {total}."],
            ),
            number_part(
                f"In how many of these ways does no {singular} hold a role?",
                none,
                none_candidates,
                marks=2,
                sensible_range=(0, None),
                working=[f"The roles are filled from the other {n - s} members: {none} ways."],
            ),
            number_part(
                f"Hence find the number of ways in which at least one {singular} holds a role.",
                at_least,
                at_least_candidates,
                marks=1,
                sensible_range=(0, None),
                follow_through="part_a - part_b",
                working=[f"{total} - {none} = {at_least}."],
            ),
            fraction_part(
                f"The roles are filled at random. Find the probability that at least one {singular} holds a role.",
                probability,
                probability_candidates,
                marks=2,
                sensible_range=(0, 1),
                follow_through="part_c/part_a",
                working=[f"{at_least}/{total} = {probability.numerator}/{probability.denominator}."],
            ),
        ],
    )
