"""Exam-tier source for `ucv1-inclusion-exclusion-two-sets`: a two-set survey that asks for "exactly one".

A group of N people is surveyed about two activities. The stem gives the size of each activity group and the
number who do neither. The learner has to choose the method:

* (a) the number who do both: |A ∩ B| = |A| + |B| - (N - neither);
* (b) the number who do exactly one of the two, |A| + |B| - 2|A ∩ B|, followed through from (a);
* (c) the probability that a person chosen at random does exactly one, as a fraction of the whole group,
  followed through from (b).

Built backwards from the four region counts (only A, only B, both, neither), so every key is exact. The draw
keeps the overlap and every region non-empty and keeps the wrong routes (forgetting the people in neither,
counting the union where "exactly one" was asked) distinct from the key.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import check, generator, pick_distractors, rational_text

# (group noun, activity A, activity B, place phrase)
_CONTEXTS = [
    ("students", "play netball", "play AFL football", "at a school in Geelong"),
    ("students", "take Japanese", "take Indonesian", "in Year 9 at a school in Perth"),
    (
        "members",
        "swim in the club carnival",
        "run in the club carnival",
        "of a surf life saving club on the Gold Coast",
    ),
    ("students", "play in the school band", "sing in the school choir", "at a school in Hobart"),
    ("people", "ride a bike to work", "catch the train to work", "surveyed in Adelaide"),
    ("students", "study Physics", "study Chemistry", "in Year 12 at a school in Darwin"),
]
_STEMS = [
    "Of the {n} {noun} {place}, {a} {act_a} and {b} {act_b}. {z} do neither.",
    "A survey of {n} {noun} {place} found that {a} {act_a}, {b} {act_b} and {z} do neither.",
]
_A_PROMPTS = [
    "How many {noun} {act_a} and {act_b}?",
    "Find the number of {noun} who both {act_a} and {act_b}.",
]
_B_PROMPTS = [
    "How many {noun} do exactly one of the two activities?",
    "Find the number of {noun} who do one of the activities but not both.",
]
_C_PROMPTS = [
    "One of the {n} {noun} is chosen at random. Find the probability that this person does exactly one of the "
    "activities, as a fraction in simplest form.",
    "A person is chosen at random from the {n} {noun}. What is the probability that the person does exactly one "
    "of the two activities? Give a fraction in simplest form.",
]


def _draw(rng: random.Random):
    for _ in range(1000):
        only_a, only_b, both, neither = (rng.randint(lo, hi) for lo, hi in ((6, 40), (6, 40), (4, 25), (4, 30)))
        n = only_a + only_b + both + neither
        a, b = only_a + both, only_b + both
        exactly_one = only_a + only_b
        slips = {
            "ignored_neither": a + b - n,
            "union": n - neither,
            "sum": a + b,
        }
        wrong_b = {a + b - both, a + b, n - neither, a - both}
        if both in slips.values() or exactly_one in wrong_b or a + b - n < 1 or len({a, b, both}) < 3:
            continue
        if Fraction(exactly_one, n) == Fraction(n - neither, n):
            continue
        return n, a, b, neither, both, exactly_one
    raise RuntimeError("no suitable survey counts in 1000 draws")


@generator("exd_x_u1_two_set_survey", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u1_two_set_survey(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: overlap, "exactly one" and its probability from a two-set survey with a neither group."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n, a, b, neither, both, exactly_one = _draw(rng)
    noun, act_a, act_b, place = rng.choice(_CONTEXTS)

    # Independent route: enumerate the people by region and count them.
    people = ["both"] * both + ["a only"] * (a - both) + ["b only"] * (b - both) + ["neither"] * neither
    check(len(people) == n, "regions do not add to the total")
    check(sum(1 for p in people if p in ("a only", "b only")) == exactly_one, "exactly-one count disagrees")
    check(sum(1 for p in people if p in ("a only", "both")) == a, "set A size disagrees with regions")
    check(a + b - (n - neither) == both, "overlap disagrees with inclusion-exclusion")
    probability = Fraction(exactly_one, n)

    def num(value, tag):
        return Distractor(str(value), tag, value)

    a_cands = [
        num(a + b - n, "ignores-neither"),
        num(n - neither, "reports-union-for-overlap"),
        num(a + b, "forgets-overlap"),
        num(n - neither - a, None),
    ]
    b_cands = [
        num(a + b - both, "at-least-one-for-exactly-one"),
        num(a + b, "forgets-overlap"),
        num(a - both, "only-one-set-counted"),
        num(n - neither - both, None),
    ]
    c_cands = [
        Distractor(
            rational_text(Fraction(a + b - both, n)),
            "at-least-one-for-exactly-one",
            exact_value(Fraction(a + b - both, n)),
        ),
        Distractor(
            rational_text(Fraction(exactly_one, n - neither)),
            "wrong-sample-space",
            exact_value(Fraction(exactly_one, n - neither)),
        ),
        Distractor(rational_text(Fraction(both, n)), None, exact_value(Fraction(both, n))),
        Distractor(rational_text(Fraction(a + b, n)), "forgets-overlap", exact_value(Fraction(a + b, n))),
    ]
    a_dist = pick_distractors(str(both), a_cands, 3)
    b_dist = pick_distractors(str(exactly_one), b_cands, 3)
    c_dist = pick_distractors(rational_text(probability), c_cands, 3)
    check(len(a_dist) >= 2 and len(b_dist) >= 2 and len(c_dist) >= 2, "too few distractors")

    fmt = dict(n=n, noun=noun, act_a=act_a, act_b=act_b, place=place, a=a, b=b, z=neither)
    return dict(
        text=rng.choice(_STEMS).format(**fmt),
        parts=[
            dict(
                prompt=rng.choice(_A_PROMPTS).format(**fmt),
                answer_kind="number",
                answer=both,
                answer_text=str(both),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"At least one activity: {n} - {neither} = {n - neither}.",
                    f"Both: {a} + {b} - {n - neither} = {both}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=rng.choice(_B_PROMPTS).format(**fmt),
                answer_kind="number",
                answer=exactly_one,
                answer_text=str(exactly_one),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{a}+{b}-2*part_a",
                working=[f"Exactly one: ({a} - {both}) + ({b} - {both}) = {exactly_one}."],
                distractors=b_dist,
            ),
            dict(
                prompt=rng.choice(_C_PROMPTS).format(**fmt),
                answer_kind="fraction",
                answer=exact_value(probability),
                answer_text=rational_text(probability),
                marks=1,
                sensible_range=(0, 1),
                follow_through=f"part_b/{n}",
                working=[f"P(exactly one) = {exactly_one}/{n} = {rational_text(probability)}."],
                distractors=c_dist,
            ),
        ],
    )
