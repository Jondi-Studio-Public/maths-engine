"""Exam-tier source for `bernoulli-mle`: estimate p from a recorded sequence, then estimate functions of p.

The data are a string of outcomes, not a count, so the learner counts first. Then

1. the maximum likelihood estimate of p (a simplified fraction),
2. the estimate of the probability that the next two independent trials both succeed, which is the estimate of
   p squared (the invariance of the maximum likelihood estimate under a function of the parameter), and
3. the estimate of the odds of success, p / (1 - p).

Every outcome string is built backwards from a chosen number of trials and successes, then shuffled.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import exact_value
from ..core import attempts, check, generator, latex_answer, rational_text
from .exd_x_u6_support import keys_clear, tagged, within_unit_interval
from ..core import pick_distractors

# intro, success letter, failure letter, what a success is called, trial noun
_CONTEXTS = [
    (
        "A netballer's shots in training are recorded below (G = goal, M = miss), each treated as an independent trial "
        "with the same chance of a goal.",
        "G",
        "M",
        "goal",
        "shot",
    ),
    (
        "A gardener in Hobart sowed seeds one at a time and recorded whether each germinated (Y = germinated, N = did "
        "not), treating the seeds as independent with a common chance of germinating.",
        "Y",
        "N",
        "germination",
        "seed",
    ),
    (
        "A factory in Newcastle tests batteries one at a time (P = pass, F = fail). Treat the tests as independent "
        "trials with the same chance of a pass.",
        "P",
        "F",
        "pass",
        "test",
    ),
]


def _build(rng: random.Random):
    n = rng.randint(12, 30)
    k = rng.randint(3, n - 3)
    if 2 * k == n:
        return None
    p_hat = Fraction(k, n)
    sq = p_hat**2
    odds = Fraction(k, n - k)
    intro, win, lose, event, noun = rng.choice(_CONTEXTS)
    seq = [win] * k + [lose] * (n - k)
    rng.shuffle(seq)
    stem = f"{intro} Outcomes in order: {' '.join(seq)}."
    p1 = f"Find the maximum likelihood estimate of the probability of a {event}, as a simplified fraction."
    p2 = f"Using your estimate, find the estimated probability that the next two {noun}s both result in a {event}."
    p3 = f"Using your estimate from (a), find the estimated odds of a {event} (the probability of a {event} divided by the probability of no {event})."
    if not keys_clear(" ".join([stem, p1, p2, p3]), [p_hat, sq, odds]):
        return None

    a_d = pick_distractors(
        rational_text(p_hat),
        within_unit_interval(
            [
                tagged(rational_text(1 - p_hat), "counts-failures-for-success-rate", 1 - p_hat),
                tagged(rational_text(Fraction(k, n + 1)), "off-by-one-denominator", Fraction(k, n + 1)),
                tagged(rational_text(Fraction(k, n - 1)), "off-by-one-denominator", Fraction(k, n - 1)),
            ]
        ),
        3,
    )
    b_d = pick_distractors(
        rational_text(sq),
        within_unit_interval(
            [
                tagged(rational_text(p_hat), "estimate-not-squared", p_hat),
                tagged(rational_text(2 * p_hat), "squared-as-doubled", 2 * p_hat),
                tagged(rational_text((1 - p_hat) ** 2), "gave-complement-probability", (1 - p_hat) ** 2),
            ]
        ),
        3,
    )
    c_d = pick_distractors(
        rational_text(odds),
        [
            tagged(rational_text(p_hat), "odds-given-as-probability", p_hat),
            tagged(rational_text(1 / odds), "odds-inverted", 1 / odds),
            tagged(rational_text(Fraction(k, n - k + 1)), "off-by-one-denominator", Fraction(k, n - k + 1)),
        ],
        3,
    )
    if min(len(a_d), len(b_d), len(c_d)) < 2:
        return None

    def part(prompt, value, marks, distractors, working, sensible, follow_through=None):
        entry = dict(
            prompt=prompt,
            answer_kind="fraction",
            answer=exact_value(value),
            answer_text=rational_text(value),
            answer_latex=latex_answer(rational_text(value)),
            marks=marks,
            sensible_range=sensible,
            working=working,
            distractors=distractors,
        )
        if follow_through:
            entry["follow_through"] = follow_through
        return entry

    return dict(
        text=stem,
        parts=[
            part(
                p1,
                p_hat,
                2,
                a_d,
                [f"Count the {win}s: {k} out of {n} trials.", f"p-hat = {k}/{n} = {rational_text(p_hat)}."],
                (0, 1),
            ),
            part(
                p2,
                sq,
                2,
                b_d,
                [f"The estimate of p² is p-hat² = ({rational_text(p_hat)})² = {rational_text(sq)}."],
                (0, 1),
                follow_through="part_a**2",
            ),
            part(
                p3,
                odds,
                2,
                c_d,
                [
                    f"Odds = p-hat / (1 - p-hat) = {rational_text(p_hat)} / {rational_text(1 - p_hat)} = {rational_text(odds)}."
                ],
                (0, None),
                follow_through="part_a/(1 - part_a)",
            ),
        ],
    )


@generator("exd_x_u6_bernoulli_invariance", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u6_bernoulli_invariance(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: Bernoulli MLE from a recorded sequence, then the estimates of p squared and of the odds."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a sequence with enough distinct distractors"):
        question = _build(rng)
        if question is not None:
            return question
