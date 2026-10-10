"""Exam-tier questions for traffic-light probability: conditional probability
about the lights AFTER learning something about the outcome, so the learner has
to choose between the product, sum and complement rules and then divide.

Three kinds, all answered as exact fractions built by enumerating the light
outcomes:

``exactly_one``  two independent lights; given that exactly one is green, the
                 probability it was light 1.
``at_least_one`` two independent lights; given that at least one is green, the
                 probability both are.
``dependent``    light 2 is NOT independent of light 1: its green probability
                 depends on light 1's colour. Given that light 2 is green, the
                 probability that light 1 was green (Bayes' rule with the law
                 of total probability).

A draw is rejected unless the answer differs from each named wrong answer,
so every distractor is a principled slip that the exact grader rejects.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text

_PROBS = (
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(2, 3),
    Fraction(1, 4),
    Fraction(3, 4),
    Fraction(2, 5),
    Fraction(3, 5),
    Fraction(1, 6),
    Fraction(5, 6),
)


@generator(
    "exd_x_s2_traffic",
    level=12,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s2_traffic(rng: random.Random, tier: str = "exam") -> dict:
    """A conditional probability about traffic lights."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice(("exactly_one", "at_least_one", "dependent"))
    while True:
        p1, p2 = rng.choice(_PROBS), rng.choice(_PROBS)
        if kind != "dependent":
            joint = {(i, j): (p1 if i else 1 - p1) * (p2 if j else 1 - p2) for i in (0, 1) for j in (0, 1)}
        else:
            # p2 = P(light 2 green | light 1 green); p3 = P(light 2 green | light 1 red)
            p3 = rng.choice(_PROBS)
            if p2 == p3:
                continue
            joint = {
                (1, 1): p1 * p2,
                (1, 0): p1 * (1 - p2),
                (0, 1): (1 - p1) * p3,
                (0, 0): (1 - p1) * (1 - p3),
            }
        check(sum(joint.values()) == 1, "light outcome probabilities must sum to 1")
        if kind == "exactly_one":
            cond = [k for k in joint if k[0] != k[1]]
            target = [k for k in cond if k[0] == 1]
            text = (
                f"A commuter passes two independent traffic lights. Light 1 is green with probability {rational_text(p1)} "
                f"and light 2 is green with probability {rational_text(p2)}. "
                "Given that exactly one of the lights is green, what is the probability that it is light 1?"
            )
            slips = [
                joint[(1, 0)],  # joint instead of conditional
                p1,  # ignores the condition
                1 - sum(joint[k] for k in target) / sum(joint[k] for k in cond),
                joint[(1, 0)] / (1 - joint[(1, 1)]) if joint[(1, 1)] != 1 else p1,
            ]
        elif kind == "at_least_one":
            cond = [k for k in joint if k != (0, 0)]
            target = [(1, 1)]
            text = (
                f"A commuter passes two independent traffic lights. Light 1 is green with probability {rational_text(p1)} "
                f"and light 2 is green with probability {rational_text(p2)}. "
                "Given that at least one of the lights is green, what is the probability that both are green?"
            )
            slips = [
                joint[(1, 1)],  # forgets to condition
                joint[(1, 1)] / p1,  # conditions on light 1 only
                1 - joint[(0, 0)],  # probability of the condition itself
                joint[(1, 1)] / (joint[(1, 0)] + joint[(0, 1)]),  # conditions on exactly one
            ]
        else:
            cond = [k for k in joint if k[1] == 1]
            target = [(1, 1)]
            text = (
                f"The first of two traffic lights is green with probability {rational_text(p1)}. "
                f"If it is green, the second is green with probability {rational_text(p2)}; "
                f"if it is red, the second is green with probability {rational_text(p3)}. "
                "Given that the second light is green, what is the probability that the first was green?"
            )
            slips = [
                p2,  # reverses the conditional
                joint[(1, 1)],  # joint probability
                p1,  # prior
                joint[(1, 1)] / (joint[(1, 1)] + joint[(0, 0)]) if joint[(1, 1)] + joint[(0, 0)] else p1,
            ]
        denom = sum(joint[k] for k in cond)
        answer = sum(joint[k] for k in target) / denom
        if 0 < answer < 1 and answer not in {s for s in slips if 0 <= s <= 1}:
            break
    answer_text = rational_text(answer)
    check(
        answer == Fraction(sum(joint[k] for k in target)) / sum(joint[k] for k in cond), "traffic conditional mismatch"
    )
    working = [
        "Conditioning restricts attention to the outcomes consistent with the given information.",
        f"P(given) = {rational_text(denom)}, P(target and given) = {rational_text(sum(joint[k] for k in target))}.",
        f"Answer = {rational_text(sum(joint[k] for k in target))}/{rational_text(denom)} = {answer_text}.",
    ]
    return dict(
        text=text,
        latex="plain:" + text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [rational_text(s) for s in slips if 0 <= s <= 1]),
    )
