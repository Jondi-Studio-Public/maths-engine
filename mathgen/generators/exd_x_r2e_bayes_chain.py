"""Exam source for `ucv1-bayes-finite-counts`: a four-part chain on one batch of items from two machines.

A batch of ``N`` items comes from two machines in a stated ratio, and a stated fraction of each machine's
items is faulty. The counts are whole numbers throughout.

- (a) the probability that an item chosen at random is faulty (2 marks);
- (b) the probability that a faulty item came from Machine 1, by Bayes' rule (2 marks, ``follow_through`` from
  (a));
- (c) the probability that two items chosen without replacement are both faulty (2 marks, ``follow_through``
  from (a), through the number of faulty items);
- (d) the expected number of items from Machine 1 among ``k`` faulty items examined (2 marks,
  ``follow_through`` from (b)).

## Construction

The ratio, the two fault fractions and a multiplier are drawn, and the multiplier is chosen so that each
machine's output times its fault fraction is a whole number of items. All answers are `fractions.Fraction`
values from those counts. The posterior of (b) is recomputed as ``faulty from Machine 1 / all faulty``, a route
that does not pass through (a)'s probability, and the two-item probability from the counts directly.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, pick_distractors, rational_text

_FAULT_FRACTIONS = [
    Fraction(1, 20),
    Fraction(1, 12),
    Fraction(1, 10),
    Fraction(1, 8),
    Fraction(1, 6),
    Fraction(3, 20),
    Fraction(1, 5),
    Fraction(1, 4),
    Fraction(3, 10),
    Fraction(1, 3),
    Fraction(2, 5),
]
_THINGS = [
    ("items", "faulty", "Machine", "machines"),
    ("components", "defective", "Line", "lines"),
    ("parts", "defective", "Press", "presses"),
    ("bolts", "faulty", "Machine", "machines"),
]


def _text(f: Fraction) -> str:
    return rational_text(f)


def _tag(value: Fraction, tag: str) -> Distractor:
    return Distractor(_text(value), tag, exact_value(value))


def _rate(f: Fraction) -> str:
    return f"{f.numerator} out of every {f.denominator}"


@generator("exd_x_r2e_bayes_chain", level=12, presets={"exam": dict(tier="exam")})
def bayes_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Total probability, Bayes' rule, a two-item draw without replacement, and an expected count."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "a batch with whole-number counts"):
        r, t = rng.sample(range(1, 6), 2)
        p1, p2 = rng.sample(_FAULT_FRACTIONS, 2)
        u = rng.randint(2, 30)
        n1, n2 = r * u, t * u
        if (n1 * p1).denominator != 1 or (n2 * p2).denominator != 1:
            continue
        total = n1 + n2
        if not 40 <= total <= 300:
            continue
        f1, f2 = int(n1 * p1), int(n2 * p2)
        faulty = f1 + f2
        if faulty < 4 or f1 == f2:
            continue
        break
    k = rng.randint(10, 60)

    share1 = Fraction(n1, total)
    prob_faulty = Fraction(faulty, total)
    check(prob_faulty == share1 * p1 + (1 - share1) * p2, "total probability disagrees with the counts")
    posterior = Fraction(f1, faulty)
    check(posterior == share1 * p1 / prob_faulty, "Bayes' rule disagrees with the counts")
    both = Fraction(faulty * (faulty - 1), total * (total - 1))
    check(both == prob_faulty * Fraction(faulty - 1, total - 1), "the two-item probability disagrees")
    expected = k * posterior

    items, bad, unit, units = rng.choice(_THINGS)
    stem = (
        f"A batch of {total} {items} is made on two {units}, {unit} 1 and {unit} 2, in the ratio {r} : {t}. "
        f"On {unit} 1, {_rate(p1)} {items} are {bad}. On {unit} 2, {_rate(p2)} {items} are {bad}."
    )

    a_cands = [
        _tag((p1 + p2) / 2, "averages-the-rates"),
        _tag(share1 * p1, "uses-only-one-branch"),
        _tag(1 - prob_faulty, "gives-the-complement"),
        _tag(Fraction(faulty, total - 1), "off-by-one-count"),
        _tag(p1 + p2, "adds-probabilities-without-weights"),
    ]
    b_cands = [
        _tag(p1, "confuses-the-direction-of-the-conditional"),
        _tag(share1, "ignores-the-evidence"),
        _tag(share1 * p1, "gives-joint-not-conditional"),
        _tag(1 - posterior, "gives-the-complement"),
    ]
    c_cands = [
        _tag(prob_faulty**2, "assumes-with-replacement"),
        _tag(Fraction(faulty * faulty, total * (total - 1)), "assumes-with-replacement"),
        _tag(2 * prob_faulty, "adds-instead-of-multiplies"),
        _tag(Fraction(faulty * (faulty - 1), total * total), "assumes-with-replacement"),
    ]
    d_cands = [
        _tag(k * share1, "ignores-the-evidence"),
        _tag(k * p1, "confuses-the-direction-of-the-conditional"),
        _tag(k * (1 - posterior), "gives-the-complement"),
        _tag(k * share1 * p1, "gives-joint-not-conditional"),
    ]
    picks = [
        pick_distractors(_text(prob_faulty), a_cands, 3),
        pick_distractors(_text(posterior), b_cands, 3),
        pick_distractors(_text(both), c_cands, 3),
        pick_distractors(_text(expected), d_cands, 3),
    ]
    check(all(len(p) >= 2 for p in picks), f"a part has too few distractors: {[len(p) for p in picks]}")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the probability that an item chosen at random from the batch is {bad}.",
                answer_kind="fraction",
                answer=exact_value(prob_faulty),
                answer_text=_text(prob_faulty),
                answer_latex=latex_answer(_text(prob_faulty)),
                marks=2,
                sensible_range=(0, 1),
                working=[
                    f"{unit} 1 makes {n1} {items}, of which {f1} are {bad}; {unit} 2 makes {n2}, of which {f2} are {bad}.",
                    f"P({bad}) = ({f1} + {f2}) / {total} = {_text(prob_faulty)}.",
                ],
                distractors=picks[0],
            ),
            dict(
                prompt=f"An item chosen at random is {bad}. Find the probability that it was made on {unit} 1.",
                answer_kind="fraction",
                answer=exact_value(posterior),
                answer_text=_text(posterior),
                answer_latex=latex_answer(_text(posterior)),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"({Fraction(n1, total) * p1})/part_a",
                working=[
                    f"P({unit} 1 | {bad}) = P({unit} 1 and {bad}) / P({bad}) = {_text(share1 * p1)} / {_text(prob_faulty)} = {_text(posterior)}."
                ],
                distractors=picks[1],
            ),
            dict(
                prompt=f"Two items are chosen at random from the batch, without replacement. Find the "
                f"probability that both are {bad}.",
                answer_kind="fraction",
                answer=exact_value(both),
                answer_text=_text(both),
                answer_latex=latex_answer(_text(both)),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"({total}*part_a)*({total}*part_a - 1)/({total * (total - 1)})",
                working=[
                    f"There are {faulty} {bad} items in {total}: P(both) = {faulty}/{total} x {faulty - 1}/{total - 1} = {_text(both)}."
                ],
                distractors=picks[2],
            ),
            dict(
                prompt=f"All the {bad} {items} are collected, and {k} of them are chosen at random to be examined. "
                f"Find the expected number of these that were made on {unit} 1.",
                answer_kind="fraction",
                answer=exact_value(expected),
                answer_text=_text(expected),
                answer_latex=latex_answer(_text(expected)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{k}*part_b",
                working=[
                    f"Each examined item is from {unit} 1 with probability {_text(posterior)}, so E = {k} x {_text(posterior)} = {_text(expected)}."
                ],
                distractors=picks[3],
            ),
        ],
    )
