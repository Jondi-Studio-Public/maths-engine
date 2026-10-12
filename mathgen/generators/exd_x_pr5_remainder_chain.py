"""Exam tier for `fraction-of-quantity`: a fraction of what is left after another fraction has been taken.

A NAPLAN-style three-part problem in an Australian setting. A group of N people is split by a first fraction
(those who do one thing); a second fraction is then taken of the people who did NOT do it; the rest do a third
thing. The learner must see that the second fraction applies to the remainder, not to the whole group.

* (a) how many are not in the first group (the remainder, 1 mark);
* (b) how many are in the second group: the second fraction of (a) (2 marks, follow-through from (a));
* (c) how many are in the third group: (a) minus (b) (1 mark, follow-through from (a) and (b)).

Built backwards from whole numbers: N is a multiple of the first denominator and the remainder is a multiple of
the second denominator, so every quantity is a whole count.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, numeric_answer, pick_distractors

# (group, noun, first activity, second activity, third activity)
_SETTINGS = [
    ("Year 6 at Northcote Public School has", "students", "walk to school", "catch the bus", "are driven"),
    (
        "The Redcliffe Surf Life Saving Club has",
        "members",
        "swim in the ocean",
        "paddle a board",
        "run the beach flags",
    ),
    (
        "A hiking club in the Blue Mountains has",
        "walkers",
        "choose the short trail",
        "choose the medium trail",
        "choose the long trail",
    ),
    (
        "A school fete at Mildura sold",
        "tickets",
        "were raffle tickets",
        "were sausage-sizzle tickets",
        "were ride tickets",
    ),
    ("The Glenelg junior cricket carnival has", "players", "bat first", "bowl first", "field first"),
    (
        "A farm shop in the Barossa Valley packed",
        "boxes",
        "went to cafes",
        "went to the market",
        "were collected by the shop",
    ),
]
_DENOMS_FIRST = [3, 4, 5, 6, 8, 10]
_DENOMS_SECOND = [3, 4, 5, 6, 8]


def _fraction_text(frac: Fraction) -> str:
    return f"{frac.numerator}/{frac.denominator}"


class _Retry(Exception):
    """This draw cannot give enough distinct distractors; draw again."""


def _attempt(rng: random.Random) -> dict:
    for _ in attempts(1000, "a remainder chain with whole-number counts"):
        d1 = rng.choice(_DENOMS_FIRST)
        n1 = rng.randint(1, d1 - 1)
        d2 = rng.choice(_DENOMS_SECOND)
        n2 = rng.randint(1, d2 - 1)
        f1, f2 = Fraction(n1, d1), Fraction(n2, d2)
        if f1.denominator != d1 or f2.denominator != d2:  # keep both fractions in lowest terms
            continue
        total = d1 * rng.randint(4, 40)
        rest = total * (1 - f1)
        second = rest * f2
        if total < 60 or total > 480 or rest.denominator != 1 or second.denominator != 1:
            continue
        rest_i, second_i, third_i = int(rest), int(second), int(rest - second)
        answers = {rest_i, second_i, third_i}
        shown = {total, n1, d1, n2, d2}
        if len(answers) < 3 or third_i < 1 or answers & shown:
            continue
        break

    used = int(total * f1)
    setting = rng.choice(_SETTINGS)
    who, noun, act1, act2, act3 = setting
    stem = (
        f"{who} {total} {noun}. {_fraction_text(f1)} of the {noun} {act1}. "
        f"Of the {noun} left over, {_fraction_text(f2)} {act2}. The rest {act3}."
    )

    d_a = pick_distractors(
        str(rest_i),
        [
            Distractor(str(used), "answers-amount-used-not-remaining", used),
            Distractor(str(total - total // d1), "ignores-numerator", total - total // d1),
            Distractor(str(total // d1), "ignores-numerator", total // d1),
        ],
        3,
    )
    d_b = pick_distractors(
        str(second_i),
        [
            Distractor(str(int(total * f2)), "applies-fraction-to-original-whole", int(total * f2)),
            Distractor(str(rest_i // d2), "ignores-numerator", rest_i // d2),
            Distractor(str(rest_i - second_i), "answers-amount-used-not-remaining", rest_i - second_i),
            Distractor(str(used), "answers-amount-used-not-remaining", used),
        ],
        3,
    )
    d_c = pick_distractors(
        str(third_i),
        [
            Distractor(str(second_i), "omitted-a-term", second_i),
            Distractor(str(total - int(total * f2)), "applies-fraction-to-original-whole", total - int(total * f2)),
            Distractor(str(total - second_i), "omitted-a-term", total - second_i),
            Distractor(str(rest_i), "omitted-a-term", rest_i),
        ],
        3,
    )
    if min(len(d_a), len(d_b), len(d_c)) < 3:
        raise _Retry

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"How many {noun} are left after the {_fraction_text(f1)} that {act1} are taken away?",
                answer_kind="number",
                answer=numeric_answer(Fraction(rest_i)),
                answer_text=str(rest_i),
                answer_latex=latex_answer(str(rest_i)),
                marks=1,
                sensible_range=(0, None),
                working=[f"{total} - {_fraction_text(f1)} of {total} = {total} - {used} = {rest_i}."],
                distractors=d_a,
            ),
            dict(
                prompt=f"How many {noun} {act2}?",
                answer_kind="number",
                answer=numeric_answer(Fraction(second_i)),
                answer_text=str(second_i),
                answer_latex=latex_answer(str(second_i)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{n2}/{d2}",
                working=[f"{_fraction_text(f2)} of the {rest_i} left = {second_i}."],
                distractors=d_b,
            ),
            dict(
                prompt=f"How many {noun} {act3}?",
                answer_kind="number",
                answer=numeric_answer(Fraction(third_i)),
                answer_text=str(third_i),
                answer_latex=latex_answer(str(third_i)),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_a - part_b",
                working=[f"{rest_i} - {second_i} = {third_i}."],
                distractors=d_c,
            ),
        ],
    )


def _build(rng: random.Random) -> dict:
    for _ in attempts(200, "a draw with three distinct distractors per part"):
        try:
            return _attempt(rng)
        except _Retry:
            continue


@generator("exd_x_pr5_remainder_chain", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_remainder_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A fraction of the remainder after a first fraction is taken, in three connected counts."""
    check(tier == "exam", "exd_x_pr5_remainder_chain: only the exam tier exists")
    return _build(rng)
