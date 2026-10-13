"""Exam tier for `f10-fractions-half-quarter-eighth`: repeated halving of what is left, in a real setting.

A block of land is split by halving the part that remains, over and over: half is used for the first purpose, half
of what is left for the second, and so on, until some land is left unused. The learner must see that every halving
is of the LEFTOVER (so the pieces are 1/2, 1/4, 1/8, ...) and carry the unused fraction into an area.

* (a) the fraction of the block still unused after the stated number of halvings (1 mark);
* (b) the area of the unused land (2 marks, follow-through from (a));
* (c) the area that has been put to use: the whole block less (b) (1 mark, follow-through from (b)).

Built backwards from the area: it is a multiple of 2^k, so the unused area is a whole number of square units.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, numeric_answer, pick_distractors, rational_text

# (opening, uses in order, what is left, unit, lowest and highest multiplier of 2^k for the area)
_SETTINGS = [
    (
        "A farmer near Wagga Wagga has a paddock of {area} hectares.",
        ["wheat", "canola", "barley", "oats", "lupins"],
        "left as bush",
        "hectares",
        (3, 60),
    ),
    (
        "A community garden in Fitzroy covers {area} square metres.",
        ["vegetable beds", "a herb garden", "a playground", "a compost area", "a seating area"],
        "left as lawn",
        "square metres",
        (2, 90),
    ),
    (
        "A primary school playground in Launceston has an area of {area} square metres.",
        ["a basketball court", "a handball area", "a sandpit", "a veggie patch", "a shade garden"],
        "left open for free play",
        "square metres",
        (6, 60),
    ),
]
_ORDINALS = ["first", "second", "third", "fourth", "fifth"]


class _Retry(Exception):
    pass


def _whole(pairs):
    """Distractors for the (value, misconception) pairs that are whole numbers, as a learner would write them."""
    return [Distractor(str(int(v)), m, v) for v, m in pairs if v.denominator == 1 and v > 0]


def _attempt(rng: random.Random) -> dict:
    opening, uses, leftover_phrase, unit, (lo, hi) = rng.choice(_SETTINGS)
    steps = rng.randint(3, 5)
    scale = rng.randint(lo, hi)
    area = scale * 2**steps
    unused = Fraction(1, 2**steps)
    unused_area = scale
    used_area = area - scale

    sentences = [opening.format(area=area)]
    for i in range(steps):
        sentences.append(f"{'Half' if i == 0 else 'Half of what is left'} is used for {uses[i]}.")
    sentences.append(f"The land still unused after that is {leftover_phrase}.")
    stem = " ".join(sentences)

    d_a = pick_distractors(
        rational_text(unused),
        [
            Distractor(
                rational_text(Fraction(1, 2 * steps)), "halvings-counted-not-compounded", Fraction(1, 2 * steps)
            ),
            Distractor(
                rational_text(Fraction(1, 2 ** (steps - 1))), "off-by-one-halving-step", Fraction(1, 2 ** (steps - 1))
            ),
            Distractor(
                rational_text(Fraction(1, 2 ** (steps + 1))), "off-by-one-halving-step", Fraction(1, 2 ** (steps + 1))
            ),
            Distractor(rational_text(Fraction(1, steps)), "halvings-counted-not-compounded", Fraction(1, steps)),
        ],
        3,
    )
    d_b = pick_distractors(
        str(unused_area),
        _whole(
            [
                (Fraction(area, 2 * steps), "halvings-counted-not-compounded"),
                (Fraction(area, 2 ** (steps - 1)), "off-by-one-halving-step"),
                (Fraction(area, 2 ** (steps + 1)), "off-by-one-halving-step"),
                (Fraction(used_area), "answers-amount-used-not-remaining"),
            ]
        ),
        3,
    )
    d_c = pick_distractors(
        str(used_area),
        _whole(
            [
                (Fraction(unused_area), "answers-amount-used-not-remaining"),
                (Fraction(area, 2), "omitted-a-term"),
                (area - Fraction(area, 2 ** (steps - 1)), "off-by-one-halving-step"),
                (area - Fraction(area, 2 * steps), "halvings-counted-not-compounded"),
            ]
        ),
        3,
    )
    if min(len(d_a), len(d_b), len(d_c)) < 3:
        raise _Retry
    check(unused_area + used_area == area, "areas do not add up")
    check(Fraction(unused_area, area) == unused, "unused fraction disagrees")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="What fraction of the land is unused once all the halving is done?",
                answer_kind="fraction",
                answer={"p": unused.numerator, "q": unused.denominator},
                answer_text=rational_text(unused),
                answer_latex=latex_answer(rational_text(unused)),
                marks=1,
                sensible_range=(0, 1),
                working=[
                    f"Each halving leaves half of what was left, so {steps} halvings leave (1/2)^{steps} = {rational_text(unused)}."
                ],
                distractors=d_a,
            ),
            dict(
                prompt=f"How many {unit} are unused?",
                answer_kind="number",
                answer=numeric_answer(Fraction(unused_area)),
                answer_text=str(unused_area),
                answer_latex=latex_answer(str(unused_area)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{area}",
                working=[f"{rational_text(unused)} of {area} = {unused_area} {unit}."],
                distractors=d_b,
            ),
            dict(
                prompt=f"How many {unit} are put to use?",
                answer_kind="number",
                answer=numeric_answer(Fraction(used_area)),
                answer_text=str(used_area),
                answer_latex=latex_answer(str(used_area)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{area} - part_b",
                working=[f"{area} - {unused_area} = {used_area} {unit}."],
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


@generator("exd_x_pr5_halving_paddock", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_halving_paddock(rng: random.Random, tier: str = "exam") -> dict:
    """Repeatedly halve what is left of a block of land, then find the unused and the used areas."""
    check(tier == "exam", "exd_x_pr5_halving_paddock: only the exam tier exists")
    return _build(rng)
