"""Exam-tier source for `f10-probability-scale-and-expectation`: from counts to a probability, to an expected
frequency, and back to the number of trials.

A bag, spinner or barrel holds items of three kinds. The learner

1. finds the probability of one kind as a decimal (the total is not given, it must be added up),
2. uses it to predict how many times that kind occurs in a long run of trials, and
3. works backwards: how many trials are needed before a stated number of that kind is expected.

Built backwards from a total that makes every probability an exact two-place decimal, and a trial count that makes
every expected frequency a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import (
    attempts,
    GRADER_TOLERANCE,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from .exd_x_u6_support import keys_clear, tagged, within_unit_interval

_TOTALS = [20, 25, 50, 100]
_SETTINGS = [
    (
        "A bag holds {r} red, {g} green and {b} blue marbles. One marble is drawn at random, its colour is recorded "
        "and it is put back.",
        "red",
        "marble",
        "draws",
    ),
    (
        "A fete lucky-dip barrel holds {r} red, {g} green and {b} blue tickets. A ticket is drawn at random, noted and "
        "returned to the barrel.",
        "red",
        "ticket",
        "draws",
    ),
    (
        "A spinner is divided into {n} equal sectors: {r} are red, {g} are green and {b} are blue. The spinner is "
        "spun and the colour it stops on is noted.",
        "red",
        "spin",
        "spins",
    ),
    (
        "A jar contains {r} red, {g} green and {b} blue jelly beans. A jelly bean is taken without looking, its "
        "colour is noted and it is put back.",
        "red",
        "jelly bean",
        "draws",
    ),
]


def _dec(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _int_or_dec(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else _dec(value)


@generator("exd_x_u6_probability_expectation", level=10, presets={"exam": dict(tier="exam")})
def exd_x_u6_probability_expectation(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: probability from counts, expected frequency, then the trials needed for a target count."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a bag with enough distractors"):
        question = _build(rng)
        if question is not None:
            return question


def _build(rng: random.Random):
    for _ in range(1000):
        n = rng.choice(_TOTALS)
        r = rng.randint(2, n // 2 - 1)
        g = rng.randint(1, n - r - 1)
        b = n - r - g
        p = Fraction(r, n)
        if b >= 1 and p.denominator in (4, 5, 10, 20, 25, 50, 100) and 0 < p < Fraction(1, 2) + Fraction(1, 5):
            break
    else:  # pragma: no cover - the draw space is large
        return None
    den = p.denominator
    trials = den * rng.randint(4, 30)
    expected = p * trials
    target_k = rng.randint(2, 12)
    target = p.numerator * target_k
    needed = Fraction(target) / p
    check(expected.denominator == 1 and needed.denominator == 1, "expected frequency is not whole")

    setting, colour, item, trial_word = rng.choice(_SETTINGS)
    stem = setting.format(r=r, g=g, b=b, n=n)
    # The total must be added up by the learner: a spinner states it as sectors, which is the same job.

    a_text = _dec(p)
    b_text = str(int(expected))
    c_text = str(int(needed))

    a_cands = within_unit_interval(
        [
            tagged(_dec(1 - p), "gave-complement-probability", round_half_up(1 - p, 2)),
            tagged(_dec(Fraction(r, n - r)), "favourable-over-unfavourable", round_half_up(Fraction(r, n - r), 2)),
            tagged(_dec(Fraction(r, r + g)), "left-out-a-category-from-total", round_half_up(Fraction(r, r + g), 2)),
            tagged(_dec(Fraction(r, r + b)), "left-out-a-category-from-total", round_half_up(Fraction(r, r + b), 2)),
        ]
    )
    a_distractors = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)

    b_cands = [
        tagged(_dec(p), "probability-not-scaled-by-trials", round_half_up(p, 2)),
        tagged(_int_or_dec(Fraction(trials) / p), "divides-trials-by-probability", Fraction(trials) / p),
        tagged(_int_or_dec((1 - p) * trials), "gave-complement-probability", (1 - p) * trials),
    ]
    b_distractors = pick_distractors(b_text, [c for c in b_cands if c.value != expected], 3)

    c_cands = [
        tagged(_int_or_dec(target * p), "multiplies-by-probability-to-invert", round_half_up(target * p, 2)),
        tagged(_int_or_dec(target / (1 - p)), "gave-complement-probability", round_half_up(target / (1 - p), 2)),
    ]
    c_distractors = pick_distractors(c_text, [c for c in c_cands if c.value != needed], 3)
    if len(a_distractors) < 2 or len(b_distractors) < 2 or not c_distractors:
        return None

    p1 = f"Find the probability that the {item} is {colour}, as a decimal."
    p2 = f"Use your answer to (a) to find how many times {colour} is expected in {trials} {trial_word}."
    p3 = f"Using your answer to (a), how many {trial_word} are needed before {target} {colour} results are expected?"
    if not keys_clear(" ".join([stem, p1, p2, p3]), [p, expected, needed]):
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p1,
                answer_kind="number",
                answer=numeric_answer(round_half_up(p, 2)),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, 1),
                working=[f"Total = {r} + {g} + {b} = {n}.", f"P({colour}) = {r}/{n} = {a_text}."],
                distractors=a_distractors,
            ),
            dict(
                prompt=p2,
                answer_kind="number",
                answer=numeric_answer(expected),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{trials}",
                working=[f"Expected = probability × trials = {a_text} × {trials} = {b_text}."],
                distractors=b_distractors,
            ),
            dict(
                prompt=p3,
                answer_kind="number",
                answer=numeric_answer(needed),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{target}/part_a",
                working=[f"Expected {colour} = {a_text} × N, so N = {target} ÷ {a_text} = {c_text}."],
                distractors=c_distractors,
            ),
        ],
    )
