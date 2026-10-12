"""Exam-tier source for `expected-frequency`: from counts to a probability to expected frequencies.

The practice tier gives the probability as a fraction and asks for one product. Here the probability is not given:
it has to be built from the composition of a bag, spinner or drum with three colours, and then used twice.

(a) the probability of one of two colours in a single trial       fraction, 1 mark
(b) the number of times those two colours are expected in N       number,   2 marks, ``part_a * N``
    trials with replacement
(c) the number of times the third colour is expected              number,   1 mark,  ``N - part_b``

N is a multiple of the total so every expected frequency is a whole number. The three colour counts and the number
of trials are all drawn, so the answers spread over a wide range.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors, rational_text


def _part(kind, prompt, value, marks, working, candidates, follow_through=None):
    text = rational_text(value)
    options = [
        Distractor(rational_text(v), tag, exact_value(v))
        for tag, v in candidates
        if v != value and (kind == "fraction" or v.denominator == 1)
    ]
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, options, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _draw(rng: random.Random) -> dict | None:
    r, b, g = (rng.randint(1, 12) for _ in range(3))
    total = r + b + g
    if not 8 <= total <= 30:
        return None
    k = rng.randint(3, 30)
    trials = total * k
    wanted = r + b
    p = Fraction(wanted, total)
    expected = p * trials
    green = trials - expected
    check(expected.denominator == 1 and green == g * k, "expected frequencies are not whole numbers")
    check(expected == k * wanted, "expected frequency disagrees with k x favourable")

    stem = phrase(
        rng,
        (
            "A bag holds {r} red, {b} blue and {g} green marbles. A marble is drawn at random, its colour is noted "
            "and it is put back. This is done {n} times.",
            "A spinner has {t} equal sectors: {r} red, {b} blue and {g} green. The spinner is spun {n} times.",
            "A raffle drum holds {t} tickets: {r} red, {b} blue and {g} green. A ticket is drawn at random and "
            "returned to the drum, and this happens {n} times.",
        ),
        r=r,
        b=b,
        g=g,
        t=total,
        n=trials,
    )
    part_a = _part(
        "fraction",
        "What is the probability that a single trial gives red or blue?",
        p,
        1,
        [f"P(red or blue) = ({r} + {b})/{total} = {rational_text(p)}."],
        [
            ("ratio-instead-of-probability", Fraction(wanted, g)),
            ("gave-complement", Fraction(g, total)),
            (None, Fraction(r, total)),
            (None, Fraction(1, total)),
        ],
    )
    part_b = _part(
        "number",
        f"Hence find how many times red or blue is expected in the {trials} trials.",
        expected,
        2,
        [f"Expected = {rational_text(p)} × {trials} = {expected}."],
        [
            ("divides-trials-only", Fraction(trials, total)),
            ("ratio-instead-of-probability", Fraction(wanted * trials, g)),
            ("gave-complement", Fraction(green)),
            (None, Fraction(r * trials, total)),
        ],
        follow_through=f"part_a*{trials}",
    )
    part_c = _part(
        "number",
        "Hence find how many times a green is expected.",
        green,
        1,
        [f"Green is expected {trials} − {expected} = {green} times."],
        [
            ("gave-complement", expected),
            (None, Fraction(trials, total)),
            (None, Fraction(g * trials, wanted)),
            (None, Fraction(g * trials)),
        ],
        follow_through=f"{trials} - part_b",
    )
    if any(len(pt["distractors"]) < 3 for pt in (part_a, part_b, part_c)):
        return None
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_u5_expected_chain", level=8, presets={"exam": dict(tier="exam")})
def exd_x_u5_expected_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a probability from counts, then expected frequencies for two outcomes."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw with three distinct distractors per part"):
        question = _draw(rng)
        if question is not None:
            return question
