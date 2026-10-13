"""Exam question on decimals on a number line with an unfamiliar scale.

Distance posts are labelled every tenth of a kilometre, and each gap between labelled posts is split into 2, 4 or
5 equal sections. Part (a) reads the position of a stop a few sections past a labelled post (the learner must work
out what one section is worth), part (b) moves a further number of sections on, past the next labelled post, and
part (c) finds the distance back to the labelled post after that.

Everything is built from exact fractions of a kilometre. The step is 0.05, 0.025 or 0.02, so readings carry 2 or
3 decimal places and every reading is stated at its own exact precision.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

_SETTINGS = (
    "Distance posts along a cycle path beside the Brisbane River are labelled every tenth of a kilometre.",
    "Distance posts on a running trail in Kings Park, Perth, are labelled every tenth of a kilometre.",
    "A trail marker board on a Blue Mountains walk is labelled every tenth of a kilometre.",
    "Distance pegs on a Canberra lake foreshore loop are labelled every tenth of a kilometre.",
)
_STOPS = ("rest stop", "drink station", "picnic table", "lookout")


@generator("exd_x_pr4_decimal_numberline_scale", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_decimal_numberline_scale(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_decimal_numberline_scale: only the exam tier exists")
    sections = rng.choice((2, 4, 5))
    step = Fraction(1, 10 * sections)
    dp = 3 if sections == 4 else 2
    label = Fraction(rng.randint(21, 189), 10)
    for _ in attempts(1000, "stops that straddle the next labelled post"):
        k = rng.randint(1, sections - 1)
        j = rng.randint(2, 2 * sections - 2)
        if sections < k + j < 2 * sections:
            break
    p_value = label + k * step
    q_value = p_value + j * step
    next_label = label + Fraction(2, 10)
    back = next_label - q_value

    # Independent route: count every section from the labelled post as a whole number of steps.
    steps_total = k + j
    check(label + steps_total * step == q_value, "reading disagrees")
    check(sections < steps_total < 2 * sections and back == (2 * sections - steps_total) * step, "back disagrees")

    def text(v: Fraction) -> str:
        return stated_precision_string(v, dp)

    p_text, q_text, back_text = text(p_value), text(q_value), text(back)
    a_candidates = [
        Distractor(text(label + Fraction(k, 10)), "gap-read-as-tenth", label + Fraction(k, 10)),
        Distractor(text(label + (k + 1) * step), "counts-marks-not-gaps", label + (k + 1) * step),
        Distractor(text(label + Fraction(k, 100)), None, label + Fraction(k, 100)),
    ]
    b_candidates = [
        Distractor(text(label + j * step), "restarts-count-from-the-labelled-mark", label + j * step),
        Distractor(text(p_value + Fraction(j, 10)), "gap-read-as-tenth", p_value + Fraction(j, 10)),
        Distractor(text(p_value + (j + 1) * step), "counts-marks-not-gaps", p_value + (j + 1) * step),
    ]
    c_candidates = [
        Distractor(
            text(q_value - (label + Fraction(1, 10))),
            "measures-to-the-wrong-labelled-mark",
            q_value - label - Fraction(1, 10),
        ),
        Distractor(
            text(label + Fraction(3, 10) - q_value),
            "measures-to-the-wrong-labelled-mark",
            label + Fraction(3, 10) - q_value,
        ),
        Distractor(text(q_value - label), "measures-to-the-wrong-labelled-mark", q_value - label),
        Distractor(
            text(Fraction(2 * sections - steps_total, 10)),
            "gap-read-as-tenth",
            Fraction(2 * sections - steps_total, 10),
        ),
    ]
    a_d = pick_distractors(p_text, a_candidates, 3, tolerance=Fraction(1, 2000))
    b_d = pick_distractors(q_text, b_candidates, 3, tolerance=Fraction(1, 2000))
    c_d = pick_distractors(back_text, c_candidates, 3, tolerance=Fraction(1, 2000))
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors")

    stop = rng.choice(_STOPS)
    stem = (
        f"{rng.choice(_SETTINGS)} Each gap between two labelled posts is divided into {sections} equal sections. "
        f"A {stop} (P) is {k} section{'s' if k != 1 else ''} past the {stated_precision_string(label, 1)} km post. "
        f"A water tap (Q) is {j} sections past P."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How far along the path is P, in kilometres?",
                answer_kind="number",
                answer=numeric_answer(p_value),
                answer_text=p_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"One section is 0.1 ÷ {sections} = {stated_precision_string(step, 3)} km.",
                    f"P = {stated_precision_string(label, 1)} + {k} × {stated_precision_string(step, 3)} = {p_text} km.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="How far along the path is Q, in kilometres?",
                answer_kind="number",
                answer=numeric_answer(q_value),
                answer_text=q_text,
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a + {j * step.numerator}/{step.denominator}",
                working=[f"Q = {p_text} + {j} × {stated_precision_string(step, 3)} = {q_text} km."],
                distractors=b_d,
            ),
            dict(
                prompt=f"How far is Q before the {stated_precision_string(next_label, 1)} km post, in kilometres?",
                answer_kind="number",
                answer=numeric_answer(back),
                answer_text=back_text,
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{next_label.numerator}/{next_label.denominator} - part_b",
                working=[f"{stated_precision_string(next_label, 1)} - {q_text} = {back_text} km."],
                distractors=c_d,
            ),
        ],
    )
