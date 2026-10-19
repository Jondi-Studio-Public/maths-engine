"""Exam source for `f10-measurement-error-interpretation`: limits of accuracy for an area from rounded sides.

A rectangle's sides are reported to the nearest 1, 2 or 5 units. The learner finds the calculated area, the
largest area the sides allow, the greatest possible absolute error in the area, and that error as a percentage
of the calculated area.

(a) calculated area, (b) greatest possible area, (c) greatest possible error, (d) percentage error.
Part (c) follows from (a) and (b); part (d) from (a) and (c). Built from exact fractions; the percentage is kept
clear of a rounding boundary so a learner's rounding route agrees with the key.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, exact_decimal_text, generator, latex_answer, round_half_up
from mathgen.generators.exd_x_m2_support import display, keys_absent, tagged_distractors

_SETTINGS = [
    ("A rectangular poster", "cm", "measured to the nearest"),
    ("A rectangular paddock", "m", "paced out and recorded to the nearest"),
    ("A rectangular courtyard", "m", "measured to the nearest"),
    ("A rectangular banner", "cm", "measured to the nearest"),
]


@generator("exd_x_m2_area_limits", level=9, presets={"exam": dict(tier="exam")})
def exd_x_m2_area_limits(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: area, greatest possible area, greatest possible error, percentage error."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw clear of a rounding boundary"):
        unit = rng.choice([1, 1, 2, 5])
        length = unit * rng.randint(4, 14)
        width = unit * rng.randint(3, 12)
        if length == width:
            continue
        half = Fraction(unit, 2)
        area = length * width
        upper = (length + half) * (width + half)
        err = upper - area
        pct = 100 * err / area
        if abs((pct * 10) % 1 - Fraction(1, 2)) <= Fraction(1, 50):
            continue
        thing, u, phrase_ = rng.choice(_SETTINGS)
        stem = f"{thing} has a length of {length} {u} and a width of {width} {u}, each {phrase_} {unit} {u}."
        prompts = [
            f"Calculate the area using these measurements, in square {u}.",
            f"Find the greatest area the {thing.split()[-1]} could actually have, in square {u}, correct to 2 decimal places.",
            f"Hence find the greatest possible error in the calculated area, in square {u}, correct to 2 decimal places.",
            "Express this error as a percentage of the calculated area, correct to 1 decimal place.",
        ]
        texts = [str(area), display(upper, 2), display(err, 2), display(pct, 1)]
        if keys_absent(stem, prompts, texts):
            break
    check(err == (length + width) * half + half * half, "error disagrees with the expanded product")
    check(round_half_up(pct, 1) == round_half_up(100 * (upper - area) / area, 1), "percentage disagrees")

    a_dist = tagged_distractors(
        texts[0],
        0,
        [
            (length + width, None),
            (2 * (length + width), None),
            (upper, None),
            (area + length, None),
            (area - width, None),
        ],
    )
    b_dist = tagged_distractors(
        texts[1],
        2,
        [
            ((length + unit) * (width + unit), None),
            ((length + half) * width, None),
            ((length - half) * (width - half), None),
            (area + half, None),
            (area + 2 * err, None),
        ],
    )
    c_dist = tagged_distractors(
        texts[2],
        2,
        [
            (half, None),
            (half * 2, None),
            (area - (length - half) * (width - half), None),
            (upper, None),
            (err * 2, None),
            (err - half, None),
        ],
    )
    d_dist = tagged_distractors(
        texts[3],
        1,
        [
            (100 * err / upper, "divides-by-final"),
            (err, "gave-change-not-percent"),
            (err / area, None),
            (100 * half / area, None),
            (2 * pct, None),
        ],
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        check(len(dist) >= 3, "not enough distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=area,
                answer_text=texts[0],
                answer_latex=latex_answer(texts[0]),
                marks=1,
                sensible_range=(0, None),
                working=[f"{length} × {width} = {area}."],
                distractors=a_dist,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=upper,
                answer_text=texts[1],
                answer_latex=latex_answer(texts[1]),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Each side is given to the nearest {unit} {u}, so it could be up to {exact_decimal_text(half)} {u} "
                    f"more: {exact_decimal_text(length + half)} and {exact_decimal_text(width + half)}.",
                    f"{exact_decimal_text(length + half)} × {exact_decimal_text(width + half)} = {texts[1]}.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=err,
                answer_text=texts[2],
                answer_latex=latex_answer(texts[2]),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_b - part_a",
                working=[f"{texts[1]} - {area} = {texts[2]}."],
                distractors=c_dist,
            ),
            dict(
                prompt=prompts[3],
                answer_kind="number",
                answer=round_half_up(pct, 1),
                answer_text=texts[3],
                answer_latex=latex_answer(texts[3]),
                marks=2,
                sensible_range=(0, None),
                follow_through="100*part_c/part_a",
                working=[f"({texts[2]} ÷ {area}) × 100 = {texts[3]}%."],
                distractors=d_dist,
            ),
        ],
    )
