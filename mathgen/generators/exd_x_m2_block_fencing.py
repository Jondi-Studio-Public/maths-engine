"""Exam source for `f10-regular-irregular-area-perimeter`: a block given by its area, with a corner cut out.

The learner is told a rectangular block's area and width and must find the length first. A rectangular corner is
then removed. They find the remaining area, and the cost of fencing the boundary of what remains. The fence
length is the original perimeter, because a rectangle cut from a corner moves the boundary without lengthening
it; adding or subtracting the cut-out's perimeter is the trap.

(a) the length of the block, (b) the remaining area, (c) the cost of fencing. Parts (b) and (c) carry follow-through.
"""

from __future__ import annotations

import random

from mathgen.core import attempts, check, generator, latex_answer
from mathgen.generators.exd_x_m2_support import keys_absent, tagged_distractors

_BLOCKS = ["A rectangular block of land", "A rectangular paddock", "A rectangular school oval", "A rectangular yard"]


@generator("exd_x_m2_block_fencing", level=7, presets={"exam": dict(tier="exam")})
def exd_x_m2_block_fencing(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: length from the area, remaining area after a corner cut-out, then fencing cost."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw whose answers are not already in the text"):
        w = rng.randint(8, 30)
        ln = rng.randint(w + 2, 60)
        p = rng.randint(2, w - 3)
        q = rng.randint(2, ln - 3)
        price = rng.choice([14, 16, 18, 22, 25, 28, 32, 36, 45])
        area = w * ln
        remaining = area - p * q
        perimeter = 2 * (w + ln)
        cost = perimeter * price
        stem = (
            f"{rng.choice(_BLOCKS)} has an area of {area} m² and a width of {w} m. "
            f"A rectangular {p} m by {q} m section is then cut out of one corner. "
            f"The whole boundary of what remains is to be fenced at ${price} per metre."
        )
        prompts = [
            "Find the length of the original block, in metres.",
            "Find the area that remains, in square metres.",
            "Find the cost of the fencing, in dollars.",
        ]
        a_text, b_text, c_text = str(ln), str(remaining), str(cost)
        if keys_absent(stem, prompts, [a_text, b_text, c_text]):
            break
    check(w * ln == area, "area disagrees with the dimensions")

    a_dist = tagged_distractors(a_text, 0, [(area + w, None), (area - w, None), (w * w, None), (area / 2 / w, None)])
    b_dist = tagged_distractors(
        b_text,
        0,
        [
            (area + p * q, None),
            (area, None),
            (ln * w - p - q, None),
            (p * q, None),
        ],
    )
    c_dist = tagged_distractors(
        c_text,
        0,
        [
            (price * (perimeter - 2 * (p + q)), None),
            (price * (perimeter + 2 * (p + q)), None),
            (price * (w + ln), None),
            (price * remaining, None),
        ],
    )
    for dist in (a_dist, b_dist, c_dist):
        check(len(dist) >= 3, "not enough distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=ln,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                sensible_range=(0, None),
                working=[f"Area = width × length, so length = {area} ÷ {w} = {ln} m."],
                distractors=a_dist,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=remaining,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{w}*part_a - {p * q}",
                working=[
                    f"Original area = {w} × {ln} = {area} m².",
                    f"Remaining = {area} - {p} × {q} = {remaining} m².",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=cost,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{price}*2*({w} + part_a)",
                working=[
                    "Cutting a rectangle from a corner moves the boundary but does not change its length.",
                    f"Perimeter = 2 × ({w} + {ln}) = {perimeter} m, so cost = {perimeter} × ${price} = ${cost}.",
                ],
                distractors=c_dist,
            ),
        ],
    )
