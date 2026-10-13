"""Exam source for `f10-irregular-composite-area`: the paintable area of a gabled wall, then the paint needed.

A rectangular wall carries a triangular gable and has some identical windows. The learner finds the outline's
area (rectangle plus triangle), removes the windows, then works out litres of paint for a number of coats at a
stated coverage.

(a) area of the wall including the gable, (b) paintable area after the windows, (c) litres for the coats.
Parts (b) and (c) carry follow-through. Built backwards from the dimensions, so every answer is exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, latex_answer
from mathgen.generators.exd_x_m2_support import display, keys_absent, tagged_distractors

_COVERAGE = [4, 5, 8, 10]
_WALLS = ["end wall of a shed", "front wall of a cottage", "end wall of a barn", "wall of a beach hut"]


@generator("exd_x_m2_wall_paint", level=8, presets={"exam": dict(tier="exam")})
def exd_x_m2_wall_paint(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: rectangle plus gable, minus windows, then litres of paint."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw whose answers are not already in the text"):
        w = rng.choice([6, 8, 10, 12, 14, 16])
        ht = rng.randint(2, 5)
        gable = rng.randint(1, 4)
        n_win = rng.randint(1, 3)
        wp, wq = rng.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
        cover = rng.choice(_COVERAGE)
        coats = rng.choice([2, 3])
        outline = w * ht + w * gable // 2
        paintable = outline - n_win * wp * wq
        litres = Fraction(coats * paintable, cover)
        win_word = "window" if n_win == 1 else "windows"
        each = "" if n_win == 1 else "each "
        are = "is" if n_win == 1 else "are"
        stem = (
            f"The {rng.choice(_WALLS)} is a rectangle {w} m wide and {ht} m high, with a triangular gable on top. "
            f"The gable has the same {w} m base and a height of {gable} m. "
            f"It has {n_win} rectangular {win_word}, {each}{wp} m by {wq} m, that {are} not painted. "
            f"One litre of paint covers {cover} m² and the wall needs {coats} coats."
        )
        prompts = [
            "Find the area of the whole wall, including the gable, in square metres.",
            "Hence find the area to be painted, in square metres.",
            "Find the litres of paint needed, correct to 2 decimal places.",
        ]
        a_text, b_text, c_text = str(outline), str(paintable), display(litres, 2)
        if keys_absent(stem, prompts, [a_text, b_text, c_text]):
            break
    check(outline == w * ht + Fraction(w * gable, 2), "outline disagrees with rectangle plus triangle")

    a_dist = tagged_distractors(
        a_text,
        0,
        [
            (w * ht, None),
            (w * ht + w * gable, None),
            (w * (ht + gable), None),
            (w * ht + gable, None),
        ],
    )
    b_dist = tagged_distractors(
        b_text,
        0,
        [
            (outline, None),
            (outline + n_win * wp * wq, None),
            (w * ht - n_win * wp * wq, None),
            (outline - wp * wq, None),
        ],
    )
    c_dist = tagged_distractors(
        c_text,
        2,
        [
            (Fraction(paintable, cover), None),
            (Fraction(coats * outline, cover), None),
            (coats * paintable * cover, None),
            (Fraction(paintable * cover, coats), None),
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
                answer=outline,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Rectangle: {w} × {ht} = {w * ht} m². Triangle: ½ × {w} × {gable} = {w * gable // 2} m².",
                    f"Total = {outline} m².",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=paintable,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a - {n_win * wp * wq}",
                working=[
                    f"Windows: {n_win} × {wp} × {wq} = {n_win * wp * wq} m².",
                    f"{outline} - {n_win * wp * wq} = {paintable} m².",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=litres,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{coats}*part_b/{cover}",
                working=[
                    f"{coats} coats of {paintable} m² = {coats * paintable} m².",
                    f"{coats * paintable} ÷ {cover} = {c_text} L.",
                ],
                distractors=c_dist,
            ),
        ],
    )
