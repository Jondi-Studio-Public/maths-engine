"""Exam-tier source for areas of regular shapes: a rectangle with a semicircle on one side, worked backwards.

The stem gives the width of a rectangle, the semicircle built on that width as its diameter, and the area of
the whole shape. The learner must

(a) find the area of the semicircle,
(b) take it from the whole area to get the area of the rectangle, and
(c) divide by the width to get the rectangle's other side.

Built backwards from a width in whole metres and a rectangle side in whole tenths. The whole area is shown
rounded to 1 decimal place, so a draw is kept only when every route a learner can reasonably take (rounding
the semicircle first or not, rounding the rectangle first or not) reaches the same displayed answers.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import VerificationError, attempts, generator, latex_answer, numeric_answer

from .exd_x_m5_support import PI, clear_of_boundary, display, keys_absent, rounded, tagged_distractors

_OBJECTS = (
    ("An arched window is a rectangle with a semicircle on top.", "window", "m"),
    ("The floor of a stage is a rectangle with a semicircular end.", "stage", "m"),
    ("A garden bed is a rectangle with a semicircle added to one of its sides.", "bed", "m"),
    ("A swimming pool is a rectangle with a semicircular end.", "pool", "m"),
    ("A tunnel entrance is a rectangle topped by a semicircle.", "entrance", "m"),
)
_FIRST = (
    "{obj} The width of the rectangle is {w} {u}, which is also the diameter of the semicircle. "
    "The whole {thing} has an area of {total} {u}².",
    "{obj} The rectangle is {w} {u} wide, and that width is the diameter of the semicircle. "
    "Altogether the {thing} covers {total} {u}².",
    "{obj} Its width is {w} {u} (the diameter of the semicircle) and its total area is {total} {u}².",
)
_A_PROMPT = (
    "Find the area of the semicircle, correct to 1 decimal place.",
    "Calculate the area of the semicircular part, in {u}², correct to 1 decimal place.",
)
_B_PROMPT = (
    "Hence find the area of the rectangular part, correct to 1 decimal place.",
    "Using your answer to (a), find the area of the rectangle in {u}², correct to 1 decimal place.",
)
_C_PROMPT = (
    "Using your answer to (b), find the other side of the rectangle in {u}, correct to 1 decimal place.",
    "Hence find the length of the rectangle, in {u}, correct to 1 decimal place. Use your answer to (b).",
)


@generator("exd_x_m5_rectangle_semicircle_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_rectangle_semicircle_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: semicircle area, then the rectangle's area, then its side, from the whole area."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a shape whose answers agree along every rounding route"):
        w = rng.randint(4, 16)
        h = Fraction(rng.randint(20, 120), 10)
        semi = PI * w * w / 8
        total = rounded(w * h + semi, 1)
        semi1 = rounded(semi, 1)
        rect = total - semi1  # on the 1 d.p. grid: the learner who rounds (a) first
        rect_exact = rounded(total - semi, 1)  # the learner who does not round (a)
        side = rounded(rect / w, 1)
        side_alt = rounded((total - semi) / w, 1)
        if not (
            rect == rect_exact
            and side == side_alt
            and side == rounded(rect_exact / w, 1)
            and clear_of_boundary(semi, 1)
            and clear_of_boundary((total - semi) / w, 1)
            and rect > 0
        ):
            continue
        a_text, b_text, c_text = display(semi1, 1), display(rect, 1), display(side, 1)
        obj, thing, unit = rng.choice(_OBJECTS)
        first = rng.choice(_FIRST).format(obj=obj, w=w, thing=thing, total=display(total, 1), u=unit)
        prompts = [rng.choice(p).format(u=unit) for p in (_A_PROMPT, _B_PROMPT, _C_PROMPT)]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    a_dis = tagged_distractors(
        a_text,
        1,
        [
            (PI * w * w / 4, None),  # the area of the whole circle
            (PI * w * w / 2, None),  # took the diameter as the radius
            (PI * w / 2, None),  # used the length of the semicircular arc
            (PI * w * w / 16, None),  # halved the diameter twice
        ],
    )
    b_dis = tagged_distractors(
        b_text,
        1,
        [
            (total + semi1, None),  # added the semicircle instead of taking it away
            (total - 2 * semi1, None),  # took away a whole circle
            (total, None),  # forgot to remove the semicircle
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        1,
        [
            (rect / (Fraction(w, 2)), None),  # divided by the radius instead of the width
            (rect, None),  # stopped at the area
            (total / w, None),  # divided the whole area by the width
            (rect + semi1, None),
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=numeric_answer(semi1),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[f"The radius is {w} ÷ 2. Semicircle area = ½ × π × ({w}/2)² = {a_text}."],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(rect),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{display(total, 1)} - part_a",
                working=[f"Rectangle = {display(total, 1)} − {a_text} = {b_text}."],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(side),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b/{w}",
                working=[f"Side = {b_text} ÷ {w} = {c_text}."],
                distractors=c_dis,
            ),
        ],
    )
