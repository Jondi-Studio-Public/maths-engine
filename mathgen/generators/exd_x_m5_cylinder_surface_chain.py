"""Exam-tier source for the surface area of familiar solids: a closed cylinder worked backwards.

The stem gives a closed cylinder's radius and its total surface area (including both circular ends), not its
height. The learner must

(a) solve ``S = 2*pi*r*(r + h)`` for the height h (a two-step rearrangement, not a substitution),
(b) use that height for the capacity in millilitres, and
(c) use it again for the area of the curved surface (a paper label that wraps the side).

Built backwards from a radius in whole centimetres and a height in whole tenths, so the height the learner
recovers from the displayed (rounded) surface area is exactly the one drawn.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import VerificationError, attempts, generator, latex_answer, numeric_answer

from .exd_x_m5_support import (
    PI,
    clear_of_boundary,
    display,
    keys_absent,
    rounded,
    tagged_distractors,
)

_OBJECTS = (
    "A closed cylindrical tin of tomatoes",
    "A sealed cylindrical drum of cooking oil",
    "A closed cylindrical milk-powder canister",
    "A closed cylindrical tub of protein powder",
    "A sealed cylindrical paint tin",
)
_FIRST = (
    "{obj} has a radius of {r} cm and a total surface area of {s} cm², counting both circular ends.",
    "{obj} has a base radius of {r} cm. Its total outer surface area, ends included, is {s} cm².",
    "The total surface area of {obj_lower}, top and bottom included, is {s} cm². Its radius is {r} cm.",
)

_A_PROMPT = (
    "Find the height of the cylinder, correct to 1 decimal place.",
    "Calculate the height of the cylinder in centimetres, correct to 1 decimal place.",
)
_B_PROMPT = (
    "Hence find the capacity of the cylinder in millilitres (1 cm³ = 1 mL), correct to the nearest millilitre.",
    "Using your height from (a), find how many millilitres the cylinder holds (1 cm³ = 1 mL), "
    "to the nearest millilitre.",
)
_C_PROMPT = (
    "A paper label wraps once around the curved surface only, with no overlap. "
    "Using your height from (a), find the area of the label, to the nearest square centimetre.",
    "A label covers the curved side only, with no gaps or overlap. "
    "Using your height from (a), find its area to the nearest square centimetre.",
)


@generator("exd_x_m5_cylinder_surface_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_cylinder_surface_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: height of a closed cylinder from its total surface area, then capacity and label area."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a cylinder whose answers are clear of rounding boundaries"):
        r = rng.randint(3, 9)
        h = Fraction(rng.randint(40, 200), 10)
        s_exact = 2 * PI * r * (r + h)
        s_shown = rounded(s_exact, 1)
        h_back = s_shown / (2 * PI * r) - r  # the learner's route, from the displayed area
        volume = PI * r * r * h
        curved = 2 * PI * r * h
        if rounded(h_back, 1) != h or not clear_of_boundary(volume, 0) or not clear_of_boundary(curved, 0):
            continue
        a_text, b_text, c_text = display(h, 1), display(volume, 0), display(curved, 0)
        obj = rng.choice(_OBJECTS)
        first = rng.choice(_FIRST).format(obj=obj, obj_lower=obj[0].lower() + obj[1:], r=r, s=display(s_shown, 1))
        prompts = [rng.choice(_A_PROMPT), rng.choice(_B_PROMPT), rng.choice(_C_PROMPT)]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    s = s_shown
    a_dis = tagged_distractors(
        a_text,
        1,
        [
            (s / (2 * PI * r), None),  # forgot to subtract the radius that the two ends contribute
            ((s - PI * r * r) / (2 * PI * r), None),  # counted only one circular end
            (s / (2 * PI * r * r), None),  # divided by 2*pi*r^2 instead of rearranging
            ((s - 2 * PI * r * r) / (PI * r), None),  # halved the curved-surface divisor
        ],
    )
    b_dis = tagged_distractors(
        b_text,
        0,
        [
            (PI * r * h, None),  # forgot to square the radius
            (PI * r * r * h / 3, None),  # used the cone formula
            (PI * (2 * r) ** 2 * h, None),  # used the diameter as the radius
            (2 * PI * r * (r + h), None),  # gave the surface area instead of the capacity
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        0,
        [
            (PI * r * h, None),  # forgot the factor of 2 in the circumference
            (2 * PI * r * r + 2 * PI * r * h, None),  # gave the whole surface
            (PI * r * r * h, None),  # gave the volume
            (2 * r * h, None),  # used 2*r*h, as if the side were a rectangle of width 2r
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=numeric_answer(h),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"S = 2πr² + 2πrh, so 2πrh = {display(s, 1)} − 2π × {r}².",
                    f"h = {display(s, 1)} ÷ (2π × {r}) − {r} = {a_text} cm (1 d.p.).",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(rounded(volume, 0)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"pi*{r}**2*part_a",
                working=[f"V = πr²h = π × {r}² × {a_text} = {b_text} cm³ = {b_text} mL."],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded(curved, 0)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"2*pi*{r}*part_a",
                working=[f"Curved area = 2πrh = 2 × π × {r} × {a_text} = {c_text} cm²."],
                distractors=c_dis,
            ),
        ],
    )
