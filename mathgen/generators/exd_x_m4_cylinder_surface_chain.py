"""Exam tier for "calculate the surface areas of standard three-dimensional objects": a cylinder chain.

The stem gives the volume and height of a cylindrical container (closed, or open at the top). Part (a) works back
to the radius (a square root, correct to 1 decimal place), part (b) uses it for the surface area, and part (c)
finds what percentage of that surface is the curved side. Parts (b) and (c) carry `follow_through`, so a learner
whose radius is wrong still earns (b) and (c) for using their own radius correctly.

## Construction: draw the radius first

The radius is drawn as a whole number of tenths, the height as a whole number of centimetres, and the volume shown
is the true volume rounded to the nearest cubic centimetre. Every canonical answer is computed from the exact radius
with true pi and rounded once, at display, at least `_MARGIN` away from a rounding boundary, so a learner who works
back from the displayed volume and rounds the radius gets the same answers.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

PI = Fraction(math.pi)
_MARGIN = Fraction(1, 1000)

_STEMS = {
    "closed": (
        "A closed cylindrical tin has a volume of {v} cm³ and a height of {h} cm.",
        "A sealed cylindrical water tank model has a capacity of {v} cm³ and is {h} cm tall.",
        "A cylindrical pet-food can, closed at both ends, holds {v} cm³ and stands {h} cm high.",
    ),
    "open": (
        "A cylindrical bucket with no lid has a volume of {v} cm³ and a height of {h} cm.",
        "An open cylindrical planter holds {v} cm³ of soil and is {h} cm deep.",
        "A cylindrical pencil holder, open at the top and closed at the base, has a capacity of {v} cm³ "
        "and a height of {h} cm.",
    ),
}


def _dp1(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


def _clear(value: Fraction) -> bool:
    return abs((value * 10) % 1 - Fraction(1, 2)) > _MARGIN * 10


def _txt(value: Fraction) -> str:
    return stated_precision_string(_dp1(value), 1)


def _surface(kind: str, r: Fraction, h: int) -> Fraction:
    return 2 * PI * r * (r + h) if kind == "closed" else PI * r * (r + 2 * h)


def _curved_percent(kind: str, r: Fraction, h: int) -> Fraction:
    return 100 * 2 * PI * r * h / _surface(kind, r, h)


@generator("exd_x_m4_cylinder_surface_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_cylinder_surface_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Radius from the volume, then the surface area, then the share that is curved. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a radius and height clear of rounding boundaries"):
        kind = rng.choice(["closed", "open"])
        r10, h = rng.randint(25, 95), rng.randint(5, 25)
        if r10 % 10 == 0:
            continue
        r = Fraction(r10, 10)
        volume = round_half_up(PI * r * r * h)
        r_back = math.sqrt(float(volume) / (math.pi * h))
        if round(r_back, 1) != float(r) or abs(r_back - float(r)) > 0.03:
            continue
        sa, pc = _surface(kind, r, h), _curved_percent(kind, r, h)
        if _clear(sa) and _clear(pc):
            break
    v_text = str(int(volume))
    stem = rng.choice(_STEMS[kind]).format(v=v_text, h=h)
    r_text, sa_text, pc_text = stated_precision_string(r, 1), _txt(sa), _txt(pc)
    check(all(t not in stem for t in (r_text, sa_text, pc_text)), "an answer appears in the stem")

    sqrt_v = Fraction(math.sqrt(float(volume) / h))
    a_wrong = [
        Distractor(_txt(Fraction(float(volume) / (math.pi * h))), "area-without-square-root", _dp1(r * r)),
        Distractor(_txt(sqrt_v), "area-root-without-pi", _dp1(sqrt_v)),
        Distractor(_txt(2 * r), "diameter-for-radius", _dp1(2 * r)),
    ]
    curved = 2 * PI * r * h
    base = PI * r * r
    other = _surface("open" if kind == "closed" else "closed", r, h)
    b_wrong = [curved, curved + base if kind == "closed" else curved + 2 * base, other, PI * r * r * h, 4 * PI * r * r]
    c_wrong = [
        100 * r / (r + h) if kind == "closed" else 100 * r / (r + 2 * h),  # the ends' share instead
        100 * h / (2 * r + h),  # diameter for radius
        _curved_percent("open" if kind == "closed" else "closed", r, h),  # the other container's surface
        100 * curved / (PI * r * r * h),
    ]
    a_d = pick_distractors(r_text, a_wrong, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(
        sa_text, [Distractor(_txt(v), None, _dp1(v)) for v in b_wrong], 3, tolerance=GRADER_TOLERANCE
    )
    c_d = pick_distractors(
        pc_text, [Distractor(_txt(v), None, _dp1(v)) for v in c_wrong], 3, tolerance=GRADER_TOLERANCE
    )
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")

    if kind == "closed":
        ft_b, ft_c = "2*pi*part_a*(part_a+%d)" % h, "100*%d/(part_a+%d)" % (h, h)
        sa_work = f"Closed: A = 2πr² + 2πrh = 2π × {r_text} × ({r_text} + {h}) = {sa_text} cm² (1 d.p.)."
    else:
        ft_b, ft_c = "pi*part_a*(part_a+%d)" % (2 * h), "200*%d/(part_a+%d)" % (h, 2 * h)
        sa_work = f"Open top: A = πr² + 2πrh = π × {r_text} × ({r_text} + {2 * h}) = {sa_text} cm² (1 d.p.)."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the radius of the cylinder, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(r),
                answer_text=r_text,
                answer_latex=latex_answer(r_text),
                marks=2,
                sensible_range=(0, None),
                working=[f"V = πr²h, so r² = {v_text} ÷ ({h}π).", f"r = {r_text} cm (1 d.p.)."],
                distractors=a_d,
            ),
            dict(
                prompt="Hence find the total outside surface area of the container, in cm², correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_dp1(sa)),
                answer_text=sa_text,
                answer_latex=latex_answer(sa_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=ft_b,
                working=[sa_work],
                distractors=b_d,
            ),
            dict(
                prompt="What percentage of the total outside surface area is the curved side? "
                "Give your answer correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_dp1(pc)),
                answer_text=pc_text,
                answer_latex=latex_answer(pc_text),
                marks=2,
                sensible_range=(0, 100),
                follow_through=ft_c,
                working=[f"Curved side = 2πrh; percentage = 2πrh ÷ total × 100 = {pc_text}% (1 d.p.)."],
                distractors=c_d,
            ),
        ],
    )
