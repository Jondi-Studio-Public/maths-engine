"""Exam tier for "n-th roots of unity and their location on the unit circle": a geometry chain.

The solutions of z^n = 1 are plotted on an Argand diagram with a stated scale, and two of them are named by their
position in the list (counting from z_0 = 1). Part (a) finds the angle at the origin between the two points: the
indices have to be turned into a step count around the circle and the smaller of the two angles taken. Part (b)
uses it to find the distance between the points, and part (c) finds the area of the triangle they make with the
origin. Parts (b) and (c) carry `follow_through`. The practice tiers ask for the argument of a single root or a
count; this question uses the equal spacing of the roots to do geometry with two of them.

## Construction

`n`, the two indices and the scale are drawn, and every answer is worked out from exact trigonometric values with
floats only for display. A draw is kept only when the answers computed from the exact angle and from the learner's
route (the angle rounded to 1 decimal place, then used) agree at the displayed precision and sit clear of a
rounding boundary.
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
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_STEMS = (
    "The solutions of z^{n} = 1 are plotted on an Argand diagram in which 1 unit represents {u} cm. In order of "
    "increasing argument, starting from z_0 = 1, they are z_0, z_1, ..., z_{last}. The points P and Q represent "
    "z_{j} and z_{k}.",
    "On an Argand diagram drawn with a scale of {u} cm to 1 unit, the {n} solutions of z^{n} = 1 are marked, "
    "starting at z_0 = 1 and moving anticlockwise through z_1, z_2, ..., z_{last}. P marks z_{j} and Q marks z_{k}.",
    "A designer plots the roots of z^{n} = 1 on axes where 1 unit is {u} cm, labelling them z_0 = 1, z_1, ..., "
    "z_{last} in anticlockwise order. A straight rod joins P = z_{j} to Q = z_{k}, and O is the origin.",
)


def _round(value: float, places: int) -> Fraction:
    return round_half_up(Fraction(value).limit_denominator(10**12), places)


def _clear(value: float, places: int) -> bool:
    scaled = abs(value) * 10**places
    return abs(scaled - math.floor(scaled) - 0.5) > 0.05


@generator("exd_x_m4_roots_of_unity_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_roots_of_unity_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Angle between two roots, distance between them, area of the triangle with the origin. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a pair of roots with consistent rounding"):
        n = rng.randint(5, 30)
        j, k = rng.sample(range(n), 2)
        step = min((k - j) % n, (j - k) % n)
        if 2 * step == n:
            continue
        u = rng.choice([2, 3, 4, 5, 6, 8, 10])
        theta = 360 * step / n
        a = _round(theta, 1)
        a_f = float(a)
        d_exact = 2 * u * math.sin(math.radians(theta) / 2)
        d_route = 2 * u * math.sin(math.radians(a_f) / 2)
        s_exact = u * u * math.sin(math.radians(theta)) / 2
        s_route = u * u * math.sin(math.radians(a_f)) / 2
        if not (_clear(theta, 1) and _clear(d_exact, 2) and _clear(s_exact, 2)):
            continue
        if _round(d_exact, 2) != _round(d_route, 2) or _round(s_exact, 2) != _round(s_route, 2):
            continue
        stem = rng.choice(_STEMS).format(n=n, u=u, last=n - 1, j=j, k=k)
        a_text = stated_precision_string(a, 1)
        d_text = stated_precision_string(_round(d_exact, 2), 2)
        s_text = stated_precision_string(_round(s_exact, 2), 2)
        if all(t not in stem.split() for t in (a_text, d_text, s_text)):
            break
    rad = math.radians(theta)

    def deg(v: float) -> Distractor:
        shown = _round(v, 1)
        return Distractor(stated_precision_string(shown, 1), None, shown)

    def two(v: float) -> Distractor | None:
        shown = _round(v, 2)
        if shown <= 0:
            return None
        return Distractor(stated_precision_string(shown, 2), None, shown)

    def keep(options):
        return [o for o in options if o is not None]

    a_d = pick_distractors(
        a_text,
        [deg(360 / n), deg(180 * step / n), deg(360 - theta), deg(theta / 2)],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_d = pick_distractors(
        d_text,
        keep([two(u * rad), two(2 * u * math.sin(rad)), two(u * math.sin(rad / 2)), two(2 * u * math.cos(rad / 2))]),
        3,
        tolerance=GRADER_TOLERANCE,
    )
    c_d = pick_distractors(
        s_text,
        keep(
            [
                two(u * u * math.sin(rad)),
                two(u * u * rad / 2),
                two(u * u * math.cos(rad) / 2),  # None for an obtuse angle, where the cosine is negative
                two(u * u * math.sin(rad / 2) / 2),
            ]
        ),
        3,
        tolerance=GRADER_TOLERANCE,
    )
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the size of angle POQ, the smaller angle between OP and OQ, in degrees, correct to "
                "1 decimal place.",
                answer_kind="number",
                answer=float(a),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, 180),
                working=[
                    f"The {n} roots are equally spaced, 360° ÷ {n} apart.",
                    f"Indices {j} and {k} are {step} step{'s' if step != 1 else ''} apart the short way round, so angle POQ = {step} × 360° ÷ {n} "
                    f"= {a_text}°.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Hence find the length of the rod PQ, in centimetres, correct to 2 decimal places.",
                answer_kind="number",
                answer=float(_round(d_exact, 2)),
                answer_text=d_text,
                answer_latex=latex_answer(d_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"2*{u}*sin(part_a*pi/360)",
                working=[f"OP = OQ = {u} cm, so PQ = 2 × {u} × sin({a_text}° ÷ 2) = {d_text} cm."],
                distractors=b_d,
            ),
            dict(
                prompt="Hence find the area of triangle OPQ, in square centimetres, correct to 2 decimal places.",
                answer_kind="number",
                answer=float(_round(s_exact, 2)),
                answer_text=s_text,
                answer_latex=latex_answer(s_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{u * u}*sin(part_a*pi/180)/2",
                working=[f"Area = 1/2 × {u} × {u} × sin({a_text}°) = {s_text} cm²."],
                distractors=c_d,
            ),
        ],
    )
