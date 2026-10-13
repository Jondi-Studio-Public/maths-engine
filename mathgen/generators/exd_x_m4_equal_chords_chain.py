"""Exam tier for "chords of equal length subtend equal angles at the centre": a three-part chain.

A circle of known radius has two, three or four consecutive chords of the same stated length (AB, BC, CD, ...).
Part (a) finds the angle each chord subtends at the centre (an inverse sine, to the nearest degree). Part (b) uses
the equal-chords theorem to find the angle the whole run of chords subtends, and part (c) finds the length of the
single chord joining the first and last points. The practice tiers state the equal angle or length directly; here
the angle has to be produced from lengths first, then used again for a different chord. Parts (b) and (c) carry
`follow_through`.

## Construction: draw the angle first

The angle each chord subtends is drawn as a whole number of degrees, the radius as a multiple of 0.5, and the
chord length shown is the true length `2 r sin(theta / 2)` rounded to 2 decimal places. The angle recovered from the
displayed length rounds back to the drawn angle (checked), and the final chord length is computed from the whole
number of degrees and kept clear of a rounding boundary.
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
    exact_decimal_string,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_ANGLE_RANGE = {2: (24, 88), 3: (20, 58), 4: (20, 42)}
_NAMES = {2: "AB and BC", 3: "AB, BC and CD", 4: "AB, BC, CD and DE"}
_END = {2: "C", 3: "D", 4: "E"}
_STEMS = (
    "In a circle with centre O and radius {r} {u}, the chords {names} are equal in length, each {c} {u} long.",
    "O is the centre of a circle of radius {r} {u}. The chords {names} each have length {c} {u}.",
    "A circular lookout platform has centre O and radius {r} {u}. Rails {names} are chords of the circle, each "
    "{c} {u} long.",
)
_UNITS = ("cm", "m", "cm")


def _chord(r: Fraction, degrees: int) -> float:
    return 2 * float(r) * math.sin(math.radians(degrees) / 2)


def _clear(value: float, places: int) -> bool:
    scaled = abs(value) * 10**places
    return abs(scaled - math.floor(scaled) - 0.5) > 0.06


def _deg(value: float) -> Fraction:
    return Fraction(round(value))


@generator("exd_x_m4_equal_chords_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_equal_chords_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Angle at the centre from the chord, the angle for the run, then the joining chord. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a radius and angle clear of rounding boundaries"):
        n = rng.choice([2, 3, 4])
        lo, hi = _ANGLE_RANGE[n]
        theta = rng.randint(lo, hi)
        r = Fraction(rng.randint(10, 40), 2)
        shown = round_half_up(Fraction(_chord(r, theta)).limit_denominator(10**9), 2)
        back = 2 * math.degrees(math.asin(float(shown) / (2 * float(r))))
        total = n * theta
        joining = _chord(r, total)
        if abs(back - theta) >= 0.2 or not _clear(joining, 1) or total >= 180:
            continue
        unit = rng.choice(_UNITS)
        c_text = stated_precision_string(shown, 2)
        r_text = exact_decimal_string(r, 1)
        end = _END[n]
        stem = rng.choice(_STEMS).format(r=r_text, u=unit, names=_NAMES[n], c=c_text)
        j_text = f"{math.floor(joining * 10 + 0.5) / 10:.1f}"
        a_text, b_text = str(theta), str(total)
        if j_text not in stem and a_text not in stem.split() and b_text not in stem.split():
            break

    cf = float(shown)
    rf = float(r)
    a_wrong = [theta / 2, 180 - theta, math.degrees(cf / rf)]
    if cf <= rf:
        a_wrong.append(2 * math.degrees(math.asin(cf / rf)))
    b_wrong = [theta, 360 - total, total / 2, (n + 1) * theta]
    tot_rad = math.radians(total)
    c_wrong = [n * cf, 2 * rf * math.sin(tot_rad), rf * tot_rad, 2 * rf * math.cos(tot_rad / 2)]

    def deg_options(values):
        return [Distractor(str(round(v)), None, Fraction(round(v))) for v in values if round(v) > 0]

    def len_options(values):
        return [
            Distractor(f"{math.floor(v * 10 + 0.5) / 10:.1f}", None, Fraction(math.floor(v * 10 + 0.5), 10))
            for v in values
            if v > 0
        ]

    a_d = pick_distractors(a_text, deg_options(a_wrong), 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, deg_options(b_wrong), 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(j_text, len_options(c_wrong), 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the angle that each of these chords subtends at O, correct to the nearest degree.",
                answer_kind="number",
                answer=theta,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, 360),
                working=[
                    "Split the isosceles triangle in two: sin(θ/2) = (c ÷ 2) ÷ r.",
                    f"θ = 2 × sin⁻¹({c_text} ÷ {2 * r}) = {theta}° (nearest degree).",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Hence find the size of angle AO{end}, in degrees, using your answer to (a).",
                answer_kind="number",
                answer=total,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, 360),
                follow_through=f"{n}*part_a",
                working=[f"Equal chords subtend equal angles, so angle AO{end} = {n} × {theta}° = {total}°."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Find the length of the chord A{end}, correct to 1 decimal place, using your answer to (b).",
                answer_kind="number",
                answer=float(j_text),
                answer_text=j_text,
                answer_latex=latex_answer(j_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"2*{r_text}*sin(part_b*pi/360)",
                working=[f"A{end} = 2r sin(θ/2) = 2 × {r_text} × sin({total}° ÷ 2) = {j_text} {unit} (1 d.p.)."],
                distractors=c_d,
            ),
        ],
    )
