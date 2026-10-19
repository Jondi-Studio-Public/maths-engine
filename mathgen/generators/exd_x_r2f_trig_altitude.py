"""Exam source for "apply the cosine and sine ratios to find lengths": a four-part chain on a right-angled triangle.

Triangle ABC has its right angle at B and a hypotenuse AC of known length; one acute angle is given (at A or at
C). The perpendicular BD is dropped from B to AC. The learner works through:

(a) the leg adjacent to the given angle           h cos(theta)                  1 mark
(b) the other leg                                 h sin(theta)                  1 mark
(c) BD, "by writing the area two ways"            leg * leg / h                 2 marks, follows from (a), (b)
(d) the segment of AC beside the given angle      BD / tan(theta)               2 marks, follows from (c)

Every canonical answer is computed from the exact drawn values and rounded once, half up, to one decimal place.
A draw is kept only when every displayed answer is clear of a rounding boundary and when working forward from the
rounded earlier answers (as a learner does) reaches the same displayed answer, so the follow-through route and the
exact route always agree.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor
from ..core import (
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

_ANGLES = (20, 25, 30, 35, 40, 50, 55, 60, 65, 70)
_MARGIN = Fraction(1, 500)  # a displayed answer is at least this far from a rounding boundary
_MIN_GAP = Fraction(1, 4)  # a distractor differs from the key by at least this much

_SETTINGS = (
    # (setting, unit, shortest and longest hypotenuse): each size is one the object could really have
    ("A triangular garden bed ABC has a right angle at B. Its longest side AC is {h} m.", "m", 8, 30),
    ("The end wall of a shed is a right-angled triangle ABC with the right angle at B and AC = {h} m.", "m", 4, 9),
    ("A triangular sail ABC has a right angle at B, and its longest edge AC measures {h} m.", "m", 5, 15),
    ("A triangular metal brace ABC has a right angle at B and its longest side AC is {h} cm.", "cm", 12, 60),
)
_PATHS = (
    "A straight line BD is drawn from B to meet AC at right angles at D.",
    "The perpendicular from B meets AC at D.",
)


def _r1(value: float | Fraction) -> Fraction:
    return round_half_up(Fraction(value).limit_denominator(10**9), 1)


def _clear(value: float) -> bool:
    offset = (Fraction(value).limit_denominator(10**9) * 10) % 1
    return abs(offset - Fraction(1, 2)) / 10 > _MARGIN


def _text(value: Fraction) -> str:
    return stated_precision_string(value, 1)


def _part(prompt, value, marks, working, distractors, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(value),
        answer_text=_text(value),
        answer_latex=latex_answer(_text(value)),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _dist(key: Fraction, candidates, also: Fraction | None = None):
    """Tagged distractors, each at least `_MIN_GAP` from the key (and from `also`, the follow-through route's value),
    deduplicated by the grader's own tolerance."""
    keys = [key] if also is None else [key, also]
    kept = [(tag, v) for tag, v in candidates if v > 0 and all(abs(_r1(v) - k) >= _MIN_GAP for k in keys)]
    tagged = [Distractor(_text(_r1(v)), tag, _r1(v)) for tag, v in kept]
    out = pick_distractors(_text(key), tagged, 3, tolerance=GRADER_TOLERANCE)
    check(len(out) >= 2, "too few distractors")
    return out


@generator(
    "exd_x_r2f_trig_altitude",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_trig_altitude(rng: random.Random, tier: str = "exam") -> dict:
    """Right-angled triangle with the perpendicular to the hypotenuse: two legs, the altitude, a segment."""
    check(tier == "exam", f"unknown tier {tier!r}")
    setting, unit, h_min, h_max = rng.choice(_SETTINGS)
    for _ in attempts(1000, "a draw clear of every rounding boundary"):
        h = rng.randint(h_min, h_max)
        theta = rng.choice(_ANGLES)
        given = rng.choice("AC")
        t = math.radians(theta)
        adj, opp = h * math.cos(t), h * math.sin(t)
        bd = adj * opp / h
        pd = adj * math.cos(t)
        qd = opp * math.sin(t)
        if not all(_clear(v) for v in (adj, opp, bd, pd)):
            continue
        a, b, c, d = _r1(adj), _r1(opp), _r1(bd), _r1(pd)
        break
    # The learner's route: each later part from the rounded earlier answers. It can land one place away from the
    # exact route, and both are accepted, so no distractor may sit on either.
    c_ft = _r1(float(a) * float(b) / h)
    d_ft = _r1(float(c) / math.tan(t))

    other = "C" if given == "A" else "A"
    adj_name = "AB" if given == "A" else "BC"
    opp_name = "BC" if given == "A" else "AB"
    angle_name = f"B{given}{other}"
    words = {"m": "metres", "cm": "centimetres"}[unit]
    stem = f"{setting.format(h=h)} Angle {angle_name} is {theta}°. {rng.choice(_PATHS)}"

    # Independent check of the geometry: BD from the triangle's area, PD from Pythagoras in triangle PBD.
    check(abs(adj * opp - h * bd) < 1e-9, "area route disagrees with BD")
    check(abs(math.sqrt(adj**2 - bd**2) - pd) < 1e-9, "Pythagoras route disagrees with the segment")

    a_dist = _dist(
        a,
        [
            ("swaps-sine-and-cosine", Fraction(opp)),
            ("inverts-trig-ratio", Fraction(h / math.cos(t))),
            ("tan-for-sin-or-cos", Fraction(h * math.tan(t))),
        ],
    )
    b_dist = _dist(
        b,
        [
            ("swaps-sine-and-cosine", Fraction(adj)),
            ("inverts-trig-ratio", Fraction(h / math.sin(t))),
            ("tan-for-sin-or-cos", Fraction(h * math.tan(t))),
        ],
    )
    c_dist = _dist(
        c,
        [
            ("product-of-legs-not-divided", Fraction(adj * opp)),
            ("area-half-kept-one-side", Fraction(bd / 2)),
            ("wrong-segment-of-hypotenuse", Fraction(qd)),
            ("altitude-taken-as-a-leg", Fraction(opp)),
        ],
        c_ft,
    )
    d_dist = _dist(
        d,
        [
            ("inverts-trig-ratio", Fraction(bd / math.cos(t))),
            ("wrong-segment-of-hypotenuse", Fraction(qd)),
            ("inverts-trig-ratio", Fraction(bd * math.tan(t))),
            ("altitude-taken-as-a-leg", Fraction(adj)),
        ],
        d_ft,
    )

    return dict(
        text=stem,
        parts=[
            _part(
                f"Find the length of {adj_name}, in {words}, correct to 1 decimal place.",
                a,
                1,
                [f"{adj_name} = {h} cos({theta}°) = {_text(a)} {unit}."],
                a_dist,
            ),
            _part(
                f"Find the length of {opp_name}, in {words}, correct to 1 decimal place.",
                b,
                1,
                [f"{opp_name} = {h} sin({theta}°) = {_text(b)} {unit}."],
                b_dist,
            ),
            _part(
                f"Hence find the length of BD, in {words}, correct to 1 decimal place.",
                c,
                2,
                [
                    "Area = ½ × AB × BC = ½ × AC × BD, so BD = AB × BC ÷ AC.",
                    f"BD = {_text(a)} × {_text(b)} ÷ {h} ≈ {_text(c)} {unit} (using the unrounded AB and BC).",
                ],
                c_dist,
                follow_through=f"part_a*part_b/{h}",
            ),
            _part(
                f"Hence find the length of {given}D, in {words}, correct to 1 decimal place.",
                d,
                2,
                [
                    f"In triangle {given}BD the angle at {given} is {theta}°, so tan({theta}°) = BD ÷ {given}D.",
                    f"{given}D = {_text(c)} ÷ tan({theta}°) = {_text(d)} {unit}.",
                ],
                d_dist,
                follow_through=f"part_c/tan({theta}*pi/180)",
            ),
        ],
    )
