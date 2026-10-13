"""Exam source of `senior-solve-problems-involving-bearings`: a two-leg journey that does not turn through a right angle.

The practice tiers resolve a journey into north and east legs of a right-angled triangle. An exam question sets a
journey whose legs are given as two compass bearings, so the learner first has to read the angle between the legs
out of the bearings, then solve a triangle that has no right angle, then turn the answer back into a bearing:

(a) the interior angle at the turning point, from the two bearings;
(b) the straight-line distance, by the cosine rule on the angle from (a);
(c) the bearing of the finishing point from the start, by the cosine rule again on (b) and then adding the angle
    at the start to the first bearing in the correct direction.

Parts (b) and (c) follow through, so a wrong angle in (a) still earns (b) and (c) when it is used correctly.

## Construction

The leg lengths, the first bearing, the size of the turn and its direction are drawn. Every displayed answer is
kept clear of a rounding boundary and is checked on the learner's route as well (the distance rounded to one
decimal place and then used for the bearing), so the exact route and the learner's route always agree. A second
route recomputes the finishing position in east and north coordinates and takes the distance and bearing from it.
"""

from __future__ import annotations

import math
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
from ._exd_distinct_parts import distinct_part_answers

_TRAVELLERS = [
    ("A yacht sails", "yacht"),
    ("A hiker walks", "hiker"),
    ("A ship steams", "ship"),
    ("A drone flies", "drone"),
    ("A bushwalker treks", "bushwalker"),
    ("A fishing boat travels", "boat"),
]


def _bearing(b: int) -> str:
    return f"{b:03d}°"


def _frac1(x: float) -> Fraction:
    return round_half_up(Fraction(x), 1)


def _clear(x: float, places: int, margin: float = 0.04) -> bool:
    """True when `x` is further than `margin` (in units of the last place) from a rounding boundary."""
    scaled = x * 10**places
    return abs((scaled % 1) - 0.5) > margin


def _cosine_distance(d1: int, d2: int, angle_deg: float) -> float:
    return math.sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * math.cos(math.radians(angle_deg)))


def _angle_at_start(d1: int, d2: int, ac: float) -> float:
    return math.degrees(math.acos((d1 * d1 + ac * ac - d2 * d2) / (2 * d1 * ac)))


def _mod360(x: float) -> float:
    return x % 360


@generator("exd_x_r3a_bearings_triangle", level=12, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_bearings_triangle(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a journey clear of every rounding boundary"):
        d1 = rng.randint(6, 24)
        d2 = rng.randint(6, 24)
        b1 = rng.randrange(0, 360, 5)
        turn = rng.randrange(40, 145, 5)
        if turn == 90:
            continue  # a right angle: the angle distractors collapse and the triangle is the practice case
        sign = rng.choice([-1, 1])  # +1 turns right (clockwise)
        b2 = (b1 + sign * turn) % 360
        interior = 180 - turn
        ac = _cosine_distance(d1, d2, interior)
        if not _clear(ac, 1):
            continue
        ac1 = float(_frac1(ac))
        angle_a = _angle_at_start(d1, d2, ac)
        bearing = _mod360(b1 + sign * angle_a)
        if not _clear(bearing, 0, 0.08):
            continue
        # The learner's route goes through the distance rounded to 1 d.p.
        bearing_learner = _mod360(b1 + sign * _angle_at_start(d1, d2, ac1))
        key_bearing = int(math.floor(bearing + 0.5)) % 360
        if int(math.floor(bearing_learner + 0.5)) % 360 != key_bearing:
            continue
        if key_bearing in (0, b1, b2):
            continue
        shown = {str(d1), str(d2), str(b1), str(b2), str(b1).zfill(3), str(b2).zfill(3), "72"}
        if {str(interior), stated_precision_string(_frac1(ac), 1), str(key_bearing)} & shown:
            continue  # a key must not appear as a number in the prompt
        break

    # Independent route: east and north coordinates of the finishing point.
    east = d1 * math.sin(math.radians(b1)) + d2 * math.sin(math.radians(b2))
    north = d1 * math.cos(math.radians(b1)) + d2 * math.cos(math.radians(b2))
    check(abs(math.hypot(east, north) - ac) < 1e-9, "coordinate route disagrees on the distance")
    check(abs(_mod360(math.degrees(math.atan2(east, north))) - bearing) < 1e-6, "coordinate route disagrees on bearing")

    ac_text = stated_precision_string(_frac1(ac), 1)
    key_text = str(key_bearing)

    # (a) interior angle at B.
    raw = (b2 - b1) % 360
    a_cands = [
        Distractor(str(turn), "turn-angle-for-interior-angle", turn),
        Distractor(str(360 - turn), "reflex-angle-for-interior-angle", 360 - turn),
        Distractor(str(180 + turn), "turn-angle-added-to-straight-angle", 180 + turn),
        Distractor(str(raw), "bearing-difference-for-interior-angle", raw),
    ]
    a_d = pick_distractors(str(interior), a_cands, 3)

    # (b) distance AC.
    def d1p(x: float):
        return stated_precision_string(_frac1(x), 1)

    wrong_plus = math.sqrt(d1 * d1 + d2 * d2 + 2 * d1 * d2 * math.cos(math.radians(interior)))
    b_cands = [
        Distractor(d1p(math.hypot(d1, d2)), "pythagoras-on-non-right-triangle", _frac1(math.hypot(d1, d2))),
        Distractor(d1p(d1 + d2), "legs-added-for-pythagoras", _frac1(d1 + d2)),
        Distractor(d1p(wrong_plus), "cosine-rule-sign-error", _frac1(wrong_plus)),
        Distractor(
            d1p(_cosine_distance(d1, d2, turn)),
            "turn-angle-for-interior-angle",
            _frac1(_cosine_distance(d1, d2, turn)),
        ),
    ]
    b_d = pick_distractors(ac_text, b_cands, 3, tolerance=GRADER_TOLERANCE)

    # (c) bearing of C from A.
    wrong_way = int(math.floor(_mod360(b1 - sign * angle_a) + 0.5)) % 360
    raw_angle = int(math.floor(angle_a + 0.5))
    back = (key_bearing + 180) % 360
    obtuse_alt = int(math.floor(_mod360(b1 + sign * (180 - angle_a)) + 0.5)) % 360
    c_cands = [
        Distractor(str(wrong_way), "angle-applied-wrong-way-to-bearing", wrong_way),
        Distractor(str(raw_angle), "triangle-angle-not-converted-to-bearing", raw_angle),
        Distractor(str(back), "back-bearing-for-forward-bearing", back),
        Distractor(str(b2), "bearing-of-wrong-point", b2),
        Distractor(str(obtuse_alt), "ambiguous-sine-rule-angle", obtuse_alt),
    ]
    c_d = pick_distractors(key_text, c_cands, 3)
    check(len(a_d) >= 2, f"too few (a) distractors: {a_d}")
    check(len(b_d) >= 2, f"too few (b) distractors: {b_d}")
    check(len(c_d) >= 2, f"too few (c) distractors: {c_d}")
    check(all(abs(Fraction(str(d)) - _frac1(ac)) > GRADER_TOLERANCE for d in b_d), "a (b) distractor is accepted")

    who, thing = rng.choice(_TRAVELLERS)
    stem = (
        f"{who} {d1} km on a bearing of {_bearing(b1)} from A to B, then {d2} km on a bearing of "
        f"{_bearing(b2)} from B to C."
    )
    ft_c = (
        f"({b1} {'+' if sign > 0 else '-'} acos(({d1}**2 + part_b**2 - {d2}**2)/(2*{d1}*part_b))*180/pi)"
        f" - 360*floor(({b1} {'+' if sign > 0 else '-'} acos(({d1}**2 + part_b**2 - {d2}**2)/(2*{d1}*part_b))*180/pi)/360)"
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the size of angle ABC, the angle between the two legs of the {thing}'s journey, in degrees.",
                answer_kind="number",
                answer=interior,
                answer_text=str(interior),
                answer_latex=latex_answer(str(interior)),
                marks=2,
                sensible_range=(0, 180),
                working=[
                    f"The bearing changes from {_bearing(b1)} to {_bearing(b2)}, a turn of {turn}° at B.",
                    f"Angle ABC = 180° - {turn}° = {interior}°.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Using your angle from (a), find the distance AC, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_frac1(ac)),
                answer_text=ac_text,
                answer_latex=latex_answer(ac_text),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"sqrt({d1}**2 + {d2}**2 - 2*{d1}*{d2}*cos(part_a*pi/180))",
                working=[
                    f"AC² = {d1}² + {d2}² - 2 × {d1} × {d2} × cos({interior}°).",
                    f"AC = {ac_text} km (1 d.p.).",
                ],
                distractors=b_d,
            ),
            dict(
                prompt="Using your distance from (b), find the bearing of C from A, to the nearest degree. "
                "Give it as a whole number of degrees (write 072° as 72).",
                answer_kind="number",
                answer=key_bearing,
                answer_text=key_text,
                answer_latex=latex_answer(key_text),
                marks=3,
                sensible_range=(0, 360),
                follow_through=ft_c,
                working=[
                    f"cos A = ({d1}² + AC² - {d2}²) ÷ (2 × {d1} × AC), so angle BAC = {angle_a:.1f}°.",
                    f"C lies to the {'right' if sign > 0 else 'left'} of AB, so the bearing is "
                    f"{b1}° {'+' if sign > 0 else '-'} {angle_a:.1f}° = {key_bearing}°.",
                ],
                distractors=c_d,
            ),
        ],
    )
