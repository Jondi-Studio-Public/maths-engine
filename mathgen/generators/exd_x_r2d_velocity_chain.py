"""Exam chain for vectors in context: an aircraft's velocity relative to the air plus a wind (four parts).

The stem gives the aircraft's velocity through the air and the wind's velocity as ``i, j`` vectors (east and north).

(a) the speed over the ground (add the two vectors, then take the magnitude);
(b) the time to fly a stated distance along the ground track (follow-through ``D / part_a``);
(c) how far north of the start the aircraft is at the end of that flight (follow-through ``q * part_b``);
(d) the bearing of the ground track (independent).

The ground velocity is drawn first from a Pythagorean pair, so its magnitude is a whole number, and the air velocity is
derived by subtracting the wind. The flight time is a multiple of a quarter of an hour and the distance is the ground
speed times that time, so every stated and asked value is an exact decimal of at most two places.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    numeric_answer,
    paren_negative,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (12, 35, 37))
_PLANES = ("A light aircraft", "A charter plane", "A survey plane", "A mail plane", "A rescue plane")
_STEM = (
    "{plane} flies with velocity {air} km/h relative to the air, where i and j are unit vectors pointing east and "
    "north. The wind has velocity {wind} km/h. The plane flies in a straight line over the ground to an airstrip "
    "{dist} km from its starting point.",
    "{plane} has a velocity of {air} km/h relative to the air, with i and j unit vectors pointing east and north. "
    "The wind blows with velocity {wind} km/h. The plane flies directly over the ground to an airstrip "
    "{dist} km from where it took off.",
)


def _fmt(value: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(value, dp), dp)


def _dists(key: Fraction, pairs, dp: int, n: int = 3):
    """Tagged distractors for a number part: `pairs` is [(catalogue id, wrong Fraction value)] in priority order."""
    cands = [Distractor(_fmt(v, dp), tag, exact_value(round_half_up(v, dp))) for tag, v in pairs]
    out = pick_distractors(_fmt(key, dp), cands, n, tolerance=GRADER_TOLERANCE if dp else 0)
    check(len(out) >= 2, "too few distractors")
    return out


def _part(prompt, key: Fraction, dp: int, marks: int, working, dists, ft=None):
    text = _fmt(key, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(round_half_up(key, dp)),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=dists,
    )
    if ft:
        part["follow_through"] = ft
    return part


def _bearing(p: int, q: int) -> Fraction:
    """Whole-degree bearing (clockwise from north) of a ground track with q > 0, as an exact-enough Fraction."""
    return Fraction(math.degrees(math.atan2(p, q)) % 360)


@generator("exd_x_r2d_velocity_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2d_velocity_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Ground speed, flight time, northward distance and bearing for a plane in a wind."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "a velocity chain with a clean bearing"):
        a, b, h = rng.choice(_TRIPLES)
        k = rng.randint(1, 12)
        east, north_c = (a, b) if rng.random() < 0.5 else (b, a)
        p, q, s = east * k * rng.choice((-1, 1)), north_c * k, h * k
        if not 110 <= s <= 400:
            continue
        wi, wj = rng.randint(2, 12) * 5 * rng.choice((-1, 1)), rng.randint(2, 12) * 5 * rng.choice((-1, 1))
        ai, aj = p - wi, q - wj
        if min(abs(ai), abs(aj)) < 5:
            continue
        air = math.hypot(ai, aj)
        if not 100 <= air <= 450 or abs(air - s) < 5:
            continue
        t = Fraction(rng.randint(5, 19), 4)
        dist = s * t
        if dist.denominator != 1:
            continue
        bearing = _bearing(p, q)
        if abs(float(bearing) % 1 - 0.5) >= 0.1:  # well clear of a rounding boundary
            break
    dist = int(dist)
    north = q * t
    check(p * p + q * q == s * s, "ground velocity is not a Pythagorean pair")
    check(ai + wi == p and aj + wj == q, "air velocity plus wind is not the ground velocity")
    check(dist == s * t, "distance and time disagree")
    bearing_whole = int(round_half_up(bearing, 0))
    check(0 < bearing_whole < 360, "bearing out of range")

    stem = rng.choice(_STEM).format(
        plane=rng.choice(_PLANES),
        air=join_terms([(ai, "i"), (aj, "j")]),
        wind=join_terms([(wi, "i"), (wj, "j")]),
        dist=dist,
    )
    air_i = Fraction(air).limit_denominator(1000)
    wind_f = Fraction(math.hypot(wi, wj)).limit_denominator(1000)

    a_pairs = [
        ("air-speed-for-ground-speed", air_i),  # the aircraft's speed through the air only
        ("added-magnitudes-not-vectors", air_i + wind_f),  # |air| + |wind|
        ("subtracted-instead-of-added-vectors", Fraction(math.hypot(ai - wi, aj - wj)).limit_denominator(1000)),
    ]
    b_pairs = [
        ("air-speed-for-ground-speed", Fraction(dist) / air_i),  # divided by the air speed
        ("inverted-rate-formula", Fraction(s, dist)),  # speed / distance
        ("added-magnitudes-not-vectors", Fraction(dist) / (air_i + wind_f)),
    ]
    c_pairs = [
        ("used-wrong-component", abs(p) * t),  # the eastward distance
        ("used-wrong-component", Fraction(dist)),  # the whole track length
        ("air-speed-for-ground-speed", aj * t),  # the aircraft's own northward speed, wind ignored
        ("inverted-rate-formula", q / t),
    ]
    theta = Fraction(math.degrees(math.atan(abs(p) / q)))
    d_pairs = [
        ("angle-from-wrong-axis", Fraction(90) - theta),  # measured from east
        ("angle-from-wrong-axis", theta if p < 0 else 360 - theta),  # swung the wrong way
        ("angle-from-wrong-axis", Fraction(math.degrees(math.atan(q / abs(p))))),
    ]
    d_pairs = [(tag, v) for tag, v in d_pairs if 0 < round_half_up(v, 0) < 360]
    t_key = Fraction(dist, s)
    return dict(
        text=stem,
        parts=[
            _part(
                "Find the speed of the plane over the ground, in km/h.",
                Fraction(s),
                0,
                3,
                [
                    f"Ground velocity = air velocity + wind = {join_terms([(p, 'i'), (q, 'j')])} km/h.",
                    f"Speed = √({paren_negative(p)}² + {paren_negative(q)}²) = {s} km/h.",
                ],
                _dists(Fraction(s), a_pairs, 0),
            ),
            _part(
                "Hence find how long the flight takes, in hours, correct to 2 decimal places.",
                t_key,
                2,
                2,
                [f"Time = distance ÷ speed = {dist} ÷ {s} = {_fmt(t_key, 2)} hours."],
                _dists(t_key, b_pairs, 2),
                ft=f"{dist}/part_a",
            ),
            _part(
                "Using your time from (b), find how far north of its starting point the plane is when it lands, "
                "in km, correct to 2 decimal places.",
                north,
                2,
                2,
                [
                    f"Northward speed = {q} km/h, so the northward distance = {q} × {_fmt(t_key, 2)} = {_fmt(north, 2)} km."
                ],
                _dists(north, c_pairs, 2),
                ft=f"{q}*part_b",
            ),
            dict(
                prompt="Find the bearing of the plane's path over the ground, in degrees clockwise from north, "
                "to the nearest degree.",
                answer_kind="number",
                answer=bearing_whole,
                answer_text=str(bearing_whole),
                answer_latex=latex_answer(str(bearing_whole)),
                marks=2,
                working=[
                    f"The ground velocity is {join_terms([(p, 'i'), (q, 'j')])}, so the angle from north is "
                    f"tan⁻¹({abs(p)}/{q}).",
                    f"Bearing = {bearing_whole}°.",
                ],
                distractors=_dists(Fraction(bearing_whole), d_pairs, 0),
            ),
        ],
    )
