"""Exam chain for vector motion in a plane: a projectile thrown from a height, apex to landing.

An object is thrown from ``r(0) = x0 i + y0 j`` with ``v(0) = vx i + vy j`` under ``a(t) = -10 j`` (``j`` points up
and the ground is ``j = 0``). Set the way a senior exam sets it:

(a) the time of the greatest height, ``vy / 10``                                       1 d.p.,  1 mark
(b) the greatest height, ``y0 + vy t - 5 t^2`` at that time                            1 d.p.,  2 marks, follow-through (a)
(c) the time of flight: the rise plus the fall from the greatest height                2 d.p.,  2 marks, follow-through (a), (b)
(d) the i-component of position at landing, ``x0 + vx t``                              1 d.p.,  2 marks, follow-through (c)

## Construction

``y0`` is a whole number of metres, ``vy`` is even and ``vx`` is a multiple of 0.5, so (a) and (b) are exact to one
decimal place. The time of flight is the positive root of ``y0 + vy t - 5 t^2``, found from an exact rational
approximation of the square root good to 20 digits, and the fall from the greatest height is worked out
separately as ``sqrt(H / 5)``; the two routes must agree. Each displayed answer is kept clear of a rounding
boundary, and (d) must round the same way from the exact time of flight as from the time rounded to 2 d.p., so
the learner's own route and the exact one agree.

## Distractors

* (a): half of gravity used, the speed not divided by g, the ratio inverted.
* (b): the launch height dropped, half gravity used, the -5t² term forgotten.
* (c): twice the rise (back to launch height), the fall measured from the launch height, the launch-speed term
  dropped from the quadratic formula.
* (d): the position at the greatest height, the launch position dropped, twice the position at the greatest height.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_G = 10
_MARGIN = Fraction(1, 10**6)  # keep every displayed value this far from a rounding boundary

_STEMS = (
    "A {obj} is thrown from {place} with initial position r(0) = {r0} metres and initial velocity "
    "v(0) = {v0} m/s, where i and j point horizontally and vertically upwards. Its acceleration is "
    "a(t) = -10j m/s², and the ground is where the j-component of position is 0.",
    "From {place}, a {obj} is launched with r(0) = {r0} m and v(0) = {v0} m/s (j is vertically upwards). "
    "Only gravity acts, so a(t) = -10j m/s², and level ground is where the j-component of position is 0.",
    "A {obj} leaves {place} at position r(0) = {r0} m with velocity v(0) = {v0} m/s, taking i horizontal and j "
    "vertically up. The acceleration is a(t) = -10j m/s², and the ground is at j = 0.",
)
_OBJECTS = ("ball", "stone", "rock", "javelin")
_PLACES = ("the top of a cliff", "a platform", "the roof of a building", "a bridge", "a tower")


def _sqrt(value: Fraction) -> Fraction:
    """The square root of a non-negative rational, to 20 decimal places."""
    scaled = value * 10**40
    return Fraction(isqrt(scaled.numerator // scaled.denominator), 10**20)


def _clear(value: Fraction, places: int) -> bool:
    """True when `value` lies at least `_MARGIN` from any rounding boundary at `places` decimal places."""
    offset = (value * 10**places) % 1
    return abs(offset - Fraction(1, 2)) / 10**places > _MARGIN


def _vec(x, y) -> str:
    """ "-3i + 26j": a coefficient of 1 is written as the bare unit."""

    def term(value, unit, first):
        magnitude = "" if abs(value) == 1 else f"{abs(value)}"
        sign = ("-" if value < 0 else "") if first else (" - " if value < 0 else " + ")
        return f"{sign}{magnitude}{unit}"

    return term(x, "i", True) + term(y, "j", False)


def _num(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


def _part(prompt, exact, places, marks, working, wrong, follow=None, sensible=True):
    """A decimal-answer part; `wrong` lists (Fraction value, misconception id)."""
    shown = round_half_up(exact, places)
    text = stated_precision_string(shown, places)
    tolerance = stated_precision_tolerance(places)
    cands = []
    for value, tag in wrong:
        rounded = round_half_up(Fraction(value), places)
        cands.append(Distractor(stated_precision_string(rounded, places), tag, exact_value(rounded)))
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(shown),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None) if sensible else None,
        working=working,
        distractors=pick_distractors(text, cands, tolerance=tolerance),
    )
    if follow:
        part["follow_through"] = follow
    return part


@generator("exd_x_r2c_projectile", level=14, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_projectile(rng, tier: str = "exam"):
    """A projectile from a height: time of greatest height, greatest height, time of flight, landing position."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(1000):  # bounded rejection loop
        y0 = rng.randint(4, 45)
        vy = rng.choice(range(4, 31, 2))
        vx = Fraction(rng.choice([v for v in range(-19, 20, 1) if v not in (0,)]), 2)
        x0 = rng.randint(-9, 9)
        t_apex = Fraction(vy, _G)
        height = y0 + Fraction(vy * vy, 4 * 5)  # y0 + vy^2 / 20
        discriminant = Fraction(vy * vy + 20 * y0)
        flight = (vy + _sqrt(discriminant)) / _G
        fall = _sqrt(height / 5)
        x_land = x0 + vx * flight
        x_from_rounded = x0 + vx * round_half_up(flight, 2)
        if (
            _clear(flight, 2)
            and _clear(x_land, 1)
            and round_half_up(x_land, 1) == round_half_up(x_from_rounded, 1)
            and _clear(x_from_rounded, 1)
        ):
            break
    else:  # pragma: no cover
        check(False, "no usable draw")

    # Independent routes: the greatest height from the position function; the flight as rise plus fall.
    check(y0 + vy * t_apex - 5 * t_apex**2 == height, "greatest height disagrees with the position function")
    check(abs(flight - (t_apex + fall)) < Fraction(1, 10**15), "flight time disagrees with rise plus fall")
    check(
        abs(y0 + vy * flight - 5 * flight**2) < Fraction(1, 10**15), "the ball is not on the ground at the flight time"
    )

    r0 = _vec(x0, y0)
    v0 = _vec(float(vx) if vx.denominator != 1 else int(vx), vy)
    obj = rng.choice(_OBJECTS)
    stem = phrase(rng, _STEMS, obj=obj, place=rng.choice(_PLACES), r0=r0, v0=v0)

    part_a = _part(
        f"Find the time at which the {obj} reaches its greatest height, in seconds, correct to 1 decimal place.",
        t_apex,
        1,
        1,
        [f"v_j(t) = {vy} - 10t = 0 gives t = {_num(t_apex)} s."],
        [
            (Fraction(vy, 5), "half-gravity-used"),
            (Fraction(vy), "forgot-to-divide-by-coefficient"),
            (Fraction(10, vy), "inverted-ratio"),
        ],
    )
    part_b = _part(
        f"Hence find the greatest height of the {obj} above the ground, in metres, correct to 1 decimal place.",
        height,
        1,
        2,
        [f"j(t) = {y0} + {vy}t - 5t² at t = {_num(t_apex)}: {_num(height)} m."],
        [
            (height - y0, "dropped-constant-term"),
            (y0 + Fraction(vy * vy, 10), "half-gravity-used"),
            (y0 + vy * t_apex, "dropped-the-squared-term"),
        ],
        follow=f"{y0} + {vy}*part_a - 5*part_a**2",
    )
    part_c = _part(
        f"Hence find the total time of flight, from launch until the {obj} hits the ground, in seconds, "
        "correct to 2 decimal places.",
        flight,
        2,
        2,
        [
            f"The object falls from rest through {_num(height)} m: 5s² = {_num(height)}, so s = {stated_precision_string(round_half_up(fall, 2), 2)} s.",
            f"Total time = {_num(t_apex)} + s = {stated_precision_string(round_half_up(flight, 2), 2)} s.",
        ],
        [
            (2 * t_apex, "symmetric-flight-assumed"),
            (t_apex + _sqrt(Fraction(y0, 5)), "fall-measured-from-launch-height"),
            (_sqrt(discriminant) / _G, "dropped-initial-velocity-term"),
        ],
        follow="part_a + sqrt(part_b/5)",
    )
    x_text = _num(x_land)
    part_d = _part(
        f"Find the i-component of the position of the {obj} when it lands, in metres, correct to 1 decimal place.",
        x_land,
        1,
        2,
        [f"i(t) = {x0} + ({vx})t at t = {stated_precision_string(round_half_up(flight, 2), 2)}: {x_text} m."],
        [
            (x0 + vx * t_apex, "used-apex-time-for-landing"),
            (vx * flight, "dropped-constant-term"),
            (x0 + 2 * vx * t_apex, "symmetric-flight-assumed"),
        ],
        follow=f"{x0} + ({vx.numerator}/{vx.denominator})*part_c",
        sensible=False,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
