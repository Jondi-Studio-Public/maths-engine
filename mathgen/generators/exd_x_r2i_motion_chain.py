"""Exam-tier chain for `senior-recognise-acceleration-as-the-second-derivative-of-position`: from rest to extremes.

A particle moves along a line with cubic position ``s(t) = a t^3 + b t^2 + c t + d``. The chain asks for

(a) the later time at which it is momentarily at rest (velocity zero),
(b) its acceleration at that time, from (a),
(c) the time at which its acceleration is zero (when its velocity is extreme), and
(d) that extreme velocity, from (c).

## Construction: backwards from the rest times

The velocity is built as ``v(t) = 3a (t - t1)(t - t2)`` with whole rest times ``t1 < t2`` an even distance apart.
Then ``b = -3a (t1 + t2) / 2`` and ``c = 3 a t1 t2`` are integers, the acceleration is ``3a (2t - t1 - t2)``, it is
zero at the whole midpoint ``(t1 + t2) / 2``, and the extreme velocity there is ``-3a ((t2 - t1) / 2)^2``. Every
answer is an integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms

from .exd_x_r2i_support import number_part

_THINGS = [
    "a particle moving along a straight line",
    "a runner on a straight path",
    "a cyclist on a straight road",
    "a drone flying along a straight line",
]


@generator("exd_x_r2i_motion_chain", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_motion_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: rest times of a cubic position, acceleration at one, and the extreme velocity."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "rest times with a modest position function"):
        a = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        gap = 2 * rng.randint(1, 4)
        t1 = rng.randint(1, 6)
        t2 = t1 + gap
        if t2 > 11:
            continue
        break
    b = -3 * a * (t1 + t2) // 2
    c = 3 * a * t1 * t2
    d = rng.choice([0, 0, 2, 5, 8, 10, -4])
    mid = (t1 + t2) // 2
    accel_at_t2 = 6 * a * t2 + 2 * b
    vmid = 3 * a * mid * mid + 2 * b * mid + c

    # Independent check from the position itself. For a cubic, the central difference (s(x+1) - s(x-1)) / 2 is
    # s'(x) + a, and the second difference s(x+1) + s(x-1) - 2 s(x) is exactly s''(x).
    def pos(x):
        return a * x**3 + b * x**2 + c * x + d

    def velocity(x):
        return Fraction(pos(x + 1) - pos(x - 1), 2) - a

    def acceleration(x):
        return pos(x + 1) + pos(x - 1) - 2 * pos(x)

    check(velocity(t1) == 0 and velocity(t2) == 0, "rest times disagree with s'(t) = 0")
    check(acceleration(t2) == accel_at_t2, "acceleration at the later rest time disagrees")
    check(acceleration(mid) == 0, "the time of zero acceleration disagrees")
    check(velocity(mid) == vmid == -3 * a * ((t2 - t1) // 2) ** 2, "extreme velocity disagrees")

    function = join_terms([(a, "t^3"), (b, "t^2"), (c, "t"), (d, "")])
    thing = rng.choice(_THINGS)
    stem = (
        f"The position of {thing} at time t minutes (t ≥ 0) is s(t) = {function} metres, measured from a fixed point."
    )
    velocity = join_terms([(3 * a, "t^2"), (2 * b, "t"), (c, "")])
    accel_expr = join_terms([(6 * a, "t"), (2 * b, "")])
    position_at_mid = a * mid**3 + b * mid**2 + c * mid + d
    velocity_at_t2 = 0
    earlier_accel = 6 * a * t1 + 2 * b

    parts = [
        number_part(
            "It is momentarily at rest twice. Find the later of the two times, in minutes, at which its velocity is zero.",
            Fraction(t2),
            0,
            2,
            [f"v(t) = s′(t) = {velocity}.", f"v(t) = 0 gives t = {t1} or t = {t2}; the later is {t2} min."],
            [
                (Fraction(t1), "took-the-wrong-root"),
                (Fraction(mid), "solved-wrong-derivative-equation"),
                (Fraction(t2 - t1), "gave-interval-not-time"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find its acceleration at that time, in m/min².",
            Fraction(accel_at_t2),
            0,
            2,
            [f"a(t) = s″(t) = {accel_expr}.", f"a({t2}) = {accel_at_t2} m/min²."],
            [
                (Fraction(velocity_at_t2), "rest-means-zero-acceleration"),
                (Fraction(3 * a * t2 + b), "halved-second-derivative"),
                (Fraction(-accel_at_t2), "answer-sign-reversed"),
                (Fraction(earlier_accel), "evaluated-at-wrong-point"),
            ],
            follow_through=f"6*({a})*part_a + 2*({b})",
        ),
        number_part(
            "Find the time, in minutes, at which the acceleration is zero.",
            Fraction(mid),
            0,
            1,
            [f"{accel_expr} = 0 gives t = {mid} min."],
            [
                (Fraction(t1), "solved-wrong-derivative-equation"),
                (Fraction(t2), "solved-wrong-derivative-equation"),
                (Fraction(abs(b), 6 * abs(a)), "divided-by-wrong-amount"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find its velocity at that time, in m/min.",
            Fraction(vmid),
            0,
            1,
            [f"v({mid}) = 3({a})({mid})² + 2({b})({mid}) + ({c}) = {vmid} m/min."],
            [
                (Fraction(position_at_mid), "evaluated-function-not-derivative"),
                (Fraction(-vmid), "answer-sign-reversed"),
                (Fraction(6 * a * mid + 2 * b), "evaluated-wrong-derivative"),
            ],
            follow_through=f"3*({a})*part_c**2 + 2*({b})*part_c + ({c})",
        ),
    ]
    return dict(text=stem, parts=parts)
