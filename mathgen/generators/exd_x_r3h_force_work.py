"""Exam tier for constant and non-constant force: a three-part work-energy chain.

A body of mass m moves along a line with initial speed u under a force F(x) = kx that grows with position.

(a) the work the force does from x = 0 to x = X, which is the integral of kx, so kX^2/2 joules;
(b) the speed at x = X, from the work-energy relation v^2 = u^2 + 2W/m;
(c) the speed at x = X when a constant resistive force R also acts over the same distance.

Built backwards from whole-number speeds u, v, w so every key is exact: k = m(v^2 - u^2)/X^2 and
R = m(v^2 - w^2)/(2X) must be whole numbers. Parts (b) and (c) follow through from the learner's earlier answers.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_x_r3h_kit import number_part

_CONTEXTS = [
    "A trolley of mass {m} kg is pushed along a smooth horizontal track by a force that grows with distance",
    "A crate of mass {m} kg is dragged across a smooth floor by a winch whose pull grows as the crate moves",
    "A sled of mass {m} kg on frictionless ice is pulled by an elastic cord that stretches as the sled moves away",
]


def _root(value: Fraction) -> Fraction:
    return Fraction(math.sqrt(value)) if value >= 0 else Fraction(0)


@generator("exd_x_r3h_force_work", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3h_force_work(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(5000, "whole-number force constants and speeds"):
        m = rng.choice((2, 3, 4, 5, 6, 8, 10, 12, 15, 20))
        big_x = rng.randint(2, 10)
        u = rng.randint(1, 10)
        v = rng.randint(u + 2, 20)
        w = rng.randint(1, v - 1)
        work2 = m * (v * v - u * u)
        loss2 = m * (v * v - w * w)
        if work2 % (big_x * big_x) or work2 % 2 or loss2 % (2 * big_x):
            continue
        k = work2 // (big_x * big_x)
        resist = loss2 // (2 * big_x)
        work = work2 // 2
        if 4 <= k <= 160 and 2 <= resist <= 150 and resist * big_x < work:
            break
    check(Fraction(k * big_x * big_x, 2) == work, "work integral")
    check(Fraction(u * u) + Fraction(2 * work, m) == v * v, "work-energy relation")
    check(Fraction(v * v) - Fraction(2 * resist * big_x, m) == w * w, "speed under resistance")

    stem = (
        f"{rng.choice(_CONTEXTS).format(m=m)}. The force is F(x) = {k}x newtons at x metres from the start, in the "
        f"direction of increasing x. The body starts at x = 0 with a speed of {u} m/s in that direction."
    )
    wa = Fraction(work)
    vf, wf = Fraction(v), Fraction(w)
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    f"Find the work done by the force in moving from x = 0 to x = {big_x}, in joules.",
                    wa,
                    0,
                    2,
                    [f"W = integral of {k}x from 0 to {big_x} = {k}({big_x})^2 / 2 = {work} J."],
                    [
                        (k * big_x * big_x, "dropped-factor-of-two"),
                        (k * big_x, "force-at-end-point-reported-for-work"),
                        (wa + Fraction(m * u * u, 2), "gave-final-kinetic-energy-for-work"),
                        (Fraction(k * big_x * big_x, 4), "integrated-to-half-the-distance"),
                    ],
                    sensible_range=(0, None),
                )
            ),
            dict(
                **number_part(
                    f"Hence find the speed of the body at x = {big_x}, in m/s, correct to 1 decimal place.",
                    vf,
                    1,
                    2,
                    [
                        f"Work-energy: W = (1/2)m v^2 - (1/2)m u^2, so v^2 = {u}^2 + 2({work})/{m} = {v * v}.",
                        f"v = {v}.",
                    ],
                    [
                        (v * v, "forgot-square-root"),
                        (_root(Fraction(2 * work, m)), "forgot-initial-kinetic-energy"),
                        (_root(Fraction(u * u) + Fraction(work, m)), "dropped-factor-of-two"),
                        (_root(Fraction(u * u) + Fraction(k * big_x * big_x, m)), "dropped-factor-of-two"),
                    ],
                    follow_through=f"sqrt({u}**2 + 2*part_a/{m})",
                    sensible_range=(0, None),
                )
            ),
            dict(
                **number_part(
                    f"Suppose a constant resistive force of {resist} N also acts against the motion over the same "
                    f"{big_x} m. Find the speed of the body at x = {big_x} in this case, in m/s, correct to 1 decimal place.",
                    wf,
                    1,
                    3,
                    [
                        f"The resistance does {resist} x {big_x} = {resist * big_x} J of negative work.",
                        f"v^2 = {v}^2 - 2({resist * big_x})/{m} = {w * w}, so v = {w}.",
                    ],
                    [
                        (w * w, "forgot-square-root"),
                        (_root(vf * vf - Fraction(resist * big_x, m)), "dropped-factor-of-two"),
                        (_root(vf * vf + Fraction(2 * resist * big_x, m)), "added-instead-of-subtracted"),
                        (vf, "resistance-ignored-in-energy-balance"),
                    ],
                    follow_through=f"sqrt(part_b**2 - 2*{resist}*{big_x}/{m})",
                    sensible_range=(0, None),
                )
            ),
        ],
    )
