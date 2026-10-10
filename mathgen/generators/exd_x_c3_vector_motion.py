"""Exam tier for vector calculus: from an acceleration vector to a position vector.

The hard tier integrates a velocity vector once. Here the learner is given a linear acceleration vector plus an
initial velocity and an initial position, and must integrate twice, placing each constant of integration from the
right initial condition. Components are drawn so that every position component is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import GRADER_TOLERANCE, check, generator, join_terms, pick_distractors
from .exd_x_c3_support import Redraw, draw


def _linear(coef_t: int, const: int) -> str:
    return join_terms([(coef_t, "t"), (const, "")])


def _build(rng: random.Random) -> dict:
    T = rng.choice([2, 3, 4])
    a = [rng.randint(-6, 6) for _ in range(2)]  # acceleration = a*t + b
    b = [rng.randint(-5, 5) for _ in range(2)]
    u = [rng.randint(-6, 6) for _ in range(2)]  # initial velocity
    p = [rng.randint(-8, 8) for _ in range(2)]  # initial position
    if 0 in a or 0 in b or u == [0, 0] or p == [0, 0]:
        raise Redraw
    pos, vel, no_p, no_u, undivided = [], [], [], [], []
    for i in range(2):
        # r(t) = p + u t + b t^2 / 2 + a t^3 / 6, v(t) = u + b t + a t^2 / 2
        r = Fraction(p[i]) + u[i] * T + Fraction(b[i] * T**2, 2) + Fraction(a[i] * T**3, 6)
        if r.denominator != 1:
            raise Redraw
        pos.append(int(r))
        vel.append(Fraction(u[i]) + b[i] * T + Fraction(a[i] * T**2, 2))
        no_p.append(int(r) - p[i])
        no_u.append(int(r) - u[i] * T)
        undivided.append(p[i] + u[i] * T + b[i] * T**2 + a[i] * T**3)
    if any(v.denominator != 1 for v in vel):
        raise Redraw
    answer_text = f"x = {pos[0]}, y = {pos[1]}"
    cands = [
        f"x = {int(vel[0])}, y = {int(vel[1])}",  # integrates once: gives the velocity, not the position
        f"x = {no_p[0]}, y = {no_p[1]}",  # leaves out the initial position
        f"x = {no_u[0]}, y = {no_u[1]}",  # leaves out the initial velocity
        f"x = {undivided[0]}, y = {undivided[1]}",  # integrates t^n to t^(n+1) without dividing
    ]
    distractors = pick_distractors(answer_text, cands, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 3, "fewer than three distinct distractors")
    text = (
        f"A particle moves in the plane with acceleration a(t) = ({_linear(a[0], b[0])}, {_linear(a[1], b[1])}) "
        f"in metres per second squared, t in seconds. At t = 0 its velocity is ({u[0]}, {u[1]}) metres per second "
        f"and its position is ({p[0]}, {p[1]}) metres. Find the particle's position vector at t = {T} seconds."
    )
    return dict(
        text=text,
        latex="",
        answer={"x": float(pos[0]), "y": float(pos[1])},
        answer_text=answer_text,
        distractors=distractors,
    )


@generator("exd_x_c3_vector_motion", level=15, presets={"exam": dict(tier="exam")})
def exd_x_c3_vector_motion(rng: random.Random, tier: str = "exam") -> dict:
    """Integrate an acceleration vector twice, using the initial velocity and position."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
