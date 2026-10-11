"""Exam tier for the sine rule: the height of a tower from two angles of elevation.

The practice tiers give a triangle and ask for a side. Here the triangle has to be found: two observers stand in
line with the foot of a vertical object, a known distance apart, and each measures an angle of elevation. The
learner must spot the non-right triangle (angle at the top is the difference of the elevations, by the exterior
angle theorem), apply the sine rule for a slant distance, and then finish in the right-angled triangle.

    h = d sin(alpha) sin(beta) / sin(alpha - beta)
"""

from __future__ import annotations

import math
import random

from ..core import GRADER_TOLERANCE, check, generator, phrase, pick_distractors

_PATTERNS = (
    "From a point A on level ground the angle of elevation of the top of a vertical tower is {near} degrees. "
    "From a point B, {d} m further from the tower and in line with A and the foot of the tower, the angle of "
    "elevation is {far} degrees. Find the height of the tower (1 d.p.).",
    "A vertical flagpole stands on level ground. The angle of elevation of its top is {near} degrees from point A. "
    "Walking {d} m directly away from the flagpole to point B, the angle of elevation of the top is {far} degrees. "
    "Find the height of the flagpole (1 d.p.).",
    "Two boats, A and B, are in line with the foot of a vertical cliff, and B is {d} m further out than A. "
    "From A the angle of elevation of the top of the cliff is {near} degrees and from B it is {far} degrees. "
    "Find the height of the cliff above sea level (1 d.p.).",
)


def _clear(value: float, eps: float = 0.03) -> bool:
    return abs((value * 10) % 1 - 0.5) > eps


@generator("exd_x_f5_sinetower", level=12, presets={"exam": dict(tier="exam")})
def exd_x_f5_sinetower(rng: random.Random, tier: str = "exam") -> dict:
    """Height of a vertical object from two angles of elevation and the distance between the observers."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    while True:
        near = rng.randint(35, 70)
        far = rng.randint(20, near - 8)
        d = rng.randint(15, 90)
        s = lambda deg: math.sin(math.radians(deg))  # noqa: E731
        t = lambda deg: math.tan(math.radians(deg))  # noqa: E731
        h_exact = d * s(near) * s(far) / s(near - far)
        if not _clear(h_exact):
            continue
        h = round(h_exact, 1)
        slant_a = d * s(far) / s(near - far)  # distance from A to the top
        cands_exact = [
            slant_a,  # stops after the sine rule
            d * s(near) * s(far) / s(near + far),  # angle at the top taken as the sum of the elevations
            slant_a * s(far),  # right-angled step uses the far angle on the near slant
            d * t(far),  # ignores the near observation entirely
        ]
        distractors = pick_distractors(
            f"h = {h:.1f}", [f"h = {c:.1f}" for c in cands_exact], n=3, tolerance=GRADER_TOLERANCE
        )
        if len(distractors) == 3:
            break
    # Independent route: the foot is at the origin, so h = x_A tan(near) = x_B tan(far) and x_B - x_A = d.
    x_a = h_exact / t(near)
    x_b = h_exact / t(far)
    check(abs((x_b - x_a) - d) < 1e-9, "two-observer geometry does not reproduce the distance d")
    text = phrase(rng, _PATTERNS, near=near, far=far, d=d)
    answer_text = f"h = {h:.1f}"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=h,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The angle at the top is {near} - {far} = {near - far} degrees, so AT = {d} sin({far}) / sin({near - far}).",
            f"h = AT sin({near}) = {h:.1f}.",
        ],
        distractors=distractors,
    )
