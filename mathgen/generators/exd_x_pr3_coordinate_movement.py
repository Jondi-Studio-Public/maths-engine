"""Exam source for coordinate movement: a route described in words, with the middle point to be found.

A walker starts at a known point and makes a first move described by compass directions (so the signs have to be
read from the words), reaching a point B that is never named. They then walk to a known end point. Parts (a) and
(b) find B's coordinates, and parts (c) and (d) use B to find the change in each coordinate over the second leg.
Parts (c) and (d) carry `follow_through`.
"""

from __future__ import annotations

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import leaks, part

_OPENINGS = [
    "A bushwalker in the Blue Mountains marks a map grid where each unit is 1 km and the car park is at (0, 0). "
    "She starts at ({x1}, {y1}) and walks {mx} units {dx_word} and {my} units {dy_word} to a waterhole, B. "
    "Then she walks to a lookout at ({x3}, {y3}).",
    "A ranger in Kakadu uses a grid map where each unit is 1 km and the ranger station is at (0, 0). He leaves "
    "a camp at ({x1}, {y1}), drives {mx} units {dx_word} and {my} units {dy_word} to a billabong, B, and then "
    "continues to a lookout at ({x3}, {y3}).",
    "On a farm in the Riverina a drone flies over a grid map where each unit is 100 m and the shed is at (0, 0). "
    "It takes off at ({x1}, {y1}), flies {mx} units {dx_word} and {my} units {dy_word} to a dam, B, and then "
    "flies on to a bore at ({x3}, {y3}).",
]


@generator("exd_x_pr3_coordinate_movement", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_coordinate_movement(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(1000, "a route whose answers are not in the prompt"):
        x1, y1, x3, y3 = (rng.randint(-12, 12) for _ in range(4))
        mx, my = rng.randint(2, 12), rng.randint(2, 12)
        sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
        xb, yb = x1 + sx * mx, y1 + sy * my
        if not all((x1, y1, x3, y3, xb, yb)) or x3 == xb or y3 == yb or x1 == x3 or y1 == y3 or xb == yb:
            continue
        dx_word = "east" if sx > 0 else "west"
        dy_word = "north" if sy > 0 else "south"
        stem = rng.choice(_OPENINGS).format(x1=x1, y1=y1, x3=x3, y3=y3, mx=mx, my=my, dx_word=dx_word, dy_word=dy_word)
        stem += " East and north are the positive directions."
        if not leaks(stem, [abs(k) for k in (xb, yb, x3 - xb, y3 - yb)]):
            break
    dx2, dy2 = x3 - xb, y3 - yb
    check(xb + dx2 == x3 and yb + dy2 == y3, "second leg does not reach the end point")
    parts = [
        part(
            "What is the x-coordinate of B?",
            xb,
            working=[
                f"{dx_word.capitalize()} is {'positive' if sx > 0 else 'negative'}: {x1} {'+' if sx > 0 else '−'} {mx} = {xb}."
            ],
            candidates=[
                (x1 - sx * mx, "direction-sign-reversed"),
                (sx * mx, "used-change-as-coordinate"),
                (yb, "swapped-coordinates"),
            ],
            sensible_range=None,
        ),
        part(
            "What is the y-coordinate of B?",
            yb,
            working=[
                f"{dy_word.capitalize()} is {'positive' if sy > 0 else 'negative'}: {y1} {'+' if sy > 0 else '−'} {my} = {yb}."
            ],
            candidates=[
                (y1 - sy * my, "direction-sign-reversed"),
                (sy * my, "used-change-as-coordinate"),
                (xb, "swapped-coordinates"),
            ],
            sensible_range=None,
        ),
        part(
            "Using your answer to (a), find the change in the x-coordinate for the journey from B to the end point.",
            dx2,
            follow_through=f"{x3}-part_a",
            working=[f"Δx = {x3} − ({xb}) = {dx2}."],
            candidates=[
                (x3 - x1, "total-change-for-leg"),
                (xb - x3, "reversed-subtraction"),
                (x3, "used-change-as-coordinate"),
            ],
            sensible_range=None,
        ),
        part(
            "Using your answer to (b), find the change in the y-coordinate for the journey from B to the end point.",
            dy2,
            follow_through=f"{y3}-part_b",
            working=[f"Δy = {y3} − ({yb}) = {dy2}."],
            candidates=[
                (y3 - y1, "total-change-for-leg"),
                (yb - y3, "reversed-subtraction"),
                (y3, "used-change-as-coordinate"),
            ],
            sensible_range=None,
        ),
    ]
    return dict(text=stem, parts=parts)
