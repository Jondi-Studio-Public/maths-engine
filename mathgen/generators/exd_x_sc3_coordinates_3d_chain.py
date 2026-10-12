"""Exam tier for `f10-three-dimensional-coordinates`: displacement over two legs, then the straight-line distance.

The hard tier combines two moves into one displacement. Here the first move is given as two points and the second
in words (metres east or west, north or south, up or down), so the learner has to translate directions into signed
components before combining. Parts (a) to (c) give the components of the total displacement from the start to the
finish; part (d) uses all three, with Pythagoras in three dimensions, for the straight-line distance. The
components are drawn from whole-number Pythagorean quadruples, so the distance is a whole number of metres.

Built backwards from the total displacement (a permuted, signed quadruple) and the second move. Part (d) carries
`follow_through` from the three components.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_sc3_support import number_part, tagged


def _quadruples() -> list[tuple[int, int, int, int]]:
    """(a, b, c, d), 0 < a <= b <= c, a^2 + b^2 + c^2 = d^2, d up to 40."""
    out = []
    for d in range(3, 41):
        for a in range(1, d):
            for b in range(a, d):
                c2 = d * d - a * a - b * b
                if c2 < b * b:
                    break
                c = math.isqrt(c2)
                if c * c == c2:
                    out.append((a, b, c, d))
    return out


_QUADRUPLES = _quadruples()
_AXES = [("east", "west"), ("north", "south"), ("up", "down")]
_SETTINGS = [
    "A survey drone is flying over a vineyard.",
    "A drone is photographing a wheat paddock.",
    "A drone is inspecting a mine site.",
    "A drone is mapping a bushfire recovery area.",
]


def _move_words(vector: tuple[int, int, int]) -> str:
    bits = []
    for value, (pos, neg) in zip(vector, _AXES):
        bits.append(f"{abs(value)} m {pos if value > 0 else neg}")
    return f"{bits[0]}, {bits[1]} and {bits[2]}"


@generator("exd_x_sc3_coordinates_3d_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc3_coordinates_3d_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Three components of a two-leg displacement, then the straight-line distance."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a drone flight inside the allowed airspace"):
        quad = rng.choice(_QUADRUPLES)
        parts_abs = list(quad[:3])
        rng.shuffle(parts_abs)
        total = tuple(v * rng.choice([-1, 1]) for v in parts_abs)  # displacement from A to C
        dist = quad[3]
        start = (rng.randint(-120, 120), rng.randint(-120, 120), rng.randint(10, 100))
        second = tuple(rng.randint(5, 45) * rng.choice([-1, 1]) for _ in range(3))
        first = tuple(t - s for t, s in zip(total, second))  # displacement from A to B
        if any(f == 0 for f in first):
            continue
        mid = tuple(a + f for a, f in zip(start, first))
        end = tuple(a + t for a, t in zip(start, total))
        if not (5 <= mid[2] <= 120 and 5 <= end[2] <= 120):
            continue
        break
    check(math.isqrt(sum(t * t for t in total)) == dist and dist * dist == sum(t * t for t in total), "distance")
    check(all(m - s == f for m, s, f in zip(mid, start, first)), "first leg disagrees with the points")
    check(tuple(f + g for f, g in zip(first, second)) == total, "legs do not add to the total")

    stem = (
        f"{rng.choice(_SETTINGS)} Positions are given as (east, north, height) in metres from a survey marker. "
        f"The drone starts at A({start[0]}, {start[1]}, {start[2]}), flies to B({mid[0]}, {mid[1]}, {mid[2]}) "
        f"and then flies a further {_move_words(second)} to reach C."
    )
    asks = [
        "Find the east component of the displacement from A to C, in metres (a negative answer means west).",
        "Find the north component of the displacement from A to C, in metres (a negative answer means south).",
        "Find the vertical component of the displacement from A to C, in metres (a negative answer means down).",
    ]
    parts = []
    for i in range(3):
        cands = [
            (Fraction(-total[i]), "wrong-direction"),
            (Fraction(second[i]), "ignores-first-leg"),
            (Fraction(first[i]), "ignores-second-leg"),
            (Fraction(end[i]), "reports-position-not-displacement"),
            (Fraction(first[i] - second[i]), "misreads-direction-word"),
        ]
        parts.append(
            number_part(
                asks[i],
                Fraction(total[i]),
                0,
                1,
                [
                    f"A to B: {mid[i]} - {start[i]} = {first[i]}. B to C: {second[i]}.",
                    f"Total = {first[i]} + {second[i]} = {total[i]}.",
                ],
                tagged(Fraction(total[i]), cands, 0),
            )
        )
    horizontal = math.sqrt(total[0] ** 2 + total[1] ** 2)
    d_cands = [
        (Fraction(sum(abs(t) for t in total)), "adds-instead-of-using-pythagoras"),
        (Fraction(round(horizontal * 10), 10), "ignores-the-vertical-component"),
        (Fraction(sum(t * t for t in total)), "forgets-the-square-root"),
        (Fraction(round(math.sqrt(sum(s * s for s in second)) * 10), 10), "ignores-first-leg"),
    ]
    parts.append(
        number_part(
            "Hence find the straight-line distance from A to C, in metres.",
            Fraction(dist),
            0,
            2,
            [f"Distance = √({total[0]}² + {total[1]}² + {total[2]}²) = √{sum(t * t for t in total)} = {dist} m."],
            tagged(Fraction(dist), d_cands, 0, lo=0),
            follow_through="sqrt(part_a**2+part_b**2+part_c**2)",
            sensible_range=(0, None),
        )
    )
    return dict(text=stem, parts=parts)
