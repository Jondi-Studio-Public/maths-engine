"""Generates `exd_x_r3f_complex_parallelogram` questions: an exam chain on a parallelogram in the complex plane.

Three vertices A, B, C of a parallelogram ABCD (taken in that order) are given as complex numbers.

(a) the x-coordinate of the fourth vertex d = a - b + c           number, 1 mark
(b) the y-coordinate of D                                         number, 1 mark
(c) the area of the parallelogram                                 number, 2 marks

Part (c) follows through from (a) and (b): the area is |Re(AB) Im(AD) - Im(AB) Re(AD)| with AD = d - a, so a
learner whose D is wrong still earns (c) for using their own D correctly. The coordinates of d are kept
non-zero so neither of the first two answers is the easily guessed 0.
"""

from __future__ import annotations

import random

from ..core import attempts, check, generator, join_terms, phrase
from ._exd_r3f_parts import number_part
from ._exd_distinct_parts import distinct_part_answers


def _ctext(re_part: int, im_part: int) -> str:
    return join_terms([(re_part, ""), (im_part, "i")])


_STEMS = (
    "In the complex plane, the points A, B and C represent the complex numbers {a}, {b} and {c}. "
    "ABCD is a parallelogram, with its vertices in that order, and D represents the complex number d.",
    "On an Argand diagram, the points A, B and C represent {a}, {b} and {c}. "
    "The point D completes the parallelogram ABCD (vertices in that order) and represents the complex number d.",
    "The complex numbers {a}, {b} and {c} are represented by the points A, B and C in the complex plane. "
    "The fourth vertex D of the parallelogram ABCD represents the complex number d.",
)


@generator("exd_x_r3f_complex_parallelogram", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_complex_parallelogram(rng: random.Random, tier: str = "exam") -> dict:
    """The fourth vertex of a parallelogram given as complex numbers, then its area."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a non-degenerate parallelogram with non-zero fourth vertex"):
        a = (rng.randint(-8, 8), rng.randint(-8, 8))
        b = (rng.randint(-8, 8), rng.randint(-8, 8))
        c = (rng.randint(-8, 8), rng.randint(-8, 8))
        if len({a, b, c}) < 3:
            continue
        d = (a[0] - b[0] + c[0], a[1] - b[1] + c[1])
        ab = (b[0] - a[0], b[1] - a[1])
        ad = (d[0] - a[0], d[1] - a[1])
        cross = ab[0] * ad[1] - ab[1] * ad[0]
        if d[0] != 0 and d[1] != 0 and abs(cross) >= 6 and abs(d[0]) <= 12 and abs(d[1]) <= 12:
            break
    area = abs(cross)

    # Re-derive the area by the shoelace formula over A, B, C, D.
    pts = [a, b, c, d]
    shoelace = abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))) // 2
    check(shoelace == area, "the area disagrees with the shoelace formula")

    def other_vertex_candidates(index: int):
        return [
            (a[index] + b[index] - c[index], "parallelogram-wrong-opposite-vertex"),
            (b[index] + c[index] - a[index], "parallelogram-wrong-opposite-vertex"),
            (a[index] + b[index] + c[index], "parallelogram-added-all-vertices"),
            (
                (a[index] + c[index]) // 2 if (a[index] + c[index]) % 2 == 0 else None,
                "parallelogram-midpoint-for-fourth-vertex",
            ),
        ]

    dot = abs(ab[0] * ad[0] + ab[1] * ad[1])
    area_candidates = [
        (dot or None, "parallelogram-area-by-dot-product"),
        (abs(ab[0] * ad[1]), "parallelogram-area-single-product"),
        (area // 2 if area % 2 == 0 else None, "parallelogram-area-as-triangle"),
        (abs(a[0] * c[1] - a[1] * c[0]), "parallelogram-area-from-position-vectors"),
    ]

    stem = phrase(rng, _STEMS, a=_ctext(*a), b=_ctext(*b), c=_ctext(*c))
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    follow = f"Abs(({bx - ax})*(part_b - ({ay})) - ({by - ay})*(part_a - ({ax})))"
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the x-coordinate of D.",
                d[0],
                other_vertex_candidates(0),
                marks=1,
                working=[f"d = a - b + c, so Re(d) = {a[0]} - ({b[0]}) + ({c[0]}) = {d[0]}."],
            ),
            number_part(
                "Find the y-coordinate of D.",
                d[1],
                other_vertex_candidates(1),
                marks=1,
                working=[f"Im(d) = {a[1]} - ({b[1]}) + ({c[1]}) = {d[1]}."],
            ),
            number_part(
                "Hence find the area of the parallelogram ABCD, in square units.",
                area,
                area_candidates,
                marks=2,
                sensible_range=(0, None),
                follow_through=follow,
                working=[
                    f"AB = ({ab[0]}, {ab[1]}) and AD = ({ad[0]}, {ad[1]}).",
                    f"Area = |{ab[0]} × {ad[1]} - ({ab[1]}) × ({ad[0]})| = {area}.",
                ],
            ),
        ],
    )
