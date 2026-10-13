"""Exam question: coordinates in a basis of R^2, then a map that is diagonal in that basis (`ucv1-coordinate-basis`).

Stem: a basis b1, b2 of R^2, a vector v, and a linear map T with T(b1) = l1 b1 and T(b2) = l2 b2. The parts chain:

* (a) the determinant of the matrix with columns b1 and b2, which is non-zero exactly when they form a basis;
* (b) the coordinate c1 in v = c1 b1 + c2 b2, by Cramer's rule (follow-through from (a));
* (c) the coordinate c2 (follow-through from (a));
* (d) the sum of the two entries of T(v) = l1 c1 b1 + l2 c2 b2 (follow-through from (b) and (c)).

Everything is an exact rational, so the keys are exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, join_terms
from .exd_x_r2g_support import build_part, redraws

_STEMS = [
    "The vectors b1 = ({x1}, {y1}) and b2 = ({x2}, {y2}) are proposed as a basis of R^2. A linear map T of R^2 "
    "satisfies T(b1) = {l1}b1 and T(b2) = {l2}b2. Let v = ({v1}, {v2}).",
    "Let b1 = ({x1}, {y1}) and b2 = ({x2}, {y2}) be vectors in R^2, and let T be the linear map with T(b1) = "
    "{l1}b1 and T(b2) = {l2}b2. Take v = ({v1}, {v2}).",
    "A linear map T on R^2 stretches along two directions: T(b1) = {l1}b1 and T(b2) = {l2}b2, where b1 = "
    "({x1}, {y1}) and b2 = ({x2}, {y2}). A vector v = ({v1}, {v2}) is given.",
]


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        x1, y1, x2, y2 = (rng.choice([-1, 1]) * rng.randint(1, 5) for _ in range(4))
        det = x1 * y2 - x2 * y1
        if not 2 <= abs(det) <= 14:
            continue
        v1, v2 = (rng.choice([-1, 1]) * rng.randint(1, 9) for _ in range(2))
        l1, l2 = rng.sample([-4, -3, -2, 2, 3, 4], 2)
        c1 = Fraction(v1 * y2 - v2 * x2, det)
        c2 = Fraction(x1 * v2 - y1 * v1, det)
        if c1 == 0 or c2 == 0 or c1 == c2 or c1 == -c2:
            continue
        if c1.denominator == 1 and c2.denominator == 1 and rng.random() < 0.7:
            continue
        return dict(x1=x1, y1=y1, x2=x2, y2=y2, v1=v1, v2=v2, l1=l1, l2=l2, det=det, c1=c1, c2=c2)
    check(False, "no acceptable basis draw")


@generator("exd_x_r2g_basis_map", level=13, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_basis_map(rng: random.Random, tier: str = "exam") -> dict:
    """Determinant, two coordinates and the image of a vector under a map that is diagonal in a basis."""
    check(tier == "exam", "exd_x_r2g_basis_map: only the exam tier exists")
    d = _draw(rng)
    x1, y1, x2, y2, v1, v2, l1, l2 = (d[k] for k in ("x1", "y1", "x2", "y2", "v1", "v2", "l1", "l2"))
    det, c1, c2 = d["det"], d["c1"], d["c2"]
    stem = rng.choice(_STEMS).format(**{k: d[k] for k in ("x1", "y1", "x2", "y2", "l1", "l2", "v1", "v2")})
    tv = (l1 * c1 * x1 + l2 * c2 * x2, l1 * c1 * y1 + l2 * c2 * y2)
    total = tv[0] + tv[1]
    num1, num2 = v1 * y2 - v2 * x2, x1 * v2 - y1 * v1
    parts = [
        build_part(
            "Find the determinant of the 2 x 2 matrix whose columns are b1 and b2.",
            det,
            kind="number",
            marks=1,
            working=[f"det = ({x1})({y2}) - ({x2})({y1}) = {det}, which is not 0."],
            wrong=[
                (x1 * y2 + x2 * y1, "determinant-formula-slip"),
                (-det, "answer-sign-reversed"),
                (x1 * x2 - y1 * y2, "determinant-formula-slip"),
                (x1 * y2, "determinant-formula-slip"),
                (x1 * y1 - x2 * y2, "determinant-formula-slip"),
                (x1 * x2 - y1 * y2, "determinant-formula-slip"),
                (x2 * y1, "determinant-formula-slip"),
            ],
        ),
        build_part(
            "Hence find c1 in v = c1 b1 + c2 b2, as a fraction in lowest terms if it is not a whole number.",
            c1,
            kind="fraction",
            marks=2,
            follow_through=f"({num1})/part_a",
            working=[f"By Cramer's rule c1 = det[v b2] / det[b1 b2] = ({num1}) / {det} = {c1}."],
            wrong=[
                (c2, "swaps-coefficients"),
                (Fraction(num1), "forgot-to-divide-by-determinant"),
                (Fraction(v1 * y2 + v2 * x2, det), "sign-error-in-cramers-numerator"),
                (Fraction(v1, x1), "ignores-second-basis-vector"),
                (-c1, "answer-sign-reversed"),
            ],
        ),
        build_part(
            "Hence find c2, as a fraction in lowest terms if it is not a whole number.",
            c2,
            kind="fraction",
            marks=2,
            follow_through=f"({num2})/part_a",
            working=[f"c2 = det[b1 v] / det[b1 b2] = ({num2}) / {det} = {c2}."],
            wrong=[
                (c1, "swaps-coefficients"),
                (Fraction(num2), "forgot-to-divide-by-determinant"),
                (Fraction(x1 * v2 + y1 * v1, det), "sign-error-in-cramers-numerator"),
                (Fraction(v2, y2), "ignores-second-basis-vector"),
                (-c2, "answer-sign-reversed"),
            ],
        ),
        build_part(
            "Hence find the sum of the two entries of the vector T(v), as a fraction in lowest terms if it is not "
            "a whole number.",
            total,
            kind="fraction",
            marks=1,
            follow_through=f"{l1}*part_b*({x1 + y1}) + {l2}*part_c*({x2 + y2})",
            working=[
                f"T(v) = {join_terms([(l1, 'c1 b1'), (l2, 'c2 b2')])} = ({tv[0]}, {tv[1]}).",
                f"The entries sum to {total}.",
            ],
            wrong=[
                (l2 * c1 * (x1 + y1) + l1 * c2 * (x2 + y2), "map-factors-on-wrong-basis-vectors"),
                (l1 * c1 + l2 * c2, "coordinates-reported-as-standard-vector"),
                (Fraction(v1 + v2), "applies-the-map-incompletely"),
                (l1 * c1 * (x1 + y1) + c2 * (x2 + y2), "applies-the-map-incompletely"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
