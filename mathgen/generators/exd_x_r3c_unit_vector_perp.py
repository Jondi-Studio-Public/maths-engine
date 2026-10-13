"""Exam source for the unit-vector and perpendicular skills: a point reached by a scaled perpendicular.

The stem gives points A and B whose separation is a Pythagorean triple (scaled), a turn of AB through 90 degrees
(clockwise or anticlockwise), a length L and the point C reached by that turned vector from B.

(a) the magnitude of AB;
(b) the j-coefficient of the unit vector along AB (its own sign), using (a);
(c) the i-coefficient of BC, which is the turned unit vector scaled to length L, using (b);
(d) the x-coordinate of C, which is the x-coordinate of B plus (c).

Parts (b), (c) and (d) carry `follow_through`. Every answer is an exact rational. The key is checked against the turn
done directly on the components (BC is perpendicular to AB and has length L).
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import check, generator, join_terms, paren_negative

from .exd_x_r3c_parts import fraction_part, number_part
from ._exd_distinct_parts import distinct_part_answers

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29))
_MAX_SCALE = {5: 4, 13: 2, 17: 1, 25: 1, 29: 1}
_STEMS = (
    "The points A({xa}, {ya}) and B({xb}, {yb}) are given. The point C is such that the vector BC has magnitude "
    "{length} and is obtained by turning the vector AB through 90° {turn}.",
    "A drone flies in a straight line from A({xa}, {ya}) to B({xb}, {yb}). It then turns through 90° {turn} and "
    "flies a further {length} units in a straight line, reaching C.",
    "Let A be the point ({xa}, {ya}) and B the point ({xb}, {yb}). The vector BC is perpendicular to AB, with "
    "length {length}, and points in the direction of AB turned through 90° {turn}.",
)


@generator("exd_x_r3c_unit_vector_perp", level=15, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_unit_vector_perp(rng, tier="exam"):
    """AB, its unit vector, the turned vector of length L, and the point C."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    p, q, h = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        p, q = q, p
    scale = rng.randint(1, _MAX_SCALE[h])
    dx, dy = rng.choice((-1, 1)) * p * scale, rng.choice((-1, 1)) * q * scale
    magnitude = h * scale
    xa, ya = rng.randint(-9, 9), rng.randint(-9, 9)
    xb, yb = xa + dx, ya + dy
    anticlockwise = rng.random() < 0.5
    length = h * rng.randint(2, 4) if rng.random() < 0.5 else rng.randint(5, 40)

    ux, uy = Fraction(dx, magnitude), Fraction(dy, magnitude)
    # Turning (ux, uy) anticlockwise gives (-uy, ux); clockwise gives (uy, -ux).
    wi = (-uy if anticlockwise else uy) * length
    wj = (ux if anticlockwise else -ux) * length
    xc = xb + wi
    # Direct check on the components: BC is perpendicular to AB and has length L.
    check(wi * dx + wj * dy == 0 and wi * wi + wj * wj == length * length, "BC is not a perpendicular of length L")
    check(magnitude * magnitude == dx * dx + dy * dy, "the magnitude disagrees with the components")
    check(xc - xb == wi and uy * magnitude == dy, "coordinates disagree")

    turn = "anticlockwise" if anticlockwise else "clockwise"
    stem = rng.choice(_STEMS).format(xa=xa, ya=ya, xb=xb, yb=yb, length=length, turn=turn)
    sign = "-" if anticlockwise else ""
    part_a = number_part(
        "Find the magnitude of the vector AB.",
        magnitude,
        marks=2,
        sensible_range=(0, None),
        working=[
            f"|AB| = √(({xb} - {paren_negative(xa)})² + ({yb} - {paren_negative(ya)})²) = √({dx**2} + {dy**2}) = {magnitude}."
        ],
        wrongs=[
            (abs(dx) + abs(dy), "magnitude-as-sum-of-components"),
            (magnitude * magnitude, "magnitude-without-square-root"),
            (abs(dx), "magnitude-as-one-leg"),
            (abs(dy), "magnitude-as-one-leg"),
        ],
    )
    part_b = fraction_part(
        "Using your answer to (a), find the coefficient of j in the unit vector in the direction of AB.",
        uy,
        working=[
            f"AB = {join_terms([(dx, 'i'), (dy, 'j')])}, so the j-coefficient of the unit vector is {dy}/{magnitude}"
            + (f" = {uy}." if uy.denominator != magnitude else ".")
        ],
        wrongs=[
            (ux, "used-other-component-of-vector"),
            (Fraction(dy), "forgot-to-normalise"),
            (-uy, "unit-component-sign-slip"),
        ],
        follow_through=f"({dy})/part_a",
    )
    part_c = fraction_part(
        "Using your answer to (b), find the coefficient of i in the vector BC.",
        wi,
        marks=2,
        working=[
            f"Turning a vector {turn} through 90° takes (i, j) components (a, b) to "
            f"{'(-b, a)' if anticlockwise else '(b, -a)'}.",
            f"So the i-coefficient of BC is {length} × {'-' if anticlockwise else ''}({uy}) = {wi}.",
        ],
        wrongs=[
            (ux * length, "unit-vector-not-rotated"),
            (-wi, "wrong-rotation-direction"),
            ((-uy if anticlockwise else uy), "forgot-to-scale-unit-vector"),
        ],
        follow_through=f"{sign}{length}*part_b",
    )
    part_d = fraction_part(
        "Using your answer to (c), find the x-coordinate of C.",
        xc,
        working=[f"x(C) = x(B) + (i-coefficient of BC) = {xb} {'+' if wi >= 0 else '-'} {abs(wi)} = {xc}."],
        wrongs=[
            (xa + wi, "started-from-wrong-point"),
            (wi, "vector-component-as-coordinate"),
            (xb - wi, "wrong-rotation-direction"),
        ],
        follow_through=f"{xb} + part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
