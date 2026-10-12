"""Exam source for `ucv1-linear-system-determinant`: Cramer's rule used to place a third line.

The stem gives two lines a x + b y = e and c x + d y = f with a non-zero determinant, and a third line g x + h y = k
whose constant k is unknown, which passes through the point where the first two meet.

(a) D, the determinant of the coefficients;
(b) D_x (the x-column replaced by the constants);
(c) D_y (the y-column replaced by the constants);
(d) k = g x + h y with x = D_x / D and y = D_y / D, from (a), (b) and (c): an exact rational.

Part (d) carries `follow_through` on all three earlier parts. The key is checked by solving the system directly.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms, paren_negative

from .exd_x_r3c_parts import fraction_part, number_part
from ._exd_r3_distinct import distinct_part_answers


def _line(p, q, rhs):
    return f"{join_terms([(p, 'x'), (q, 'y')])} = {rhs}"


@generator("exd_x_r3c_cramer_line", level=16, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_cramer_line(rng, tier="exam"):
    """D, D_x, D_y of a 2 by 2 system and the constant that puts a third line through its solution."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    nz = [v for v in range(-6, 7) if v]
    for _ in attempts(500, "a system with a determinant of at least 2 and distinct options"):
        a, b, c, d = (rng.choice(nz) for _ in range(4))
        e, f = rng.randint(-12, 12), rng.randint(-12, 12)
        g, h = rng.choice([v for v in range(-5, 6) if v]), rng.choice([v for v in range(-5, 6) if v])
        det = a * d - b * c
        dx, dy = e * d - b * f, a * f - e * c
        if abs(det) >= 2 and dx and dy and dx != dy and abs(dx) != abs(dy):
            break
    x, y = Fraction(dx, det), Fraction(dy, det)
    k = g * x + h * y
    check(a * x + b * y == e and c * x + d * y == f, "(x, y) does not solve the system")

    stem = (
        f"Two lines have equations {_line(a, b, e)} and {_line(c, d, f)}. A third line {_line(g, h, 'k')} "
        f"passes through the point where the first two lines meet."
    )
    part_a = number_part(
        "Find D, the determinant of the matrix of coefficients of x and y in the first two equations.",
        det,
        working=[f"D = ({a})({d}) - ({b})({c}) = {det}."],
        wrongs=[
            (a * d + b * c, "determinant-products-added"),
            (b * c - a * d, "determinant-order-reversed"),
            (a * c - b * d, "determinant-order-reversed"),
        ],
    )
    part_b = number_part(
        "Find D_x, the determinant of the matrix formed by replacing the column of coefficients of x with the constants on the right-hand sides.",
        dx,
        working=[f"D_x = ({e})({d}) - ({b})({f}) = {dx}."],
        wrongs=[
            (dy, "wrong-column-replaced"),
            (e * d + b * f, "determinant-products-added"),
            (-dx, "determinant-order-reversed"),
        ],
    )
    part_c = number_part(
        "Find D_y, the determinant of the matrix formed by replacing the column of coefficients of y with the constants on the right-hand sides.",
        dy,
        working=[f"D_y = ({a})({f}) - ({e})({c}) = {dy}."],
        wrongs=[
            (dx, "wrong-column-replaced"),
            (a * f + e * c, "determinant-products-added"),
            (-dy, "determinant-order-reversed"),
        ],
    )
    part_d = fraction_part(
        "Using your answers to (a), (b) and (c), find the value of k.",
        k,
        marks=2,
        working=[
            f"x = D_x/D = {dx}/{paren_negative(det)} = {x}, y = D_y/D = {dy}/{paren_negative(det)} = {y}.",
            f"k = {join_terms([(g, 'x'), (h, 'y')])}, so with x = {x} and y = {y}, k = {k}.",
        ],
        wrongs=[
            (g * dx + h * dy, "forgot-to-divide-by-determinant"),
            (Fraction(g * dy + h * dx, det), "swapped-x-and-y-values"),
            (g * Fraction(det, dx) + h * Fraction(det, dy), "inverted-ratio"),
        ],
        follow_through=f"({g}*part_b + {h}*part_c)/part_a",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
