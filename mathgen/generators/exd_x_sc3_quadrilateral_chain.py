"""Exam tier for `f10-quadrilateral-properties`: a connected question on one rhombus.

The hard tier asks for one length or angle from one stated property. Here several properties are chained on the
same figure, and a later part uses the earlier answer. There are two forms.

* **Angles.** Two consecutive angles of a rhombus are given as algebraic expressions. Part (a) solves for x (the two
  angles add to 180 degrees), part (b) substitutes back for an angle, and part (c) uses the fact that a diagonal
  bisects the angle at the vertices it joins to find the angle between the diagonal and a side.
* **Lengths.** The perimeter and one diagonal of a rhombus are given. Part (a) finds the side, part (b) uses
  Pythagoras in the right-angled triangle made by the diagonals to find half of the other diagonal, and part (c)
  doubles it.

Both are built backwards from whole-number answers, so every key is exact. Parts (b) and (c) carry
`follow_through`, so a learner whose first answer is wrong still earns the marks for using it correctly.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_sc3_support import number_part, tagged

_NAMES = ["ABCD", "PQRS", "WXYZ", "EFGH"]
_OBJECTS = ["tile", "stained-glass panel", "paving stone", "quilt patch", "metal plate"]


def _triples() -> list[tuple[int, int, int]]:
    """(p, q, z) with p, q, z whole, p^2 + q^2 = z^2, p != q and z up to 40: a half-diagonal pair and its side."""
    out = []
    for z in range(5, 41):
        for p in range(3, z):
            q2 = z * z - p * p
            q = math.isqrt(q2)
            if q * q == q2 and q >= 3 and q != p:
                out.append((p, q, z))
    return out


_TRIPLES = _triples()


def _angles(rng: random.Random) -> dict:
    nm = rng.choice(_NAMES)
    a, b, c, d = nm
    for _ in attempts(1000, "a rhombus angle pair with whole-number x"):
        big = rng.randint(40, 140)  # angle ABC
        if big == 90:
            continue
        x = rng.randint(8, 35)
        p = rng.randint(2, 5)
        q = big - p * x
        r = rng.randint(2, 5)
        s = r * x - (180 - big)  # angle BCD = rx - s = 180 - big
        if 3 <= q <= 40 and 3 <= s <= 40:
            break
    half = Fraction(180 - big, 2)  # angle BAC
    # An independent route: solve the stated equation by trying every whole x.
    hits = [v for v in range(1, 100) if (p * v + q) + (r * v - s) == 180]
    check(hits == [x], "x is not the only whole solution")
    check(p * x + q == big and r * x - s == 180 - big, "the stated angles disagree with the construction")

    stem = (
        f"{nm} is a rhombus. Opposite angles of a rhombus are equal, consecutive angles add to 180°, and a diagonal "
        f"bisects the angles at the vertices it joins. Angle {a}{b}{c} = ({p}x + {q})° and angle {b}{c}{d} = "
        f"({r}x - {s})°."
    )
    x_cands = []
    if (r - p) != 0:
        x_cands.append((Fraction(s + q, r - p), "equates-consecutive-angles"))
    x_cands.append((Fraction(360 - q + s, p + r), "uses-360-for-two-angles"))
    x_cands.append((Fraction(180 - q - s, p + r), "sign-error-moving-term"))
    x_cands.append((Fraction(180 - q, p + r), "ignored-constant-term"))
    b_cands = [
        (Fraction(x), "reports-the-unknown-not-the-answer"),
        (Fraction(180 - big), "reports-the-neighbouring-angle"),
        (Fraction(360 - big), "uses-360-for-two-angles"),
        (Fraction(big, 2), "halves-the-given-angle"),
    ]
    c_cands = [
        (Fraction(180 - big), "omits-the-final-halving"),
        (Fraction(big, 2), "halves-the-given-angle"),
        (Fraction(90 - big), "uses-the-wrong-triangle-total"),
        (Fraction(big), "copies-the-given-angle"),
    ]
    parts = [
        number_part(
            "Find the value of x.",
            Fraction(x),
            0,
            2,
            [
                f"Angles {a}{b}{c} and {b}{c}{d} are consecutive, so ({p}x + {q}) + ({r}x - {s}) = 180.",
                f"{p + r}x = {180 - q + s}, so x = {x}.",
            ],
            tagged(Fraction(x), x_cands, 0, lo=0, hi=180),
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the size of angle {a}{b}{c}, in degrees, correct to 1 decimal place.",
            Fraction(big),
            1,
            1,
            [f"Angle {a}{b}{c} = {p} × {x} + {q} = {big}°."],
            tagged(Fraction(big), b_cands, 1, lo=0, hi=360),
            follow_through=f"{p}*part_a+{q}",
            sensible_range=(0, 180),
        ),
        number_part(
            f"The diagonal {a}{c} bisects angle {b}{a}{d}. Find the size of angle {b}{a}{c}, in degrees, "
            "correct to 1 decimal place.",
            half,
            1,
            2,
            [
                f"Angle {b}{a}{d} = angle {b}{c}{d} = 180° - {big}° = {180 - big}° (opposite angles).",
                f"Angle {b}{a}{c} = {180 - big}° ÷ 2 = {float(half):.1f}°.",
            ],
            tagged(half, c_cands, 1, lo=0, hi=180),
            follow_through="90 - part_b/2",
            sensible_range=(0, 90),
        ),
    ]
    return dict(text=stem, parts=parts)


def _lengths(rng: random.Random) -> dict:
    nm = rng.choice(_NAMES)
    a, b, c, d = nm
    p, q, z = rng.choice(_TRIPLES)  # AM = p, BM = q, side z
    obj = rng.choice(_OBJECTS)
    check(p * p + q * q == z * z, "not a right-angled triple")
    stem = (
        f"The diagonals of a rhombus bisect each other at right angles. A rhombus-shaped {obj} {nm} has a perimeter "
        f"of {4 * z} cm. Its diagonal {a}{c} is {2 * p} cm long, and the diagonals cross at M."
    )
    perim = 4 * z
    a_cands = [
        (Fraction(perim, 2), "misreads-the-perimeter"),
        (Fraction(perim), "misreads-the-perimeter"),
        (Fraction(perim, 8), "misreads-the-perimeter"),
    ]
    b_cands = [
        (Fraction(z - p), "subtracts-instead-of-using-pythagoras"),
        (Fraction(2 * q), "diagonal-not-halved-or-doubled"),
        (Fraction(z), "reports-the-hypotenuse"),
        (Fraction(math.isqrt(max(z * z - 4 * p * p, 0))) if z > 2 * p else Fraction(0), "uses-whole-diagonals"),
    ]
    c_cands = [
        (Fraction(q), "diagonal-not-halved-or-doubled"),
        (Fraction(2 * p), "uses-the-other-diagonal"),
        (Fraction(2 * (z - p)), "subtracts-instead-of-using-pythagoras"),
        (Fraction(p + q), "adds-instead-of-using-pythagoras"),
    ]
    parts = [
        number_part(
            f"Find the length of side {a}{b}, in cm.",
            Fraction(z),
            0,
            1,
            [f"All four sides of a rhombus are equal, so {a}{b} = {perim} ÷ 4 = {z} cm."],
            tagged(Fraction(z), a_cands, 0, lo=0),
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the length of {b}M, in cm.",
            Fraction(q),
            0,
            2,
            [
                f"{a}M = {2 * p} ÷ 2 = {p} cm, and angle {a}M{b} is a right angle.",
                f"{b}M = √({z}² - {p}²) = {q} cm.",
            ],
            tagged(Fraction(q), b_cands, 0, lo=0),
            follow_through=f"sqrt(part_a**2-{p * p})",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the length of the diagonal {b}{d}, in cm.",
            Fraction(2 * q),
            0,
            1,
            [f"M is the midpoint of {b}{d}, so {b}{d} = 2 × {q} = {2 * q} cm."],
            tagged(Fraction(2 * q), c_cands, 0, lo=0),
            follow_through="2*part_b",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_sc3_quadrilateral_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc3_quadrilateral_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A three-part chain on one rhombus: solve for x then find angles, or find the side then both diagonals."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return rng.choice([_angles, _lengths])(rng)
