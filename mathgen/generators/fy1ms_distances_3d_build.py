"""Generates `fy1ms_distances_3d` questions and the exam chain `exd_x_fy1ms_distances_3d`.

Skill: `ucv1-distances-and-triple-product-3d` (vectors). Distances in three dimensions, and the scalar triple product
with the volumes it gives.

Tiers (topic `fy1ms_distances_3d`; every answer is a number):

* easy   -- the distance from a point to a plane ax + by + cz = d, with normals whose length is a whole number
            (so the denominator is clean) and a signed numerator that is sometimes negative;
* medium -- one of: the distance from a point to a line r = A + t u (cross product, then length); or the scalar triple
            product a . (b x c) and the volume of the parallelepiped on three vectors;
* hard   -- one of: the shortest distance between two skew lines; the volume of a tetrahedron from its four vertices;
            the distance from a vertex to the plane through the other three.

Exam (topic `exd_x_fy1ms_distances_3d`): a solid with edges a, b, c from one corner. (a) the scalar triple product
(2 marks), (b) the volume (1 mark, follow-through from (a)), (c) the area of the face on a and b (2 marks), (d) the
perpendicular height of the solid above that face (2 marks, follow-through from (b) and (c)).

Every key is computed from the formulas and checked a second way with SymPy (exact `Matrix` determinant for the
triple product and exact norms for the distances).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy
from sympy import Matrix

from ..core import attempts, check, generator
from ._fy1ms_common import (
    Fm,
    build_until,
    clear_of_tie,
    dp_text,
    frac_dist,
    ft_close,
    num_dist,
    number_part,
    number_question,
    say,
)
from ._fy1ms_common import cross3 as _cross
from ._fy1ms_common import dot3 as _dot
from ._fy1ms_common import plane_text as _plane
from ._fy1ms_common import sub3 as _sub
from ._fy1ms_common import vec_text as _v

#: Normals whose length is a whole number (Pythagorean triples and quadruples).
_NORMALS = [
    (1, 2, 2),
    (2, 1, 2),
    (2, 2, 1),
    (2, 3, 6),
    (3, 2, 6),
    (6, 3, 2),
    (1, 4, 8),
    (4, 4, 7),
    (2, 6, 9),
    (6, 6, 7),
    (3, 4, 0),
    (0, 3, 4),
    (4, 0, 3),
    (5, 12, 0),
    (0, 5, 12),
    (6, 8, 0),
]
_LENGTH = {n: int(math.isqrt(_dot(n, n))) for n in _NORMALS}


def _length(v) -> float:
    return math.sqrt(_dot(v, v))


def _exact_length(v):
    return sympy.sqrt(_dot(v, v))


# ---------------------------------------------------------------------------
# Easy: point to plane


def _easy(rng: random.Random) -> dict:
    for _ in attempts(3000, "a point-to-plane distance"):
        base = rng.choice(_NORMALS)
        n = [c * rng.choice((1, -1)) if c else 0 for c in base]
        p = [rng.randint(-5, 5) for _ in range(3)]
        d = rng.randint(-9, 9)
        signed = _dot(n, p) - d
        if signed == 0 or abs(signed) > 70 or d == 0:
            continue
        length = _LENGTH[base]
        value = Fraction(abs(signed), length)
        if clear_of_tie(value, 2) and value.denominator != 1:
            break
    # second route: SymPy with exact square roots
    exact = abs(sympy.Integer(signed)) / _exact_length(n)
    check(abs(float(exact) - float(value)) < 1e-12, "distance disagrees with SymPy")
    cands = [
        num_dist(Fraction(signed, length), 2, "distance-signed-left-negative"),
        num_dist(Fraction(abs(signed), _dot(n, n)), 2, "distance-plane-denominator-not-rooted"),
        num_dist(Fraction(abs(_dot(n, p) + d), length), 2, "distance-plane-constant-sign"),
        num_dist(abs(signed), 2, "distance-plane-not-normalised"),
        num_dist(Fraction(abs(signed), sum(abs(c) for c in n)), 2, "distance-plane-denominator-sum-of-coefficients"),
    ]
    text, latex = say(
        "Find the shortest distance from the point ",
        Fm(f"P{_v(p)}"),
        " to the plane ",
        Fm(_plane(n, d)),
        ". Give your answer correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        2,
        cands,
        working=[
            f"|n · P - d| = |{_dot(n, p)} - ({d})| = {abs(signed)}, and |n| = {length}.",
            f"Distance = {abs(signed)}/{length} = {dp_text(value, 2)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Medium: point to line, and the triple product


def _medium_line(rng: random.Random) -> dict:
    for _ in attempts(3000, "a point-to-line distance"):
        a = [rng.randint(-4, 4) for _ in range(3)]
        u = [rng.randint(-3, 3) for _ in range(3)]
        p = [rng.randint(-4, 4) for _ in range(3)]
        ap = _sub(p, a)
        cross = _cross(ap, u)
        if not any(u) or not any(cross):
            continue
        value = math.sqrt(_dot(cross, cross)) / _length(u)
        if not clear_of_tie(value, 2) or value > 12 or abs(_dot(ap, u)) == 0:
            continue
        break
    exact = sympy.sqrt(_dot(cross, cross)) / sympy.sqrt(_dot(u, u))
    check(abs(float(exact) - value) < 1e-12, "distance disagrees with SymPy")
    # Pythagoras gives the same distance from the other side
    perp_sq = _dot(ap, ap) - _dot(ap, u) ** 2 / _dot(u, u)
    check(abs(math.sqrt(perp_sq) - value) < 1e-9, "distance disagrees with Pythagoras")
    cands = [
        num_dist(abs(_dot(ap, u)) / _length(u), 2, "distance-line-dot-product-used"),
        num_dist(math.sqrt(_dot(cross, cross)), 2, "distance-line-not-normalised"),
        num_dist(_length(_cross(p, u)) / _length(u), 2, "distance-line-point-not-shifted"),
        num_dist(_length(ap), 2, "distance-line-to-base-point"),
    ]
    text, latex = say(
        "Find the shortest distance from the point ",
        Fm(f"P{_v(p)}"),
        " to the line ",
        Fm(f"r = {_v(a)} + t{_v(u)}", rf"\mathbf{{r}} = {_v(a)} + t{_v(u)}"),
        ". Give your answer correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        2,
        cands,
        working=[
            f"AP = {_v(ap)}, and AP × u = {_v(cross)}.",
            f"Distance = |AP × u|/|u| = {dp_text(value, 2)}.",
        ],
    )


def _triple(a, b, c) -> int:
    t = _dot(a, _cross(b, c))
    check(t == int(Matrix([a, b, c]).det()), "triple product disagrees with the determinant")
    return t


def _medium_triple(rng: random.Random) -> dict:
    for _ in attempts(3000, "a parallelepiped volume"):
        a, b, c = ([rng.randint(-3, 4) for _ in range(3)] for _ in range(3))
        t = _triple(a, b, c)
        if t == 0 or abs(t) > 60 or abs(t) == 1:
            continue
        break
    volume = abs(t)
    bxc = _cross(b, c)
    slip = _dot(a, [bxc[0], -bxc[1], bxc[2]])
    alt = a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    check(alt == t, "expansion disagrees")
    unsigned_expansion = (
        a[0] * (b[1] * c[2] + b[2] * c[1]) + a[1] * (b[2] * c[0] + b[0] * c[2]) + a[2] * (b[0] * c[1] + b[1] * c[0])
    )
    cands = [
        num_dist(t, 0, "volume-negative-not-absolute"),
        num_dist(abs(slip), 0, "cross-product-middle-sign-slip"),
        num_dist(abs(unsigned_expansion), 0, "determinant-expansion-signs-forgotten"),
        frac_dist(Fraction(volume, 6), "parallelepiped-as-tetrahedron"),
    ]
    base_area = math.sqrt(_dot(bxc, bxc))
    if base_area == int(base_area):
        cands.append(num_dist(int(base_area), 0, "volume-as-base-area-only"))
    text, latex = say(
        rng.choice(
            [
                "A parallelepiped has edges ",
                "A parallelepiped is built on the edges ",
                "A parallelepiped has the three edges ",
            ]
        ),
        Fm(f"a = {_v(a)}"),
        ", ",
        Fm(f"b = {_v(b)}"),
        " and ",
        Fm(f"c = {_v(c)}"),
        " from one corner. Find its volume.",
    )
    return number_question(
        text,
        latex,
        volume,
        0,
        cands,
        working=[f"b × c = {_v(bxc)}.", f"a · (b × c) = {t}, so the volume is |{t}| = {volume}."],
    )


# ---------------------------------------------------------------------------
# Hard: skew lines, tetrahedron volume, a vertex above a face


def _hard_skew(rng: random.Random) -> dict:
    for _ in attempts(3000, "two skew lines"):
        a1, a2 = ([rng.randint(-4, 4) for _ in range(3)] for _ in range(2))
        u1, u2 = ([rng.randint(-3, 3) for _ in range(3)] for _ in range(2))
        n = _cross(u1, u2)
        gap = _sub(a2, a1)
        if not any(n) or _dot(gap, n) == 0:
            continue
        value = abs(_dot(gap, n)) / _length(n)
        if not clear_of_tie(value, 2) or value > 10 or value < 0.3:
            continue
        break
    exact = abs(sympy.Integer(_dot(gap, n))) / sympy.sqrt(_dot(n, n))
    check(abs(float(exact) - value) < 1e-12, "skew distance disagrees with SymPy")
    cands = [
        num_dist(abs(_dot(gap, n)), 2, "skew-distance-not-normalised"),
        num_dist(_length(_cross(gap, u1)) / _length(u1), 2, "skew-distance-one-line-only"),
        num_dist(abs(_dot(gap, u1)) / _length(u1), 2, "distance-line-dot-product-used"),
        num_dist(_length(gap), 2, "skew-distance-between-base-points"),
    ]
    text, latex = say(
        "Line ",
        Fm("L₁", "L_1"),
        " has equation ",
        Fm(f"r = {_v(a1)} + s{_v(u1)}", rf"\mathbf{{r}} = {_v(a1)} + s{_v(u1)}"),
        " and line ",
        Fm("L₂", "L_2"),
        " has equation ",
        Fm(f"r = {_v(a2)} + t{_v(u2)}", rf"\mathbf{{r}} = {_v(a2)} + t{_v(u2)}"),
        ". The lines do not meet. Find the shortest distance between them, correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        2,
        cands,
        working=[
            f"u₁ × u₂ = {_v(n)}, and the gap between base points is {_v(gap)}.",
            f"Distance = |gap · (u₁ × u₂)|/|u₁ × u₂| = {abs(_dot(gap, n))}/{dp_text(_length(n), 3)} = {dp_text(value, 2)}.",
        ],
    )


def _four_points(rng: random.Random):
    for _ in attempts(3000, "four points that are not coplanar"):
        a, b, c, d = ([rng.randint(-3, 4) for _ in range(3)] for _ in range(4))
        ab, ac, ad = _sub(b, a), _sub(c, a), _sub(d, a)
        t = _triple(ab, ac, ad)
        if t != 0 and abs(t) <= 60 and len({tuple(a), tuple(b), tuple(c), tuple(d)}) == 4:
            return a, b, c, d, ab, ac, ad, t
    raise AssertionError("unreachable")  # pragma: no cover


def _hard_tetra(rng: random.Random) -> dict:
    for _ in attempts(500, "a tetrahedron volume away from a tie"):
        a, b, c, d, ab, ac, ad, t = _four_points(rng)
        value = Fraction(abs(t), 6)
        if value.denominator != 1 and clear_of_tie(value, 2):
            break
    cands = [
        num_dist(abs(t), 2, "tetrahedron-volume-factor-forgotten"),
        num_dist(Fraction(abs(t), 3), 2, "tetrahedron-volume-third-of-parallelepiped"),
        num_dist(Fraction(abs(t), 2), 2, "tetrahedron-volume-half-of-parallelepiped"),
        num_dist(Fraction(t, 6), 2, "volume-negative-not-absolute"),
    ]
    text, latex = say(
        "A tetrahedron has vertices ",
        Fm(f"A{_v(a)}"),
        ", ",
        Fm(f"B{_v(b)}"),
        ", ",
        Fm(f"C{_v(c)}"),
        " and ",
        Fm(f"D{_v(d)}"),
        ". Find its volume, correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        2,
        cands,
        working=[
            f"AB = {_v(ab)}, AC = {_v(ac)}, AD = {_v(ad)}, and AB · (AC × AD) = {t}.",
            f"Volume = |{t}|/6 = {dp_text(value, 2)}.",
        ],
    )


def _hard_height(rng: random.Random) -> dict:
    for _ in attempts(500, "a tetrahedron height away from a tie"):
        a, b, c, d, ab, ac, ad, t = _four_points(rng)
        n = _cross(ab, ac)
        if not any(n):
            continue
        value = abs(t) / _length(n)
        if clear_of_tie(value, 2) and 0.3 < value < 10:
            break
    exact = abs(sympy.Integer(t)) / sympy.sqrt(_dot(n, n))
    check(abs(float(exact) - value) < 1e-12, "height disagrees with SymPy")
    area = _length(n) / 2
    cands = [
        num_dist(abs(t), 2, "distance-plane-not-normalised"),
        num_dist(abs(t) / 6 / area, 2, "height-pyramid-factor-forgotten"),
        num_dist(abs(_dot(n, ad)) / _dot(n, n), 2, "distance-plane-denominator-not-rooted"),
        num_dist(abs(t) / (3 * _length(n)), 2, "height-pyramid-factor-applied-twice"),
    ]
    text, latex = say(
        "The points ",
        Fm(f"A{_v(a)}"),
        ", ",
        Fm(f"B{_v(b)}"),
        " and ",
        Fm(f"C{_v(c)}"),
        " lie in a plane. Find the shortest distance from ",
        Fm(f"D{_v(d)}"),
        " to that plane, correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        2,
        cands,
        working=[
            f"AB × AC = {_v(n)}, and AD · (AB × AC) = {t}.",
            f"Distance = |{t}|/|{_v(n)}| = {dp_text(value, 2)}.",
        ],
    )


@generator(
    "fy1ms_distances_3d",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_distances_3d(rng: random.Random, tier: str = "easy") -> dict:
    """3D distances and volumes: point to plane (easy), point to line or a triple product (medium), skew lines, tetrahedra (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy 3D distance")
    if tier == "medium":
        return build_until(
            rng, lambda r: _medium_line(r) if r.random() < 0.5 else _medium_triple(r), "a medium 3D distance"
        )
    if tier == "hard":

        def hard(r: random.Random) -> dict:
            pick = r.random()
            return _hard_skew(r) if pick < 0.4 else _hard_tetra(r) if pick < 0.7 else _hard_height(r)

        return build_until(rng, hard, "a hard 3D distance")
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain: a solid on three edge vectors


@generator("exd_x_fy1ms_distances_3d", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_distances_3d(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the triple product, the volume, the area of a face, the height above it (7 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a triple-product exam question")


def _exam_chain(rng: random.Random) -> dict:
    for _ in attempts(3000, "a solid with clear numbers"):
        a, b, c = ([rng.randint(-3, 4) for _ in range(3)] for _ in range(3))
        t = _triple(a, b, c)
        axb = _cross(a, b)
        if t == 0 or abs(t) > 40 or not any(axb) or abs(t) == 1:
            continue
        area = _length(axb)
        height = abs(t) / area
        if clear_of_tie(area, 2) and clear_of_tie(height, 2) and 0.2 < height < 8 and area < 25:
            break
    volume = abs(t)
    dp = 2
    area_text, height_text = dp_text(area, dp), dp_text(height, dp)
    ft_close(volume / float(area_text), height_text, dp)
    a_cands = [
        num_dist(-t, 0, "triple-product-order-reversed"),
        num_dist(_dot(a, [_cross(b, c)[0], -_cross(b, c)[1], _cross(b, c)[2]]), 0, "cross-product-middle-sign-slip"),
        num_dist(_dot(_cross(a, b), c) + _dot(a, b), 0, "triple-product-extra-dot-term"),
        num_dist(_dot(a, b) * _dot(b, c), 0, "triple-product-dots-multiplied"),
    ]
    b_cands = [
        num_dist(t, 0, "volume-negative-not-absolute"),
        frac_dist(Fraction(volume, 6), "parallelepiped-as-tetrahedron"),
        num_dist(volume * 2, 0, "volume-doubled"),
        num_dist(volume + abs(_dot(a, b)), 0, "volume-extra-dot-term"),
    ]
    c_cands = [
        num_dist(abs(_dot(a, b)), dp, "area-from-dot-product"),
        num_dist(_length(a) * _length(b), dp, "area-as-product-of-lengths"),
        num_dist(_dot(axb, axb), dp, "area-cross-magnitude-squared"),
        num_dist(_length(axb) / 2, dp, "area-parallelogram-as-triangle"),
    ]
    d_cands = [
        num_dist(volume * area, dp, "height-volume-multiplied-by-area"),
        num_dist(area / volume, dp, "height-ratio-inverted"),
        num_dist(volume / _length(_cross(b, c)), dp, "height-over-the-wrong-face"),
        num_dist(3 * volume / area, dp, "height-pyramid-factor-applied-to-prism"),
    ]
    stem, stem_latex = say(
        rng.choice(
            [
                "A solid crystal is a parallelepiped with edges ",
                "A crystal grows in the shape of a parallelepiped with edges ",
                "A parallelepiped-shaped crystal has edges ",
            ]
        ),
        Fm(f"a = {_v(a)}"),
        ", ",
        Fm(f"b = {_v(b)}"),
        " and ",
        Fm(f"c = {_v(c)}"),
        " from one corner, measured in nanometres.",
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            number_part(
                "Find the scalar triple product a · (b × c).",
                t,
                0,
                a_cands,
                marks=2,
                prompt_latex=r"\text{Find the scalar triple product }\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})\text{.}",
                working=[f"b × c = {_v(_cross(b, c))}, so a · (b × c) = {t}."],
            ),
            number_part(
                "Hence find the volume of the crystal, in cubic nanometres.",
                volume,
                0,
                b_cands,
                marks=1,
                follow_through="Abs(part_a)",
                sensible_range=(0, None),
                working=[f"Volume = |{t}| = {volume}."],
            ),
            number_part(
                "Find the area of the face spanned by a and b, in square nanometres, correct to 2 decimal places.",
                area,
                dp,
                c_cands,
                marks=2,
                sensible_range=(0, None),
                working=[f"a × b = {_v(axb)}, so the area is |a × b| = {area_text}."],
            ),
            number_part(
                "Hence find the perpendicular distance from the opposite face to that face, in nanometres, correct to 2 decimal places.",
                height,
                dp,
                d_cands,
                marks=2,
                follow_through="part_b/part_c",
                sensible_range=(0, None),
                working=[f"Height = volume/area = {volume}/{area_text} = {height_text}."],
            ),
        ],
    )
