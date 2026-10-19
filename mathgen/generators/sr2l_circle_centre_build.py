"""The centre of a circle locus, from the form the condition is written in: `sr2l_circle_centre` (easy, medium, hard)
and `sr2l_circle_centre_exam`.

Skill: locus-circle-centre-radius. The answer to every practice question is the centre as an ordered pair (x, y) with
z = x + iy; the radius and distances come in the exam chain, where each is a number.

* easy: |z - w| = r written three ways: with the centre's signs to resolve (z + 3 - 2i), with the centre bracketed and
  negative (z - (-3 + 2i)), or on an axis (z + 3, z - 2i).
* medium: the form is not yet |z - w| = r. Either |kz + w| = m, where the centre is w/k with its sign reversed, or
  |z - w| = |u|, where the complex number on the right is a radius and not a centre.
* hard: the circle has to be recognised before its centre can be read: a Cartesian equation with squared terms
  (sometimes scaled by 2), z z-bar + w-bar z + w z-bar = k, the image of a circle under w = kz + b, or a circle with
  two stated points as the ends of a diameter.
* exam: a circle given as |kz + w| = m or by a Cartesian equation, and a point outside it. The centre (pair), the radius,
  the distance from the point to the centre, and the shortest distance from the point to the circle (follow-through).

Every key is re-derived by testing the stated condition at points of the claimed circle with Python's complex
arithmetic: the condition holds at every one of them and fails just inside and just outside.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms, phrase

from . import _sr2l_kit as kit

_ASK = "State the centre of the circle as an ordered pair (x, y), where z = x + iy."


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng):
    form = rng.choice(("expanded", "bracketed", "axis"))
    for _ in attempts(500, "a centre whose misreadings are all distinct"):
        r = rng.randint(2, 9)
        if form == "axis":
            if rng.random() < 0.5:
                c = (rng.choice([v for v in range(-8, 9) if v]), 0)
            else:
                c = (0, rng.choice([v for v in range(-8, 9) if v]))
        else:
            c = (rng.randint(-8, 8), rng.randint(-8, 8))
        if c[0] != 0 and c[1] != 0 and abs(c[0]) != abs(c[1]) or form == "axis":
            if c[0] < 0 or c[1] < 0 or form != "bracketed":
                break
    if form == "bracketed":
        inner = f"z - ({kit.cplx(*c)})"
    else:
        inner = kit.z_minus(*c)
    rel = kit.circle_rel(0, 0, r).replace(kit.mod(kit.z_minus(0, 0)), kit.mod(inner))
    key = kit.pair_fields(*c)
    wrong = [
        (-c[0], -c[1], "sr2l-centre-sign-flipped"),
        (c[1], c[0], "sr2l-real-imaginary-swapped"),
        (-c[1], -c[0], "sr2l-real-imaginary-swapped"),
        (c[0], -c[1], "sr2l-centre-sign-flipped"),
        (-c[0], c[1], "sr2l-centre-sign-flipped"),
    ]
    kit.check_locus(kit.plain(rel), c, r)
    stems = (
        "The locus of the complex number z is given by {rel}. " + _ASK,
        "A complex number z satisfies {rel}. " + _ASK,
        "The points z satisfying {rel} form a circle. " + _ASK,
    )
    return dict(
        **kit.prompt(phrase(rng, stems, rel=kit.maths(rel))),
        working=[
            f"|z - w| = r is a circle with centre w and radius r. Here z - w = {inner}, so w = {kit.cplx(*c)}.",
            f"The centre is the point {key['answer_text']}.",
        ],
        distractors=kit.tagged("pair", key, kit.pair_cands(wrong)),
        **key,
    )


# -- medium ----------------------------------------------------------------------------------------------------------


def _medium_scaled(rng):
    k = rng.choice((2, 3, 4))
    c = (rng.randint(-6, 6), rng.randint(-6, 6))
    r = rng.randint(1, 5)
    if c[0] == 0 or c[1] == 0 or abs(c[0]) == abs(c[1]):
        c = (rng.choice((-5, -3, -2, 2, 3, 4)), rng.choice((-4, -1, 1, 5, 6)))
    a, b = -k * c[0], -k * c[1]
    inner = join_terms([(k, "z"), (a, ""), (b, "i")])
    rel = rf"{kit.mod(inner)} = {k * r}"
    wrong = [
        (-a, -b, "sr2l-scale-factor-not-divided"),
        (a, b, "sr2l-centre-sign-flipped"),
        (-c[0], -c[1], "sr2l-centre-sign-flipped"),
        (Fraction(-a, 1), c[1], "sr2l-scale-factor-not-divided"),
        (c[0], -b, "sr2l-scale-factor-not-divided"),
    ]
    kit.check_locus(kit.plain(rel), c, r)
    return (
        rel,
        c,
        wrong,
        [
            f"Factor out {k}: |{inner}| = {k}|z - ({kit.cplx(*c)})|, so the condition is |z - ({kit.cplx(*c)})| = {r}.",
            "The centre is the complex number subtracted from z.",
        ],
    )


def _medium_modulus(rng):
    for _ in attempts(500, "a centre and a radius number that cannot be confused"):
        c = (rng.randint(-7, 7), rng.randint(-7, 7))
        u = rng.choice([(3, 4), (4, 3), (-3, 4), (3, -4), (5, 12), (-5, 12), (12, 5), (6, 8), (8, 6)])
        u = (u[0] * rng.choice((1, -1)), u[1] * rng.choice((1, -1)))
        if c[0] != 0 and c[1] != 0 and c != u and c != (-u[0], -u[1]):
            break
    r = int(math.hypot(*u))
    rel = rf"{kit.mod(kit.z_minus(*c))} = {kit.mod(kit.cplx(*u))}"
    wrong = [
        (u[0], u[1], "sr2l-radius-modulus-as-centre"),
        (-c[0], -c[1], "sr2l-centre-sign-flipped"),
        (-u[0], -u[1], "sr2l-radius-modulus-as-centre"),
        (c[0] + u[0], c[1] + u[1], "sr2l-radius-modulus-as-centre"),
    ]
    kit.check_locus(kit.plain(rel), c, r)
    return (
        rel,
        c,
        wrong,
        [
            f"The right-hand side {kit.cplx(*u)} is a fixed number: |{kit.cplx(*u)}| = {r}, the radius.",
            f"The left-hand side gives the centre {kit.cplx(*c)}.",
        ],
    )


def _medium(rng):
    rel, c, wrong, working = rng.choice((_medium_scaled, _medium_modulus))(rng)
    key = kit.pair_fields(*c)
    stems = (
        "The locus of the complex number z is given by {rel}. " + _ASK,
        "A complex number z satisfies {rel}. " + _ASK,
        "The points z satisfying {rel} form a circle. " + _ASK,
    )
    return dict(
        **kit.prompt(phrase(rng, stems, rel=kit.maths(rel))),
        working=working + [f"The centre is {key['answer_text']}."],
        distractors=kit.tagged("pair", key, kit.pair_cands(wrong)),
        **key,
    )


# -- hard ------------------------------------------------------------------------------------------------------------


def _general_form(c, r, k):
    """k(x^2 + y^2) + Dx + Ey + F = 0 as tex, with D = -2k cx, E = -2k cy, F = k(cx^2 + cy^2 - r^2)."""
    terms = [
        (k, "x^2"),
        (k, "y^2"),
        (-2 * k * c[0], "x"),
        (-2 * k * c[1], "y"),
        (k * (c[0] ** 2 + c[1] ** 2 - r * r), ""),
    ]
    return join_terms(terms) + " = 0"


def _hard_cartesian(rng):
    k = rng.choice((1, 1, 2))
    r = rng.randint(2, 6)
    c = (rng.choice([v for v in range(-6, 7) if v]), rng.choice([v for v in range(-6, 7) if v]))
    eq = _general_form(c, r, k)
    d, e = -2 * k * c[0], -2 * k * c[1]
    wrong = [
        (-d, -e, "sr2l-centre-not-halved"),
        (d, e, "sr2l-centre-sign-flipped"),
        (Fraction(-d, 2), Fraction(-e, 2), "sr2l-coefficient-scale-ignored")
        if k != 1
        else (-c[0], -c[1], "sr2l-centre-sign-flipped"),
        (Fraction(d, 2), Fraction(e, 2), "sr2l-centre-sign-flipped"),
        (c[1], c[0], "sr2l-real-imaginary-swapped"),
    ]
    kit.check_locus(kit.plain(eq), c, r)
    stem = rf"The complex number z = x + iy satisfies \({eq}\). The locus of z is a circle. " + _ASK
    return (
        stem,
        c,
        wrong,
        [
            "Divide by the coefficient of x^2 and complete the square in x and in y.",
            f"This gives ({kit.shifted('x', c[0])})^2 + ({kit.shifted('y', c[1])})^2 = {r * r}.",
        ],
    )


def _hard_conjugate(rng):
    r = rng.randint(2, 6)
    c = (rng.choice([v for v in range(-5, 6) if v]), rng.choice([v for v in range(-5, 6) if v]))
    w = (-c[0], -c[1])  # z z-bar + conj(w) z + w z-bar = r^2 - |w|^2, centre -w
    rhs = r * r - (w[0] ** 2 + w[1] ** 2)
    lhs = rf"z\,\bar{{z}} + ({kit.cplx(w[0], -w[1])})z + ({kit.cplx(*w)})\bar{{z}}"
    wrong = [
        (w[0], w[1], "sr2l-centre-sign-flipped"),
        (c[0], -c[1], "sr2l-conjugate-coefficient"),
        (-c[0], c[1], "sr2l-conjugate-coefficient"),
        (c[1], c[0], "sr2l-real-imaginary-swapped"),
    ]

    kit.check_locus(kit.plain(rf"{lhs} = {rhs}"), c, r)
    stem = rf"The complex number z satisfies \({lhs} = {rhs}\). The locus of z is a circle. " + _ASK
    return (
        stem,
        c,
        wrong,
        [
            "Compare with |z - c|^2 = (z - c)(z̄ - c̄) = z z̄ - c̄ z - c z̄ + |c|^2.",
            f"The coefficient of z̄ is {kit.cplx(*w)}, which is -c, so c = {kit.cplx(*c)}.",
        ],
    )


def _hard_image(rng):
    r = rng.randint(1, 5)
    c = (rng.randint(-4, 4), rng.randint(-4, 4))
    b = (rng.randint(-5, 5), rng.randint(-5, 5))
    scale = rng.choice((2, 3, -1, -2, "i", "-i"))
    if scale == "i":
        img = (-c[1] + b[0], c[0] + b[1])
        form = join_terms([(1, "iz"), (b[0], ""), (b[1], "i")])
        mult = 1j
    elif scale == "-i":
        img = (c[1] + b[0], -c[0] + b[1])
        form = join_terms([(-1, "iz"), (b[0], ""), (b[1], "i")])
        mult = -1j
    else:
        img = (scale * c[0] + b[0], scale * c[1] + b[1])
        form = join_terms([(scale, "z"), (b[0], ""), (b[1], "i")])
        mult = scale
    if img == c or c == (0, 0):
        return _hard_image(rng)
    check(abs(complex(*img) - (mult * complex(*c) + complex(*b))) < 1e-9, "image centre disagrees with w = kz + b")
    for k in range(12):  # every image of a point on the circle lies at distance |scale| r from the claimed centre
        z = complex(*c) + r * cmath.exp(1j * (k + 0.3))
        w = mult * z + complex(*b)
        check(abs(abs(w - complex(*img)) - abs(mult) * r) < 1e-9, "image circle disagrees")
    only_scaled = (
        (scale if not isinstance(scale, str) else 1) * c[0],
        (scale if not isinstance(scale, str) else 1) * c[1],
    )
    wrong = [
        (c[0], c[1], "sr2l-transformation-ignored"),
        (c[0] + b[0], c[1] + b[1], "sr2l-translation-only"),
        (only_scaled[0], only_scaled[1], "sr2l-translation-only"),
        (img[1], img[0], "sr2l-real-imaginary-swapped"),
        (-img[0], -img[1], "sr2l-centre-sign-flipped"),
    ]
    stem = (
        rf"The complex number z moves on the circle \({kit.mod(kit.z_minus(*c))} = {r}\). "
        rf"The complex number w is defined by \(w = {form}\), so w moves on a circle too. "
        "State the centre of that circle as an ordered pair (x, y), where w = x + iy."
    )
    return (
        stem,
        img,
        wrong,
        [
            f"The circle for z has centre {kit.cplx(*c)}. The centre moves in the same way as every point does.",
            f"w = {form} at z = {kit.cplx(*c)} gives {kit.cplx(*img)}.",
        ],
    )


def _hard_diameter(rng):
    for _ in attempts(500, "the ends of a diameter"):
        a = (rng.randint(-8, 8), rng.randint(-8, 8))
        b = (rng.randint(-8, 8), rng.randint(-8, 8))
        if (
            a != b
            and a[0] != b[0]
            and a[1] != b[1]
            and (a[0] + a[1] + b[0] + b[1]) % 2 == 0
            or a != b
            and rng.random() < 0.25
        ):
            break
    c = (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))
    check(complex(*a) + complex(*b) == 2 * complex(float(c[0]), float(c[1])), "midpoint disagrees")
    wrong = [
        (a[0] + b[0], a[1] + b[1], "sr2l-midpoint-not-halved"),
        (Fraction(a[0] - b[0], 2), Fraction(a[1] - b[1], 2), "sr2l-difference-for-midpoint"),
        (a[0] - b[0], a[1] - b[1], "sr2l-difference-for-midpoint"),
        (Fraction(a[0] + b[1], 2), Fraction(a[1] + b[0], 2), "sr2l-real-imaginary-swapped"),
    ]
    stem = (
        rf"In the complex plane the points representing \({kit.cplx(*a)}\) and \({kit.cplx(*b)}\) are the ends of a "
        "diameter of a circle. " + _ASK
    )
    return (
        stem,
        c,
        wrong,
        [
            "The centre of a circle is the midpoint of any diameter: add the ends and halve.",
        ],
    )


def _hard(rng):
    stem, c, wrong, working = rng.choice((_hard_cartesian, _hard_conjugate, _hard_image, _hard_diameter))(rng)
    key = kit.pair_fields(*c)
    return dict(
        **kit.prompt(stem),
        working=working + [f"The centre is {key['answer_text']}."],
        distractors=kit.tagged("pair", key, kit.pair_cands(wrong)),
        **key,
    )


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("sr2l_circle_centre", level=18, presets={t: {"tier": t} for t in _TIERS})
def sr2l_circle_centre(rng, tier: str = "easy"):
    """The centre of a circle locus, as an ordered pair, from the form the condition is written in."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return kit.redraw(rng, _TIERS[tier])


# -- exam ------------------------------------------------------------------------------------------------------------


@generator("sr2l_circle_centre_exam", level=18, presets={"exam": {}})
def sr2l_circle_centre_exam(rng):
    return kit.redraw(rng, _exam)


def _exam(rng):
    """A circle locus and a point outside it: centre, radius, distance to the centre, distance to the circle."""
    for _ in attempts(2000, "a circle with a point clearly outside it"):
        r = rng.randint(2, 7)
        c = (rng.choice([v for v in range(-6, 7) if v]), rng.choice([v for v in range(-6, 7) if v]))
        pt = (rng.randint(-9, 9), rng.randint(-9, 9))
        dist = math.hypot(pt[0] - c[0], pt[1] - c[1])
        if dist - r >= 1.5 and abs(dist * 100 - round(dist * 100)) > 1e-6:
            break
    if rng.random() < 0.5:
        k = rng.choice((2, 3))
        a, b = -k * c[0], -k * c[1]
        rel = rf"{kit.mod(join_terms([(k, 'z'), (a, ''), (b, 'i')]))} = {k * r}"
        how = f"Dividing by {k} gives |z - ({kit.cplx(*c)})| = {r}."
        radius_wrong = [
            (k * r, "sr2l-scale-factor-not-divided"),
            (r * r, "sr2l-radius-squared"),
            (2 * r, "sr2l-diameter-for-radius"),
        ]
        kit.check_locus(kit.plain(rel), c, r)
        stem = (
            rf"The locus of the complex number z is the circle \({rel}\). The point P represents \({kit.cplx(*pt)}\)."
        )
    else:
        eq = _general_form(c, r, 1)
        radius_wrong = [(r * r, "sr2l-radius-squared"), (2 * r, "sr2l-diameter-for-radius")]
        how = f"Completing the square gives ({kit.shifted('x', c[0])})^2 + ({kit.shifted('y', c[1])})^2 = {r * r}."
        kit.check_locus(kit.plain(eq), c, r)
        stem = (
            rf"The complex number z = x + iy has a locus given by the equation \({eq}\), which is a circle. "
            rf"The point P represents \({kit.cplx(*pt)}\)."
        )
    dx, dy = pt[0] - c[0], pt[1] - c[1]
    part_a = kit.pair_part(
        "State the centre of the circle as an ordered pair (x, y).",
        c[0],
        c[1],
        2,
        [how, f"The centre is ({c[0]}, {c[1]})."],
        [
            (-c[0], -c[1], "sr2l-centre-sign-flipped"),
            (c[1], c[0], "sr2l-real-imaginary-swapped"),
            (c[0], -c[1], "sr2l-centre-sign-flipped"),
        ],
    )
    part_b = kit.number_part(
        "Find the radius of the circle.",
        r,
        0,
        1,
        [f"The right-hand side of |z - c| = r gives r = {r}."],
        radius_wrong,
        sensible=(0, None),
    )
    part_c = kit.number_part(
        "Find the distance from P to the centre of the circle, correct to 2 decimal places.",
        dist,
        2,
        2,
        [
            f"d = sqrt(({pt[0]} - ({c[0]}))^2 + ({pt[1]} - ({c[1]}))^2) = sqrt({dx * dx + dy * dy}) = {kit.number_text(dist, 2)}."
        ],
        [
            (abs(dx) + abs(dy), "sr2l-components-added"),
            (dx * dx + dy * dy, "sr2l-square-root-omitted"),
            (math.hypot(pt[0] + c[0], pt[1] + c[1]), "sr2l-centre-sign-flipped"),
            (math.hypot(*pt), "sr2l-distance-from-origin"),
        ],
        sensible=(0, None),
    )
    part_d = kit.number_part(
        "Find the shortest distance from P to the circle, correct to 2 decimal places.",
        dist - r,
        2,
        2,
        [
            f"P is outside the circle, so the shortest distance is the distance to the centre less the radius: {kit.number_text(dist, 2)} - {r}."
        ],
        [
            (dist, "sr2l-distance-to-centre-not-circle"),
            (dist + r, "sr2l-nearest-and-farthest-confused"),
            (dist - 2 * r, "sr2l-diameter-for-radius"),
            (r - dist, "sr2l-subtracted-the-wrong-way"),
        ],
        follow=f"part_c - {r}",
        sensible=(0, None),
    )
    return dict(**kit.prompt(stem), parts=[part_a, part_b, part_c, part_d])
