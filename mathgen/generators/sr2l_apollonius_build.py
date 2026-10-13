"""The Apollonius circle |z - a| = k|z - b| and where it meets a line: `sr2l_apollonius` (easy, medium, hard) and
`sr2l_apollonius_exam`.

Skill: locus-apollonius-and-intersection. When k is not 1 the points z whose distances from a and b are in the ratio
k : 1 form a circle, not a line. With k = 2 or 3 the points are built so every answer is exact: a = b + (k + 1)u for a whole-number
vector u, which puts the point of the circle between a and b at P = b + u and the other end of the diameter on the line
through them at Q = b - (k + 1)u/(k - 1).

* easy: a and b on the real axis, k = 2 or 3. Give the centre of the circle (a pair).
* medium: a and b anywhere, the relation written in one of three ways (|z - a| = k|z - b|, k|z - b| = |z - a| or the squared
  form). Give the centre.
* hard: give the point of the locus that lies between a and b on the line through them (a pair). The two end-points of the
  diameter on that line divide ab in the ratio k : 1, one inside and one outside.
* exam: k = 2 and u of whole-number length, so the radius is a whole number. The centre (pair), the radius, the imaginary parts of the
  two points where the circle meets a stated vertical line (a list), and the length of the chord that line cuts (follow-through
  from the radius).

Each key is re-derived by evaluating the printed relation at points of the claimed circle (it holds there and fails just
off it), and the point or chord is checked on the printed relation too.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from . import _sr2l_kit as kit

_ASK_CENTRE = "State the centre of the circle as an ordered pair (x, y), where z = x + iy."


def _geometry(b, u, k):
    """(a, centre, radius, internal point, external point) for |z - a| = k|z - b| with a = b + (k + 1)u."""
    a = (b[0] + (k + 1) * u[0], b[1] + (k + 1) * u[1])
    p = (Fraction(b[0] + u[0]), Fraction(b[1] + u[1]))
    q = (b[0] - Fraction((k + 1) * u[0], k - 1), b[1] - Fraction((k + 1) * u[1], k - 1))
    centre = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    radius = math.hypot(float(p[0] - q[0]), float(p[1] - q[1])) / 2
    return a, centre, radius, p, q


def _relation_forms(a, b, k):
    """Three equivalent printed forms of |z - a| = k|z - b|."""
    za, zb = kit.mod(kit.z_minus(*a)), kit.mod(kit.z_minus(*b))
    return (
        rf"{za} = {k}{zb}",
        rf"{k}{zb} = {za}",
        rf"{za}^2 = {k * k}{zb}^2",
    )


def _verify(relation, centre, radius, p) -> None:
    kit.check_locus(kit.plain(relation), centre, radius)
    check(
        kit.residual(kit.plain(relation), complex(float(p[0]), float(p[1]))) < 1e-9,
        "the internal point is not on the locus",
    )


def _distractors_centre(a, b, k, centre, p, q, u):
    mid = (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))
    return [
        (mid[0], mid[1], "sr2l-midpoint-for-centre"),
        (b[0], b[1], "sr2l-closer-point-for-centre"),
        (p[0], p[1], "sr2l-internal-point-for-centre"),
        (a[0], a[1], "sr2l-closer-point-for-centre"),
        ((a[0] + k * b[0]) / Fraction(k + 1), (a[1] + k * b[1]) / Fraction(k + 1), "sr2l-internal-point-for-centre"),
        (-centre[0], -centre[1], "sr2l-centre-sign-flipped"),
        (q[0], q[1], "sr2l-external-point-for-centre"),
    ]


def _draw(rng, k, tier):
    for _ in attempts(500, "points for an Apollonius circle"):
        if tier == "easy":
            b = (rng.randint(-6, 6), 0)
            u = (rng.choice([v for v in (-3, -2, -1, 1, 2, 3, 4) if v]), 0)
        else:
            b = (rng.randint(-6, 6), rng.randint(-6, 6))
            u = (rng.randint(-3, 3), rng.randint(-3, 3))
        if k == 3 and tier != "hard" and (u[0] % 2 or u[1] % 2):
            continue  # a centre b - u/2 is a whole-number point only for an even u
        if u != (0, 0) and b != (0, 0) and (tier == "easy" or (u[0] != 0 and u[1] != 0)):
            a = (b[0] + (k + 1) * u[0], b[1] + (k + 1) * u[1])
            if max(map(abs, a)) <= 14 and a != (0, 0):
                return a, b, u
    raise AssertionError("unreachable")  # pragma: no cover


def _centre_question(rng, tier):
    k = rng.choice((2, 2, 3)) if tier == "easy" else rng.choice((2, 3))
    a, b, u = _draw(rng, k, tier)
    _, centre, radius, p, q = _geometry(b, u, k)
    forms = _relation_forms(a, b, k)
    rel = forms[0] if tier == "easy" else rng.choice(forms)
    _verify(rel, centre, radius, p)
    fields = kit.pair_fields(*centre)
    stems = (
        "The locus of the complex number z is given by {rel}, which is a circle. " + _ASK_CENTRE,
        "A complex number z satisfies {rel}. The locus of z is a circle. " + _ASK_CENTRE,
        "The points z with {rel} lie on a circle. " + _ASK_CENTRE,
    )
    working = [
        f"Write z = x + iy and square both sides: the points {kit.cplx(*a)} and {kit.cplx(*b)} are at distances in the ratio {k} : 1.",
        f"The circle's diameter lies along the line through them. It meets the segment at {kit.cplx(*p)} and the line again at {kit.cplx(q[0], q[1])}.",
        f"The centre is the midpoint of that diameter: {fields['answer_text']}.",
    ]
    return dict(
        **kit.prompt(phrase(rng, stems, rel=kit.maths(rel))),
        working=working,
        distractors=kit.tagged("pair", fields, kit.pair_cands(_distractors_centre(a, b, k, centre, p, q, u))),
        **fields,
    )


def _hard(rng):
    k = rng.choice((2, 3))
    a, b, u = _draw(rng, k, "hard")
    _, centre, radius, p, q = _geometry(b, u, k)
    rel = rng.choice(_relation_forms(a, b, k))
    _verify(rel, centre, radius, p)
    fields = kit.pair_fields(*p)
    mid = (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))
    inverted = ((k * a[0] + b[0]) / Fraction(k + 1), (k * a[1] + b[1]) / Fraction(k + 1))
    cands = [
        (inverted[0], inverted[1], "sr2l-ratio-inverted"),
        (mid[0], mid[1], "sr2l-midpoint-for-centre"),
        (centre[0], centre[1], "sr2l-centre-for-point"),
        (q[0], q[1], "sr2l-external-point-for-internal"),
        (b[0] + 2 * u[0], b[1] + 2 * u[1], "sr2l-ratio-inverted"),
    ]
    stem = (
        rf"The locus of the complex number z is the circle \({rel}\). The circle meets the line segment joining "
        f"{kit.cplx(*a)} and {kit.cplx(*b)} at exactly one point. State that point as an ordered pair (x, y), where z = x + iy."
    )
    return dict(
        **kit.prompt(stem),
        working=[
            f"The point is {k} times as far from a = {kit.cplx(*a)} as it is from b = {kit.cplx(*b)}, so it divides the segment from b to a in the ratio 1 : {k}.",
            f"P = (a + {k}b)/{k + 1} = {kit.cplx(*p)}.",
        ],
        distractors=kit.tagged("pair", fields, kit.pair_cands(cands)),
        **fields,
    )


_TIERS = {
    "easy": lambda rng: _centre_question(rng, "easy"),
    "medium": lambda rng: _centre_question(rng, "medium"),
    "hard": _hard,
}


@generator("sr2l_apollonius", level=18, presets={t: {"tier": t} for t in _TIERS})
def sr2l_apollonius(rng, tier: str = "easy"):
    """The centre of an Apollonius circle, or the point of it between the two given points."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return kit.redraw(rng, _TIERS[tier])


# -- exam ------------------------------------------------------------------------------------------------------------

_TRIPLES = {
    10: (6, 8),
    20: (12, 16),
    26: (10, 24),
}  # radius -> the two whole offsets t with t^2 + s^2 = r^2 (t, s) and (s, t)


@generator("sr2l_apollonius_exam", level=18, presets={"exam": {}})
def sr2l_apollonius_exam(rng):
    """|z - a| = 2|z - b|: centre, radius, where it meets a vertical line (a list), and the chord (follow-through)."""
    return kit.redraw(rng, _exam)


def _exam(rng):
    k = 2
    leg = rng.choice(((3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5)))
    u = (leg[0] * rng.choice((1, -1)), leg[1] * rng.choice((1, -1)))
    b = (rng.randint(-5, 5), rng.randint(-5, 5))
    a, centre, radius, p, q = _geometry(b, u, k)
    r = int(round(radius))
    check(abs(radius - r) < 1e-9, "the radius must be whole")
    centre = (int(centre[0]), int(centre[1]))
    t, s = _TRIPLES[r]
    if rng.random() < 0.5:
        t, s = s, t
    side = rng.choice((1, -1))
    x_line = centre[0] + side * t
    rel = rng.choice(_relation_forms(a, b, k)[:2])
    _verify(rel, centre, r, p)
    ys = [centre[1] + s, centre[1] - s]
    for y in ys:  # the two points lie on the printed locus
        check(
            kit.residual(kit.plain(rel), complex(x_line, y)) < 1e-9 * (1 + x_line**2 + y**2),
            "an intersection point is off the locus",
        )
    chord = 2 * s
    d = math.hypot(*u)
    part_a = kit.pair_part(
        "State the centre of the circle as an ordered pair (x, y).",
        centre[0],
        centre[1],
        2,
        [
            f"Square both sides and complete the square, or use the two ends of the diameter on the line through {kit.cplx(*a)} and {kit.cplx(*b)}."
        ],
        [(w[0], w[1], w[2]) for w in _distractors_centre(a, b, k, centre, p, q, u)],
    )
    part_b = kit.number_part(
        "Find the radius of the circle.",
        r,
        0,
        2,
        [
            f"The diameter runs from {kit.cplx(p[0], p[1])} to {kit.cplx(q[0], q[1])}, so its length is {2 * r} and the radius is {r}."
        ],
        [
            (int(d * 3), "sr2l-distance-between-points-for-radius"),
            (int(d), "sr2l-distance-ratio-for-radius"),
            (r * r, "sr2l-radius-squared"),
            (2 * r, "sr2l-diameter-for-radius"),
        ],
        sensible=(0, None),
    )
    part_c = kit.list_part(
        f"The circle crosses the line Re(z) = {x_line} at two points. Find the imaginary parts of those two points.",
        ys,
        2,
        [f"With x = {x_line}, (y - ({centre[1]}))^2 = {r * r} - {t * t} = {s * s}, so y = {centre[1]} ± {s}."],
        [
            ([centre[1] + r, centre[1] - r], "sr2l-offset-ignored"),
            ([centre[1] + t, centre[1] - t], "sr2l-wrong-offset-used"),
            ([centre[0] + s, centre[0] - s], "sr2l-centre-real-for-imaginary"),
            ([centre[1] + s, centre[1] + s - 1], "sr2l-one-intersection-only"),
        ],
    )
    part_d = kit.number_part(
        "Find the length of the chord of the circle that lies on this line.",
        chord,
        0,
        2,
        [f"The chord joins the two points, so its length is 2 x {s} = {chord} (or 2 sqrt({r}^2 - {t}^2))."],
        [
            (s, "sr2l-chord-half-only"),
            (r + t, "sr2l-chord-by-sum"),
            (2 * t, "sr2l-wrong-offset-used"),
            (2 * r, "sr2l-diameter-for-radius"),
        ],
        follow=f"2*sqrt(part_b**2 - {t * t})",
        sensible=(0, None),
    )
    stem = rf"The locus of the complex number z = x + iy is the circle \({rel}\)."
    return dict(**kit.prompt(stem), parts=[part_a, part_b, part_c, part_d])
