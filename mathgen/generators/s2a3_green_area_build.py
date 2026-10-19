"""Generates `s2a3_green_area` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_green_area`.

``uc2-greens-theorem-area`` (``multivariable-and-vector-calculus``, answer ``expression``): the area enclosed by a
closed curve, found as the line integral A = (1/2) ∮ (x dy - y dx).

* easy: a tilted ellipse r(t) = (a cos t + b sin t, c cos t + d sin t), or a convex quadrilateral given by its vertices.
* medium: a curve with cusps, written with trigonometric parametrisation: an astroid x = a cos^3 t, y = b sin^3 t, a
  deltoid, a one-cusped epicycloid (cardioid) or a nephroid. The multiple angles have to be expanded.
* hard: a closed curve made of pieces: an arc y = x^n closed by its chord (traversed either way round), a non-convex
  polygon listed clockwise or counterclockwise, or the loop of x = p(t^2 - s^2), y = q t (t^2 - s^2). The area is
  positive whichever way round the curve is traversed.
* exam (``exd_x_s2a3_green_area``): a cusped curve inside a circumscribing ellipse or circle: (a) the enclosed area,
  (b) the area after an enlargement (follows through from (a)), (c) the area between the curve and the circumscribing
  curve (follows through), (d) the fraction of the circumscribing curve's area that the curve encloses (follows through).

Every area is the line integral worked out exactly (Wallis formulas for trigonometric parametrisations) and by
quadrature, and it is compared with a closed form that does not use line integrals: the known area of the family,
the shoelace formula, or an ordinary integral under the graph.
"""

from __future__ import annotations

import math

import sympy as sp

from ..core import attempts, check, generator
from . import _s2a3_vc as vc
from ._s2a3_vc import C, S, t, x, y

F_AREA = [-y / 2, x / 2]  # (1/2)(x dy - y dx) = F . dr


def _signed_area(curves) -> sp.Basic:
    return sum((vc.line(F_AREA, c) for c in curves), sp.Integer(0))


def _tcurve(xc, yc, quarters=(0, 4)) -> vc.Curve:
    return vc.Curve(sp.Matrix([xc, yc]), vc.Dom((t,), (tuple(quarters),), {t: (C, S)}))


def _lin(*terms) -> str:
    """'3cos t - sin t' from [(3, 'cos t'), (-1, 'sin t')]; zero terms are dropped."""
    out = ""
    for coeff, name in terms:
        if coeff == 0:
            continue
        piece = (str(abs(coeff)) if abs(coeff) != 1 else "") + name
        out += ("-" if coeff < 0 else "") + piece if not out else (" - " if coeff < 0 else " + ") + piece
    return out or "0"


def _scaled(a, inner: str) -> str:
    return inner if a == 1 else f"{a}({inner})"


# ---------------------------------------------------------------------------------------------------------------
# polygons


def _shoelace(pts) -> sp.Rational:
    n = len(pts)
    return sp.Rational(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)), 2)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1, p2, p3, p4) -> bool:
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def _simple(pts) -> bool:
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if j in (i, (i + 1) % n) or (j + 1) % n == i:
                continue
            if _segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return False
    return True


def _polygon(rng, sides: int, convex: bool):
    """Lattice vertices in counterclockwise order, round the origin; convex or with a reflex vertex."""
    for _ in attempts(500, "a polygon"):
        angles = sorted(rng.sample(range(0, 360, 15), sides))
        if max(b - a for a, b in zip(angles, angles[1:] + [angles[0] + 360])) >= 170:
            continue
        pts = [
            (round(r * math.cos(math.radians(a))), round(r * math.sin(math.radians(a))))
            for a in angles
            for r in [rng.randint(2, 5)]
        ]
        if len(set(pts)) < sides or not _simple(pts):
            continue
        crosses = [_cross(pts[i - 1], pts[i], pts[(i + 1) % sides]) for i in range(sides)]
        if 0 in crosses or all(c > 0 for c in crosses) != convex:
            continue
        if _shoelace(pts) > 4:
            return pts
    raise AssertionError("unreachable")


def _poly_pieces(pts):
    n = len(pts)
    return [vc.segment(pts[i], pts[(i + 1) % n]) for i in range(n)]


def _pts_text(pts) -> str:
    return ", ".join(f"({a}, {b})" for a, b in pts)


# ---------------------------------------------------------------------------------------------------------------
# tilted ellipse, cusped families, composite curves


def _ellipse(rng):
    for _ in attempts(200, "a tilted ellipse"):
        a, b, c, d = (rng.randint(-3, 3) for _ in range(4))
        det = a * d - b * c
        if det > 0 and (b != 0 or c != 0) and abs(a) + abs(b) > 0 and abs(c) + abs(d) > 0:
            break
    curve = _tcurve(a * C + b * S, c * C + d * S)
    area = sp.pi * det
    text = f"«r(t) = ({_lin((a, 'cos t'), (b, 'sin t'))}, {_lin((c, 'cos t'), (d, 'sin t'))}), 0 ≤ t ≤ 2π»"
    half = _tcurve(a * C + b * S, c * C + d * S, (0, 2))
    return dict(
        curves=[curve], expected=area, text=f"the ellipse {text}", clockwise=False, ccw_area=area, partial=[half]
    )


_FAMILIES = ("astroid", "deltoid", "cardioid", "nephroid")


def _cusped(rng, family=None):
    family = family or rng.choice(_FAMILIES)
    a, b = rng.randint(1, 3), rng.randint(1, 3)
    c2, s2, c3, s3 = C**2 - S**2, 2 * C * S, C**3 - 3 * C * S**2, 3 * C**2 * S - S**3
    if family == "astroid":
        curve = _tcurve(a * C**3, b * S**3)
        area = sp.Rational(3, 8) * sp.pi * a * b
        text = f"«r(t) = ({'' if a == 1 else a}cos³t, {'' if b == 1 else b}sin³t), 0 ≤ t ≤ 2π»"
        name = "astroid-like curve"
    elif family == "deltoid":
        curve = _tcurve(a * (2 * C + c2), a * (2 * S - s2))
        area = 2 * sp.pi * a * a
        text = f"«r(t) = ({_scaled(a, '2cos t + cos 2t')}, {_scaled(a, '2sin t - sin 2t')}), 0 ≤ t ≤ 2π»"
        name = "deltoid"
    elif family == "cardioid":
        curve = _tcurve(a * (2 * C - c2), a * (2 * S - s2))
        area = 6 * sp.pi * a * a
        text = f"«r(t) = ({_scaled(a, '2cos t - cos 2t')}, {_scaled(a, '2sin t - sin 2t')}), 0 ≤ t ≤ 2π»"
        name = "cardioid"
    else:
        curve = _tcurve(a * (3 * C - c3), a * (3 * S - s3))
        area = 12 * sp.pi * a * a
        text = f"«r(t) = ({_scaled(a, '3cos t - cos 3t')}, {_scaled(a, '3sin t - sin 3t')}), 0 ≤ t ≤ 2π»"
        name = "nephroid"
    half = _tcurve(curve.r[0], curve.r[1], (0, 2))
    return dict(
        curves=[curve],
        expected=area,
        text=f"the {name} {text}",
        clockwise=False,
        ccw_area=area,
        partial=[half],
        family=family,
        a=a,
        b=b,
    )


def _arc_chord(rng):
    n, a = rng.choice([2, 3, 4]), rng.randint(1, 3)
    arc = vc.Curve(sp.Matrix([a * vc.t, a**n * vc.t**n]), vc.Dom((vc.t,), ((0, 1),), {}))
    chord = vc.segment((a, a**n), (0, 0))
    out = vc.segment((0, 0), (a, a**n))
    expected = sp.Rational(a ** (n + 1)) * (sp.Rational(1, 2) - sp.Rational(1, n + 1))
    clockwise = rng.random() < 0.4
    p = f"y = x{'²³⁴'[n - 2]}"
    if clockwise:
        reverse = vc.Curve(sp.Matrix([a * (1 - vc.t), a**n * (1 - vc.t) ** n]), vc.Dom((vc.t,), ((0, 1),), {}))
        curves = [out, reverse]
        text = f"the closed curve that goes along the line from «(0, 0)» to «({a}, {a**n})» and returns to the origin along the arc «{p}»"
        partial = [out]
    else:
        curves = [arc, chord]
        text = f"the closed curve that goes along the arc «{p}» from «(0, 0)» to «({a}, {a**n})» and returns to the origin along the straight line"
        partial = [arc]
    return dict(curves=curves, expected=expected, text=text, clockwise=clockwise, ccw_area=expected, partial=partial)


def _loop(rng):
    s, p, q = rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 2)
    xt = p * (vc.t**2 - s * s)
    yt = q * vc.t * (vc.t**2 - s * s)
    curve = vc.Curve(sp.Matrix([xt, yt]), vc.Dom((vc.t,), ((-s, s),), {}))
    expected = sp.Rational(8 * p * q * s**5, 15)
    xs = f"t² - {s * s}" if p == 1 else f"{p}(t² - {s * s})"
    ys = f"{'' if q == 1 else q}t(t² - {s * s})"
    half = vc.Curve(curve.r, vc.Dom((vc.t,), ((0, s),), {}))
    return dict(
        curves=[curve],
        expected=expected,
        clockwise=None,
        ccw_area=expected,
        partial=[half],
        text=f"the loop of the curve «x = {xs}, y = {ys}», for «-{s} ≤ t ≤ {s}»",
    )


def _polygon_item(rng, convex: bool):
    sides = 4 if convex else rng.choice([5, 6])
    pts = _polygon(rng, sides, convex)
    clockwise = (not convex) and rng.random() < 0.5
    if clockwise:
        pts = pts[::-1]
    pieces = _poly_pieces(pts)
    area = abs(_shoelace(pts))
    word = "clockwise" if clockwise else "counterclockwise"
    text = f"the {'quadrilateral' if sides == 4 else ('pentagon' if sides == 5 else 'hexagon')} whose vertices, in order {word}, are «{_pts_text(pts)}»"
    return dict(
        curves=pieces, expected=area, text=text, clockwise=clockwise, ccw_area=area, partial=pieces[:-1], polygon=pts
    )


# ---------------------------------------------------------------------------------------------------------------
# single questions


def _item(rng, tier):
    if tier == "easy":
        return _ellipse(rng) if rng.random() < 0.6 else _polygon_item(rng, True)
    if tier == "medium":
        return _cusped(rng)
    return rng.choice([_arc_chord, _loop, lambda r: _polygon_item(r, False)])(rng)


_STEMS = [
    "Use a line integral to find the area enclosed by {curve}.",
    "Find the area of the region enclosed by {curve}, by evaluating a line integral round its boundary.",
    "Use Green's theorem in the form of a line integral to find the area bounded by {curve}.",
]


def _draw(rng, tier):
    item = _item(rng, tier)
    raw = _signed_area(item["curves"])
    area = abs(raw)
    check(sp.expand(area - item["expected"]) == 0, f"line-integral area {raw} but closed form {item['expected']}")
    if "polygon" in item:
        check(
            sp.expand(area - abs(_shoelace(item["polygon"]))) == 0, "polygon area disagrees with the shoelace formula"
        )
    cands = [("area-signed-clockwise", -area), ("area-forgot-half", 2 * area), ("area-terms-added", sp.Integer(0))]
    open_value = _signed_area(item["partial"])
    cands.append(("area-partial-curve", -open_value if raw < 0 else open_value))
    template = rng.choice(_STEMS).replace("{curve}", item["text"])
    text = vc.stem(template)
    working = [
        "A = (1/2) times the line integral of x dy - y dx round the boundary, taken counterclockwise "
        "(a clockwise traversal gives minus the area).",
    ]
    return dict(**text, answer=area, candidates=cands, working=working)


@generator(
    "s2a3_green_area",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a3_green_area(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return vc.single_question(rng, _draw, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "A garden bed has as its boundary {curve}. Lengths are in metres.",
    "A landscape designer draws the boundary of a garden bed as {curve}, with lengths in metres.",
]
_A = [
    "Use a line integral round the boundary to find the area of the garden bed, in square metres.",
    "Use a line integral to find the area of the bed, in square metres.",
]
_B = [
    "The design is enlarged by a factor of «{k}» in every direction. Find the area of the enlarged bed, in square metres.",
    "A larger copy is made with every length multiplied by «{k}». Find its area, in square metres.",
]
_C = [
    "The bed lies inside {outer}. Find the area of the ground that is inside that curve but outside the bed, in square metres.",
    "The bed lies inside {outer}. Find the area between the boundary of the bed and that curve, in square metres.",
]
_D_FRACTION = [
    "The bed lies inside {outer}. What fraction of the area inside that curve does the bed take up?",
    "The bed lies inside {outer}. Express the area of the bed as a fraction of the area inside that curve.",
]
_D_MULCH = [
    "Mulch is spread over the original bed at «{k}» kilograms per square metre. Find the mass of mulch needed, in kilograms.",
    "The original bed is covered with mulch at «{k}» kg per square metre. Find the mass of mulch in kilograms.",
]


def _exam_once(rng):
    family = rng.choice(["astroid", "deltoid"])
    item = _cusped(rng, family)
    a, b = item["a"], item["b"]
    area = abs(_signed_area(item["curves"]))
    check(sp.expand(area - item["expected"]) == 0, "exam area disagrees with its closed form")
    if family == "astroid":
        outer_area = sp.pi * a * b
        xa = "x" if a == 1 else f"(x/{a})"
        ya = "y" if b == 1 else f"(y/{b})"
        outer = f"the ellipse «{xa}² + {ya}² = 1» through the four cusps"
        fraction = sp.Rational(3, 8)
    else:
        outer_area = 9 * sp.pi * a * a
        outer = f"the circle «x² + y² = {9 * a * a}» through the three cusps"
        fraction = sp.Rational(2, 9)
    k = rng.choice([2, 3])
    between = outer_area - area
    check(sp.expand(area / outer_area - fraction) == 0, "fraction of the circumscribing curve disagrees")
    if rng.random() < 0.5:
        part_d = vc.part_dict(
            vc.prompt(rng.choice(_D_FRACTION).replace("{outer}", outer)),
            fraction,
            [
                ("area-fraction-inverted", 1 / fraction),
                ("area-fraction-complement", 1 - fraction),
                ("area-fraction-of-complement", fraction / (1 - fraction)),
            ],
            2,
            ["Divide the area of the bed by the area inside the outer curve."],
            follow_through=f"part_a / ({sp.sstr(outer_area)})",
        )
    else:
        m = rng.choice([2, 3, 4, 5])
        part_d = vc.part_dict(
            vc.prompt(rng.choice(_D_MULCH), k=m),
            m * area,
            [
                ("area-mulch-on-enlarged", m * k * k * area),
                ("area-mulch-on-between", m * between),
                ("area-forgot-half", 2 * m * area),
            ],
            2,
            ["Mass = rate times area of the original bed, from (a)."],
            follow_through=f"{m}*part_a",
        )
    parts = [
        vc.part_dict(
            vc.prompt(rng.choice(_A)),
            area,
            [
                ("area-forgot-half", 2 * area),
                ("area-terms-added", sp.Integer(0)),
                ("area-signed-clockwise", -area),
                ("area-partial-curve", abs(_signed_area(item["partial"]))),
            ],
            3,
            [
                "A = (1/2) times the line integral of x dy - y dx; expand the multiple angles and integrate over 0 ≤ t ≤ 2π."
            ],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_B), k=k),
            k * k * area,
            [("area-scale-linear", k * area), ("area-scale-cubed", k**3 * area), ("area-scale-unchanged", area)],
            2,
            ["Areas scale with the square of the enlargement factor."],
            follow_through=f"{k * k}*part_a",
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_C).replace("{outer}", outer)),
            between,
            [("area-between-reversed", area - outer_area), ("area-between-outer-only", outer_area)],
            2,
            ["Subtract the area of the bed, from (a), from the area inside the outer curve."],
            follow_through=f"{sp.sstr(outer_area)} - part_a",
        ),
        part_d,
    ]
    template = rng.choice(_EXAM_STEMS).replace("{curve}", item["text"])
    return dict(**vc.stem(template), parts=parts)


@generator("exd_x_s2a3_green_area", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_green_area(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
