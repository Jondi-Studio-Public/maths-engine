"""Generates `s2a1_region_integral` questions (easy, medium, hard) and the exam chain `exd_x_s2a1_region_integral`.

``uc2-double-integral-general-region`` (``multivariable-and-vector-calculus``, answer ``fraction``): double integrals
over regions bounded by curves, where the learner finds the intersection points, chooses between vertical and
horizontal slices, and writes limits that depend on the outer variable. (``ucv1-double-integral-rectangle`` does the
rectangle.)

* easy: a region between a parabola and a line, or two curves meeting at the origin, with a linear integrand, or a
  triangle cut off by the axes.
* medium: a parabola and a line that meet at points on both sides of the axis, written with x as a function of y
  (the learner may have to choose the slicing), the same region asked as a mass, a volume under a surface, or an
  average value.
* hard: a triangle with no side parallel to an axis, whose slices change at the middle vertex; the lens between
  y = x² and x = y²; the region between two parabolas; each with a non-separable integrand and asked as an integral,
  a mass, a volume or an average value.
* exam (``exd_x_s2a1_region_integral``): a thin plate between a parabola and a line with density ρ = k + m·w
  (w = x or y): (a) the area, (b) the mass (follows through from (a)), (c) one coordinate of the centre of mass
  (follows through from (b)), (d) the average density (follows through from (a) and (b)).

Each key is worked out twice. The exact route integrates the limits term by term. The second route lives in
tests/test_s2a1_region.py: a midpoint-rule sum on a fine grid over the points that satisfy the region's inequalities, which knows nothing
about the limits.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts, check, generator, rational_text
from ._s2_options import keep_answer_shape
from ._s2a1_common import close, exam_part, grid_integral, stem, wrong_answers

x, y = sympy.symbols("x y", real=True)
S = sympy.Integer
R = sympy.Rational


def integrate_terms(expr, var, lo, hi):
    """Exact integral of a sum of c·var^p terms (p rational) between `lo` and `hi`."""
    total = S(0)
    for term in sympy.Add.make_args(sympy.expand(expr)):
        coeff, dep = term.as_independent(var, as_Add=False)
        if dep == 1:
            total += coeff * (hi - lo)
        else:
            base, power = dep.as_base_exp()
            check(base == var, "a term is not a power of the variable")
            total += coeff * (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)
    return sympy.expand(total)


def iterated(f, pieces, order: str = "I", swap_limits: bool = False):
    """∬ f over a region given as pieces (a, b, lo, hi): order "I" has x outer (lo, hi in x) and y inner,
    order "II" has y outer (lo, hi in y) and x inner."""
    total = S(0)
    outer, inner = (x, y) if order == "I" else (y, x)
    for a, b, lo, hi in pieces:
        if swap_limits:
            lo, hi = hi, lo
        total += integrate_terms(integrate_terms(f, inner, lo, hi), outer, a, b)
    total = sympy.nsimplify(total) if total.has(sympy.Float) else sympy.simplify(total)
    check(total.is_Rational, f"the integral is not rational: {total}")
    return total


def _line(m, c, var=x) -> str:
    """'m·var + c' as text: 2x + 3, -x, x - 4, 5."""
    parts = []
    if m:
        parts.append(f"{'-' if m == -1 else '' if m == 1 else m}{var}")
    if c or not parts:
        parts.append(str(c) if not parts else f"{'+' if c > 0 else '-'} {abs(c)}")
    return " ".join(parts)


def _poly_text(expr) -> str:
    t = str(sympy.expand(expr)).replace("**2", "²").replace("**3", "³").replace("*", "")
    return t


def _f_text(f) -> str:
    text = _poly_text(f)
    return f"({text})" if len(sympy.Add.make_args(sympy.expand(f))) > 1 else text


class Shape:
    """A region: its words, slicing pieces in both orders where they exist, a membership test and a bounding box."""

    def __init__(self, words, pieces, order, inside, box, extra=None):
        self.words, self.pieces, self.order, self.inside, self.box = words, pieces, order, inside, box
        self.extra = extra or {}


def _curve_line(rng, orient: str, hard: bool = False) -> Shape:
    for _ in attempts(100, "a parabola and a line"):
        c = rng.choice([1, 1, 2, 3]) if not hard else rng.choice([1, 2])
        x1, x2 = sorted(rng.sample(range(-3, 4), 2))
        if x2 - x1 > 4 or x1 == x2:
            continue
        if not hard and rng.random() < 0.4:
            x1 = 0
            if x2 <= 0:
                continue
        m, b = c * (x1 + x2), -c * x1 * x2
        if m == 0 and b == 0:
            continue
        break
    u = x if orient == "x" else y
    v = y if orient == "x" else x
    curve_words = f"{v} = {'' if c == 1 else c}{u}²"
    line_words = f"{v} = {_line(m, b, u)}"
    pieces = [(S(x1), S(x2), c * u**2, m * u + b)]
    if orient == "x":
        inside = lambda gx, gy: (gy >= c * gx**2) & (gy <= m * gx + b)  # noqa: E731
        ymax = max(m * x1 + b, m * x2 + b, 0)
        # the line is above the parabola between the roots; the box covers its highest point
        box = (min(x1, x2) - 0.1, max(x1, x2) + 0.1, -0.1, max(c * x1**2, c * x2**2, (m * x1 + b), (m * x2 + b)) + 0.1)
    else:
        inside = lambda gx, gy: (gx >= c * gy**2) & (gx <= m * gy + b)  # noqa: E731
        box = (-0.1, max(c * x1**2, c * x2**2, (m * x1 + b), (m * x2 + b)) + 0.1, min(x1, x2) - 0.1, max(x1, x2) + 0.1)
    return Shape(
        f"bounded by the parabola «{curve_words}» and the line «{line_words}»",
        pieces,
        "I" if orient == "x" else "II",
        inside,
        box,
        extra=dict(c=c, x1=x1, x2=x2, m=m, b=b, orient=orient),
    )


def _axes_triangle(rng) -> Shape:
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    # the triangle with vertices (0, 0), (a, 0), (0, b): bounded by the axes and the line x/a + y/b = 1
    big = math.lcm(a, b)
    words = f"bounded by the coordinate axes and the line «{'' if big == a else big // a}x + {'' if big == b else big // b}y = {big}»"
    pieces = [(S(0), S(a), S(0), S(b) * (1 - x / a))]
    inside = lambda gx, gy: (gx >= 0) & (gy >= 0) & (gx / a + gy / b <= 1)  # noqa: E731
    return Shape(words, pieces, "I", inside, (-0.1, a + 0.1, -0.1, b + 0.1), extra=dict(a=a, b=b, kind="axes"))


def _two_curves_origin(rng) -> Shape:
    a = rng.randint(1, 3)
    k = rng.randint(1, 2)
    # y = k x² and y = k a x meet at 0 and a
    words = f"bounded by the curves «y = {'' if k == 1 else k}x²» and «y = {_line(k * a, 0)}»"
    pieces = [(S(0), S(a), k * x**2, k * a * x)]
    inside = lambda gx, gy: (gy >= k * gx**2) & (gy <= k * a * gx)  # noqa: E731
    return Shape(words, pieces, "I", inside, (-0.1, a + 0.1, -0.1, k * a * a + 0.1), extra=dict(kind="origin"))


def _triangle(rng) -> Shape:
    """A triangle with no side parallel to an axis: vertical slices change at the middle vertex."""
    for _ in attempts(500, "a triangle with distinct coordinates"):
        pts = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(3)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if len(set(xs)) < 3 or len(set(ys)) < 3:
            continue
        a, b, c = sorted(pts)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area2 < 6 or area2 > 20:
            continue
        break

    def line_through(p, q):
        slope = Fraction(q[1] - p[1], q[0] - p[0])
        return sympy.Rational(slope.numerator, slope.denominator) * (x - p[0]) + p[1]

    ac, ab, bc = line_through(a, c), line_through(a, b), line_through(b, c)
    b_above = ab.subs(x, b[0]) > ac.subs(x, b[0])
    if b_above:
        pieces = [(S(a[0]), S(b[0]), ac, ab), (S(b[0]), S(c[0]), ac, bc)]
    else:
        pieces = [(S(a[0]), S(b[0]), ab, ac), (S(b[0]), S(c[0]), bc, ac)]

    def inside(gx, gy):
        s1 = (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0])
        s2 = (c[0] - b[0]) * (gy - b[1]) - (c[1] - b[1]) * (gx - b[0])
        s3 = (a[0] - c[0]) * (gy - c[1]) - (a[1] - c[1]) * (gx - c[0])
        return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))

    words = (
        f"whose boundary is the triangle with vertices «({a[0]}, {a[1]})», «({b[0]}, {b[1]})» and «({c[0]}, {c[1]})»"
    )
    return Shape(
        words,
        pieces,
        "I",
        inside,
        (min(xs) - 0.1, max(xs) + 0.1, min(ys) - 0.1, max(ys) + 0.1),
        extra=dict(kind="triangle", vertices=(a, b, c)),
    )


def _lens(rng) -> Shape:
    k = rng.choice([1, 1, 2])
    # y = k x² and x = k y²? the region needs equal curvature; use y = x² and x = y² scaled by choosing k = 1 only
    words = "bounded by the curves «y = x²» and «x = y²»"
    pieces = [(S(0), S(1), x**2, sympy.sqrt(x))]
    inside = lambda gx, gy: (gy >= gx**2) & (gx >= gy**2)  # noqa: E731
    return Shape(words, pieces, "I", inside, (-0.1, 1.1, -0.1, 1.1), extra=dict(kind="lens"))


def _two_parabolas(rng) -> Shape:
    a, c, e = rng.randint(1, 2), rng.choice([1, 2]), rng.choice([1, 2])
    d = (c + e) * a * a
    words = f"bounded by the parabolas «y = {'' if c == 1 else c}x²» and «y = {d} - {'' if e == 1 else e}x²»"
    pieces = [(S(-a), S(a), c * x**2, d - e * x**2)]
    inside = lambda gx, gy: (gy >= c * gx**2) & (gy <= d - e * gx**2)  # noqa: E731
    return Shape(words, pieces, "I", inside, (-a - 0.1, a + 0.1, -0.1, d + 0.1), extra=dict(kind="parabolas"))


# ---------------------------------------------------------------------------------------------------------------
# questions


_INTEGRANDS_ANY = [
    x,
    y,
    x * y,
    x + y,
    x**2,
    y**2,
    x + 2 * y,
    2 * x + 3 * y,
    x * y**2,
    x**2 * y,
    x**2 + y,
    x * y + x,
    3 * x - y,
]
_INTEGRANDS_POS = [y, 1 + y, x**2 + y, y**2, 2 * y + 3, x**2 + y**2, x**2 + 2 * y, 1 + x**2, y + x**2 + 1]


def _wrong_routes(f, shape: Shape, value):
    """Tagged wrong values from the usual slips for this region."""
    pieces = shape.pieces
    order = shape.order
    cands = []
    cands.append(("region-limits-swapped", -value))
    if len(pieces) == 1:
        a, b, lo, hi = pieces[0]
        outer, inner = (x, y) if order == "I" else (y, x)
        # the inner limits kept constant: the bounding rectangle
        lo_vals = [lo.subs(outer, a), lo.subs(outer, b)] if lo.has(outer) else [lo]
        hi_vals = [hi.subs(outer, a), hi.subs(outer, b)] if hi.has(outer) else [hi]
        rect = (min(lo_vals + hi_vals), max(lo_vals + hi_vals))
        cands.append(("region-bounding-rectangle", iterated(f, [(a, b, rect[0], rect[1])], order)))
        cands.append(("region-lower-curve-dropped", iterated(f, [(a, b, S(0), hi)], order)))
        # limits written in the wrong variable
        swapped_lo, swapped_hi = lo.subs(outer, inner), hi.subs(outer, inner)
        other = "II" if order == "I" else "I"
        cands.append(("region-limit-variable-confused", iterated(f, [(a, b, swapped_lo, swapped_hi)], other)))
    else:
        (a0, b0, lo0, hi0), (a1, b1, lo1, hi1) = pieces
        cands.append(("region-second-slice-dropped", iterated(f, [pieces[0]], order)))
        cands.append(("region-slice-change-ignored", iterated(f, [(a0, b1, lo0, hi0)], order)))
        cands.append(("region-slice-change-ignored", iterated(f, [(a0, b1, lo1, hi1)], order)))
        cands.append(("region-limits-swapped-outer", iterated(f, [(a0, b0, lo0, hi0), (a1, b1, hi1, lo1)], order)))
    return cands


def _nonneg(cands):
    return [(tag, v) for tag, v in cands if sympy.sympify(v) >= 0]


def _stem_for(rng, kind: str, f, shape: Shape) -> str:
    words = shape.words
    if kind == "integral":
        return f"Evaluate «∬ {_f_text(f)} dA», where R is the region {words}."
    if kind == "mass":
        return (
            f"A thin plate occupies the region R {words}, with lengths in metres. Its density at the point «(x, y)» is "
            f"«ρ = {_poly_text(f)}» (in kilograms per square metre). Find the mass of the plate in kilograms."
        )
    if kind == "volume":
        return (
            f"Find the volume of the solid that lies above the region R in the xy-plane {words} "
            f"and below the surface «z = {_poly_text(f)}»."
        )
    return f"Find the average value of «f(x, y) = {_poly_text(f)}» over the region R {words}."


def _build(rng, shape: Shape, kinds):
    kind = rng.choice(kinds)
    for _ in attempts(60, "an integrand with a non-zero integral"):
        f = rng.choice(_INTEGRANDS_ANY if kind == "integral" else _INTEGRANDS_POS)
        if kind != "integral" and not _positive_on(f, shape):
            continue
        base = iterated(f, shape.pieces, shape.order)
        if base != 0:
            break
    area = iterated(S(1), shape.pieces, shape.order)
    value = base / area if kind == "average" else base
    cands = _wrong_routes(f, shape, value)
    if kind == "average":
        cands = [("region-average-not-divided", base)] + [(t, c / area) for t, c in cands]
    return _stem_for(rng, kind, f, shape), value, cands


def _positive_on(f, shape: Shape) -> bool:
    """f is non-negative on the region (checked on a grid of its points)."""
    fx = sympy.lambdify((x, y), f, "numpy")
    gx = np.linspace(shape.box[0], shape.box[1], 80)
    gy = np.linspace(shape.box[2], shape.box[3], 80)
    mx, my = np.meshgrid(gx, gy, indexing="ij")
    mask = shape.inside(mx, my)
    return bool(np.all(fx(mx, my)[mask] >= -1e-12)) if mask.any() else False


def _easy(rng):
    pick = rng.random()
    if pick < 0.45:
        shape = _two_curves_origin(rng)
    elif pick < 0.8:
        shape = _curve_line(rng, "x")
        if shape.extra["x1"] != 0:
            shape = _two_curves_origin(rng)
    else:
        shape = _axes_triangle(rng)
    return _build(rng, shape, ["integral", "integral", "integral", "volume"])


def _medium(rng):
    shape = _curve_line(rng, rng.choice(["x", "y", "y"]))
    return _build(rng, shape, ["integral", "mass", "volume", "average"])


def _hard(rng):
    pick = rng.random()
    if pick < 0.5:
        shape = _triangle(rng)
    elif pick < 0.7:
        shape = _lens(rng)
    elif pick < 0.85:
        shape = _two_parabolas(rng)
    else:
        shape = _curve_line(rng, rng.choice(["x", "y"]), hard=True)
    return _build(rng, shape, ["integral", "mass", "volume", "average"])


@generator(
    "s2a1_region_integral",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@keep_answer_shape
def s2a1_region_integral(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(60, "a region integral with enough distractors"):
        try:
            text, value, cands = build(rng)
        except VerificationError:
            continue
        value = sympy.Rational(value)
        signed = text.startswith("Evaluate")
        if not signed:
            cands = _nonneg(cands)  # a negative area, volume, mass or average is a giveaway
        wrong = wrong_answers(value, cands, 4, kind="fraction", allow_negative=signed)
        if len(wrong) >= 3 and value.denominator <= 400:
            break
    f = Fraction(int(value.p), int(value.q))
    return dict(
        **stem(text),
        answer=exact_value(f),
        answer_text=rational_text(f),
        answer_kind="fraction",
        distractors=wrong,
        working=[
            "Find where the boundary curves meet, choose the order of integration, and integrate the limits in turn."
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    orient = rng.choice(["x", "y"])
    shape = _curve_line(rng, orient, hard=False)
    w, u = rng.choice([x, y]), rng.choice([x, y])
    k, m = rng.randint(1, 4), rng.randint(1, 3)
    rho = k + m * w
    order, pieces = shape.order, shape.pieces
    A = iterated(S(1), pieces, order)
    Iw = iterated(w, pieces, order)
    M = iterated(rho, pieces, order)
    Mu = iterated(u * rho, pieces, order)
    check(sympy.simplify(M - k * A - m * Iw) == 0, "mass is not kA + m times the first moment")
    ubar, avg = Mu / M, M / A
    check(ubar != 0 and avg > 0 and A > 0, "degenerate plate")
    shape_ext = shape.extra
    uname = "x" if u == x else "y"
    wname = "x" if w == x else "y"
    # extents of the plate along each axis
    box = shape.box
    lo, hi = (box[0] + 0.1, box[1] - 0.1) if u == x else (box[2] + 0.1, box[3] - 0.1)
    lo, hi = (sympy.nsimplify(round(float(t), 6), rational=True) for t in (lo, hi))
    check(lo < float(ubar) < hi, "centre of mass outside the plate's extent")
    unit_cent = Iu_over_A = iterated(u, pieces, order) / A

    def cand_table():
        a_c = [
            ("region-limits-swapped", -A),
            (
                "region-bounding-rectangle",
                iterated(S(1), [(pieces[0][0], pieces[0][1], *_rect(pieces[0], order))], order),
            ),
            ("region-lower-curve-dropped", iterated(S(1), [(pieces[0][0], pieces[0][1], S(0), pieces[0][3])], order)),
            (
                "region-limit-variable-confused",
                iterated(S(1), [_confused(pieces[0], order)], "II" if order == "I" else "I"),
            ),
        ]
        b_c = [
            ("triple-integral-density-term-dropped", k * A),
            ("triple-integral-density-maximum", (k + m * max(abs(hi), abs(lo))) * A),
            ("triple-integral-density-midpoint", (k + m * (lo + hi) / 2) * A),
            ("region-limits-swapped", -M),
        ]
        c_c = [
            ("triple-integral-moment-not-divided", Mu),
            ("triple-integral-centroid-uniform", unit_cent),
            ("triple-integral-moment-inverted", M / Mu),
            ("triple-integral-wrong-coordinate", iterated((y if u == x else x) * rho, pieces, order) / M),
        ]
        d_c = [
            ("triple-integral-moment-inverted", A / M),
            ("triple-integral-mass-not-divided", M),
            ("triple-integral-density-midpoint", k + m * (lo + hi) / 2),
            ("triple-integral-density-maximum", k + m * max(abs(hi), abs(lo))),
        ]
        return a_c, b_c, c_c, d_c

    a_c, b_c, c_c, d_c = (_nonneg(c) for c in cand_table())
    Iw_text = rational_text(Fraction(int(Iw.p), int(Iw.q)))
    Mu_text = rational_text(Fraction(int(Mu.p), int(Mu.q)))
    parts = [
        exam_part(
            "Find the area of the plate, in square metres.",
            A,
            3,
            ["The area is the double integral of 1 over the plate."],
            a_c,
            sensible_range=(0, None),
            allow_negative=False,
        ),
        exam_part(
            "Hence find the mass of the plate, in kilograms.",
            M,
            3,
            [f"M = {k}A + {m}∬ {wname} dA."],
            b_c,
            follow_through=f"{k}*part_a + {m}*({Iw_text})",
            sensible_range=(0, None),
            allow_negative=False,
        ),
        exam_part(
            f"Find the «{uname}»-coordinate of the centre of mass of the plate, in metres.",
            ubar,
            3,
            [f"{uname}-bar = (∬ {uname}ρ dA)/M."],
            c_c,
            follow_through=f"({Mu_text})/part_b",
            sensible_range=(float(min(lo, hi)), float(max(lo, hi))),
            allow_negative=False,
        ),
        exam_part(
            "Find the average density of the plate, in kilograms per square metre.",
            avg,
            1,
            ["Average density = mass / area."],
            d_c,
            follow_through="part_b/part_a",
            sensible_range=(0, None),
            allow_negative=False,
        ),
    ]
    check(all(parts), "a part has too few distractors")
    return dict(
        **stem(
            f"A thin metal plate occupies the region {shape.words}, with lengths in metres. At the point «(x, y)» its "
            f"density is «ρ = {k} + {m if m > 1 else ''}{wname}» (in kilograms per square metre)."
        ),
        parts=parts,
    )


def _rect(piece, order):
    a, b, lo, hi = piece
    outer = x if order == "I" else y
    values = [e.subs(outer, a) if getattr(e, "has", None) and e.has(outer) else e for e in (lo, hi)]
    values += [e.subs(outer, b) if getattr(e, "has", None) and e.has(outer) else e for e in (lo, hi)]
    return min(values), max(values)


def _confused(piece, order):
    a, b, lo, hi = piece
    outer, inner = (x, y) if order == "I" else (y, x)
    return a, b, lo.subs(outer, inner), hi.subs(outer, inner)


@generator("exd_x_s2a1_region_integral", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a1_region_integral(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam region chain"):
        try:
            return _exam_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
