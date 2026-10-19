"""Generates `f10c_quadrilateral_properties` questions for `f10-quadrilateral-properties`: a property of a
quadrilateral that follows from congruent triangles (opposite sides and angles of a parallelogram, diagonals that
bisect each other, a rhombus's perpendicular diagonals, equal rectangle diagonals, the axis of a kite) is stated in
the question, and the learner applies it to find one unknown length or angle in a named quadrilateral.

Nothing needs a diagram: the property and every number are in the words, and the answer follows from them alone.
Every answer is a number of cm or degrees, always asked to 1 decimal place; because lengths and angles are
integers and several steps halve them, the answers are whole numbers or end in .5.

Tiers escalate by the amount of reasoning, not by bigger numbers:

- **easy** (one stated property, one step): halve or double a diagonal that is bisected; the side of a rhombus from
  its perimeter, or its perimeter from a side; the other side of a rectangle from its perimeter; the perimeter of a
  kite from its two pairs of equal sides.
- **medium** (the property plus an angle fact): the angle next to a given one in a parallelogram (angles on one
  side add to 180 degrees); a base angle of an isosceles triangle made by a rhombus or rectangle diagonal; the
  right angle where a rhombus's diagonals cross; the remaining angle of a kite; the perimeter of the triangle cut
  off by the diagonals.
- **hard** (a combination): a side or angle that is given as an algebraic expression and has to be solved before
  the property is applied; a rhombus side from its diagonals (right angle, then Pythagoras); the distance from the
  centre of a rectangle to a corner (Pythagoras, then equal bisected diagonals); the axis of a kite (right angle,
  Pythagoras); and an angle between a rectangle's diagonals found from an angle at a corner.

Verification: every answer is also computed by a different route from the construction. Where the figure can be
drawn, a coordinate model is built from the givens (not from the answer) and the quantity is measured from the
coordinates; algebraic cases are re-solved by trying every whole value of x. Each distractor is what one named
error produces, is checked against the answer with the grader's own tolerance, and is kept only when it is a
possible length or angle for the figure.
"""

from __future__ import annotations

import math
import re

from ..core import GRADER_TOLERANCE, check, generator, phrase, pick_distractors

TIERS = ("easy", "medium", "hard")

# Note ids (they must match the `misconceptions` of the skill file).
M_DIAG = "diagonal-not-halved-or-doubled"
M_OTHER_DIAG = "uses-the-other-diagonal"
M_HALVE_TWICE = "halves-twice"
M_PERIM = "misreads-the-perimeter"
M_CONSEC_EQ = "consecutive-angles-equal"
M_TOTAL_360 = "uses-360-for-two-angles"
M_HALVE_ANGLE = "halves-the-given-angle"
M_NO_HALF = "omits-the-final-halving"
M_COPY = "copies-the-given-angle"
M_RIGHT_ANGLE = "uses-the-wrong-triangle-total"
M_PAIR_ONCE = "counts-a-pair-once"
M_KITE = "misplaces-the-equal-kite-angles"
M_OTHER_SIDE = "uses-the-angle-at-the-other-side"
M_WHOLE_DIAG = "uses-whole-diagonals"
M_OMITS = "leaves-out-a-length"
M_XVALUE = "reports-x-not-the-measurement"
M_ADD_OPP = "adds-the-opposite-sides"
M_SLIP = "transposition-slip"
M_PYTH_ADD = "adds-instead-of-using-pythagoras"
M_PYTH_SUB = "subtracts-instead-of-using-pythagoras"
M_AVG = "averages-the-sides"
M_SUPP = "names-the-supplementary-angle"
M_ONE_ANGLE = "counts-one-angle-not-both"

_NAME_SETS = ["ABCD", "PQRS", "WXYZ", "EFGH"]
_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15), (7, 24, 25), (20, 21, 29), (12, 16, 20)]


# -- Independent geometry (coordinates) -------------------------------------


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _angle_at(p, v, q):
    """The angle PVQ in degrees, from the dot product."""
    a = (p[0] - v[0], p[1] - v[1])
    b = (q[0] - v[0], q[1] - v[1])
    c = (a[0] * b[0] + a[1] * b[1]) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _cross(p1, p2, p3, p4):
    """Intersection of line p1p2 with line p3p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def _parallelogram_from_diagonals(d, e, phi=1.0):
    """A, B, C, D of a parallelogram whose diagonals AC and BD have lengths d and e (angle phi between them)."""
    ac = (d, 0.0)
    bd = (e * math.cos(phi), e * math.sin(phi))
    a = (0.0, 0.0)
    q = ((ac[0] + bd[0]) / 2, (ac[1] + bd[1]) / 2)  # D - A
    p = ((ac[0] - bd[0]) / 2, (ac[1] - bd[1]) / 2)  # B - A
    return a, (p[0], p[1]), ac, q


def _parallelogram_from_sides(s, t, theta_deg):
    """A, B, C, D with AB = s, AD = t and angle DAB = theta."""
    th = math.radians(theta_deg)
    a = (0.0, 0.0)
    b = (s, 0.0)
    d = (t * math.cos(th), t * math.sin(th))
    c = (b[0] + d[0], b[1] + d[1])
    return a, b, c, d


def _rhombus_from_side_angle(s, theta_deg):
    return _parallelogram_from_sides(s, s, theta_deg)


# -- Case plumbing ------------------------------------------------------------


class Case:
    def __init__(self, text, value, unit, cands, working, truth, lo=0, hi=400):
        self.text = text  # the stem, without the closing precision sentence
        self.value = value  # the keyed answer (multiple of 0.5)
        self.unit = unit  # "cm" or "degrees"
        self.cands = cands  # [(value, note id)] in priority order
        self.working = working
        self.truth = truth  # the same quantity, found by a different route
        self.lo = lo  # a distractor must lie strictly between lo and hi
        self.hi = hi


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def _is_half(v: float) -> bool:
    return abs(v * 2 - round(v * 2)) < 1e-9


def _names(rng):
    nm = rng.choice(_NAME_SETS)
    return nm, nm[0], nm[1], nm[2], nm[3]


# -- Easy ----------------------------------------------------------------------


def _e_diag_half(rng):
    nm, a, b, c, d = _names(rng)
    ac, bd = rng.randint(10, 40), rng.randint(10, 40)
    if ac == bd:
        bd += 2
    text = phrase(
        rng,
        [
            f"The diagonals of a parallelogram bisect each other. In parallelogram {nm}, the diagonals {a}{c} and {b}{d} "
            f"meet at M. {a}{c} = {ac} cm and {b}{d} = {bd} cm. Find the length {a}M in cm.",
            f"In parallelogram {nm} the diagonals {a}{c} and {b}{d} cross at M, where {a}{c} = {ac} cm and {b}{d} = {bd} cm. "
            f"Use the fact that the diagonals of a parallelogram bisect each other to find {a}M in cm.",
        ],
    )
    pa, pb, pc, pd = _parallelogram_from_diagonals(ac, bd)
    m = _cross(pa, pc, pb, pd)
    return Case(
        text,
        ac / 2,
        "cm",
        [(ac, M_DIAG), (bd / 2, M_OTHER_DIAG), (ac / 4, M_HALVE_TWICE), (bd, M_OTHER_DIAG)],
        [f"The diagonals bisect each other, so M is the midpoint of {a}{c}.", f"{a}M = {ac} / 2 = {_fmt(ac / 2)} cm."],
        _dist(pa, m),
    )


def _e_diag_double(rng):
    nm, a, b, c, d = _names(rng)
    am, bm = rng.randint(5, 20), rng.randint(5, 20)
    if am == bm:
        bm += 1
    text = phrase(
        rng,
        [
            f"The diagonals of a parallelogram bisect each other. In parallelogram {nm}, the diagonals meet at M, "
            f"with {a}M = {am} cm and {b}M = {bm} cm. Find the length of the diagonal {b}{d} in cm.",
            f"Parallelogram {nm} has diagonals {a}{c} and {b}{d} meeting at M. Because the diagonals bisect each other, "
            f"M is the midpoint of each. Given {a}M = {am} cm and {b}M = {bm} cm, find {b}{d} in cm.",
        ],
    )
    pa, pb, pc, pd = _parallelogram_from_diagonals(2 * am, 2 * bm)
    m = _cross(pa, pc, pb, pd)
    check(
        abs(_dist(pa, m) - am) < 1e-6 and abs(_dist(pb, m) - bm) < 1e-6,
        "f10c_quadrilateral_properties: model misses the given half-diagonals",
    )
    return Case(
        text,
        2 * bm,
        "cm",
        [(bm, M_DIAG), (2 * am, M_OTHER_DIAG), (am + bm, M_OMITS), (4 * bm, M_HALVE_TWICE)],
        [f"M is the midpoint of {b}{d}, so {b}{d} = 2 x {b}M.", f"{b}{d} = 2 x {bm} = {_fmt(2 * bm)} cm."],
        _dist(pb, pd),
    )


def _perimeter_of_rhombus(side):
    pa, pb, pc, pd = _rhombus_from_side_angle(float(side), 70)
    return _dist(pa, pb) + _dist(pb, pc) + _dist(pc, pd) + _dist(pd, pa)


def _e_rhombus_perim(rng):
    nm, a, b, c, d = _names(rng)
    s = rng.randint(4, 25)
    if rng.random() < 0.5:
        text = (
            f"All four sides of a rhombus are equal. Rhombus {nm} has a perimeter of {4 * s} cm. "
            f"Find the length of side {a}{b} in cm."
        )
        value, cands = s, [(2 * s, M_PERIM), (4 * s / 3, M_PERIM), (4 * s, M_PERIM), (s / 2, M_PERIM)]
        wk = [f"All four sides are equal, so each is {4 * s} / 4.", f"{a}{b} = {_fmt(s)} cm."]
    else:
        text = (
            f"All four sides of a rhombus are equal. In rhombus {nm}, the side {a}{b} is {s} cm long. "
            f"Find the perimeter of the rhombus in cm."
        )
        value, cands = 4 * s, [(2 * s, M_PERIM), (3 * s, M_PERIM), (s * s, M_PERIM), (s, M_PERIM)]
        wk = ["All four sides are equal to " + str(s) + " cm.", f"Perimeter = 4 x {s} = {_fmt(4 * s)} cm."]
    pa, pb, pc, pd = _rhombus_from_side_angle(float(s), 70)
    perimeter = _dist(pa, pb) + _dist(pb, pc) + _dist(pc, pd) + _dist(pd, pa)
    if value == s:  # asked for the side: find the s whose rhombus has the stated perimeter
        truth = next(
            t for t in range(1, 200) if abs(4 * t - 4 * s) < 1e-9 and abs(_perimeter_of_rhombus(t) - 4 * s) < 1e-9
        )
    else:
        truth = perimeter
    return Case(text, value, "cm", cands, wk, truth)


def _rect_perimeter(length, width):
    pts = [(0.0, 0.0), (float(length), 0.0), (float(length), float(width)), (0.0, float(width))]
    return sum(_dist(pts[i], pts[(i + 1) % 4]) for i in range(4))


def _e_rect_perim(rng):
    nm, a, b, c, d = _names(rng)
    length, width = rng.randint(6, 25), rng.randint(3, 15)
    if length == width:
        length += 2
    p = 2 * (length + width)
    text = phrase(
        rng,
        [
            f"Opposite sides of a rectangle are equal. Rectangle {nm} has a perimeter of {p} cm and {a}{b} = {length} cm. "
            f"Find the length of {b}{c} in cm.",
            f"In rectangle {nm}, {a}{b} = {length} cm and the perimeter is {p} cm. Opposite sides of a rectangle are equal, "
            f"so find {b}{c} in cm.",
        ],
    )
    return Case(
        text,
        width,
        "cm",
        [(p - length, M_PERIM), (p / 2, M_PERIM), (p / 4, M_PERIM), (p / 2 - 2 * length, M_PERIM)],
        [
            f"{a}{b} + {b}{c} is half the perimeter: {p} / 2 = {p // 2} cm.",
            f"{b}{c} = {p // 2} - {length} = {_fmt(width)} cm.",
        ],
        next(w for w in range(1, 200) if abs(_rect_perimeter(length, w) - p) < 1e-9),
    )


def _kite_perimeter(x, y):
    """Perimeter of a kite with AB = AD = x and CB = CD = y, from coordinates (AC chosen between |x - y| and x + y)."""
    ac = 0.8 * (x + y) + 0.2 * abs(x - y)
    bx = (x * x - y * y + ac * ac) / (2 * ac)
    by = math.sqrt(x * x - bx * bx)
    pa, pc, pb, pd = (0.0, 0.0), (ac, 0.0), (bx, by), (bx, -by)
    return _dist(pa, pb) + _dist(pb, pc) + _dist(pc, pd) + _dist(pd, pa)


def _e_kite_perim(rng):
    nm, a, b, c, d = _names(rng)
    x, y = rng.randint(4, 14), rng.randint(5, 20)
    if x == y:
        y += 3
    text = (
        f"A kite has two pairs of equal adjacent sides. In kite {nm}, {a}{b} = {a}{d} = {x} cm and "
        f"{c}{b} = {c}{d} = {y} cm. Find the perimeter of the kite in cm."
    )
    return Case(
        text,
        2 * x + 2 * y,
        "cm",
        [(x + y, M_PAIR_ONCE), (2 * x + y, M_PAIR_ONCE), (x + 2 * y, M_PAIR_ONCE), (4 * max(x, y), M_PERIM)],
        [f"Each pair of equal sides is counted twice: 2 x {x} + 2 x {y}.", f"Perimeter = {_fmt(2 * x + 2 * y)} cm."],
        _kite_perimeter(x, y),
    )


# -- Medium ----------------------------------------------------------------------


def _m_cointerior(rng):
    nm, a, b, c, d = _names(rng)
    t = rng.choice([x for x in range(52, 129) if x != 90])
    text = phrase(
        rng,
        [
            f"Angles on the same side of a parallelogram add to 180 degrees. In parallelogram {nm}, angle {a}{b}{c} = {t} degrees. "
            f"Find angle {b}{c}{d} in degrees.",
            f"In parallelogram {nm}, angle {a}{b}{c} is {t} degrees. Consecutive angles of a parallelogram add to 180 degrees. "
            f"Find angle {b}{c}{d} in degrees.",
        ],
    )
    pa, pb, pc, pd = _parallelogram_from_sides(7, 5, 180 - t)  # angle ABC = 180 - angle DAB
    check(
        abs(_angle_at(pa, pb, pc) - t) < 1e-6, "f10c_quadrilateral_properties: coordinate model misses the given angle"
    )
    return Case(
        text,
        180 - t,
        "degrees",
        [(t, M_CONSEC_EQ), (360 - t, M_TOTAL_360), (t / 2, M_HALVE_ANGLE), (90 - t / 2, M_HALVE_ANGLE)],
        [
            f"{b}{c} is a transversal between parallel sides, so the angles at {b} and {c} add to 180.",
            f"180 - {t} = {_fmt(180 - t)}.",
        ],
        _angle_at(pb, pc, pd),
        lo=0,
        hi=180,
    )


def _m_rhombus_base(rng):
    nm, a, b, c, d = _names(rng)
    t = rng.choice([x for x in range(40, 141) if x != 90])
    text = (
        f"All four sides of a rhombus are equal, so a diagonal cuts it into isosceles triangles. In rhombus {nm}, "
        f"angle {a}{b}{c} = {t} degrees. Find angle {b}{a}{c} in degrees."
    )
    pa, pb, pc, pd = _rhombus_from_side_angle(6, 180 - t)
    return Case(
        text,
        (180 - t) / 2,
        "degrees",
        [(t / 2, M_HALVE_ANGLE), (180 - t, M_NO_HALF), (90 - t, M_RIGHT_ANGLE), (t, M_COPY)],
        [
            f"{b}{a} = {b}{c}, so triangle {a}{b}{c} is isosceles and angles {b}{a}{c} and {b}{c}{a} are equal.",
            f"(180 - {t}) / 2 = {_fmt((180 - t) / 2)}.",
        ],
        _angle_at(pb, pa, pc),
        lo=0,
        hi=180,
    )


def _m_rhombus_perp(rng):
    nm, a, b, c, d = _names(rng)
    u = rng.choice([x for x in range(15, 76) if x != 45])
    text = (
        f"The diagonals of a rhombus cross at right angles. In rhombus {nm}, the diagonals {a}{c} and {b}{d} cross at M "
        f"and angle {b}{a}{c} = {u} degrees. Find angle {a}{b}{d} in degrees."
    )
    m_, n_ = math.cos(math.radians(u)), math.sin(math.radians(u))
    pa, pb, pc, pd = (0.0, 0.0), (m_, n_), (2 * m_, 0.0), (m_, -n_)
    return Case(
        text,
        90 - u,
        "degrees",
        [(u, M_COPY), (180 - u, M_RIGHT_ANGLE), (90 + u, M_RIGHT_ANGLE), (u / 2, M_HALVE_ANGLE)],
        [
            f"Triangle {a}M{b} has a right angle at M and angle {u} degrees at {a}.",
            f"Angle {a}{b}{d} = 90 - {u} = {_fmt(90 - u)}.",
        ],
        _angle_at(pa, pb, pd),
        lo=0,
        hi=180,
    )


def _m_rect_isosceles(rng):
    nm, a, b, c, d = _names(rng)
    t = rng.choice([x for x in range(40, 141) if x != 90])
    ask_ab = rng.random() < 0.5
    text = (
        f"The diagonals of a rectangle are equal and bisect each other, so M is the same distance from every corner. "
        f"In rectangle {nm}, the diagonals cross at M and angle {a}M{b} = {t} degrees. "
        f"Find angle M{a}{b if ask_ab else d} in degrees."
    )
    r = 5.0
    pm, pa = (0.0, 0.0), (r, 0.0)
    pb = (r * math.cos(math.radians(t)), r * math.sin(math.radians(t)))
    pc, pd = (-pa[0], -pa[1]), (-pb[0], -pb[1])
    if ask_ab:
        return Case(
            text,
            (180 - t) / 2,
            "degrees",
            [(t / 2, M_HALVE_ANGLE), (180 - t, M_NO_HALF), (90 - t, M_RIGHT_ANGLE), (t, M_COPY)],
            [
                f"M{a} = M{b}, so triangle {a}M{b} is isosceles.",
                f"Angle M{a}{b} = (180 - {t}) / 2 = {_fmt((180 - t) / 2)}.",
            ],
            _angle_at(pm, pa, pb),
            lo=0,
            hi=180,
        )
    return Case(
        text,
        t / 2,
        "degrees",
        [((180 - t) / 2, M_OTHER_SIDE), (180 - t, M_NO_HALF), (t, M_COPY), (90 - t, M_RIGHT_ANGLE)],
        [
            f"Angle M{a}{b} = (180 - {t}) / 2 and angle {d}{a}{b} = 90 degrees (rectangle).",
            f"Angle M{a}{d} = 90 - (180 - {t}) / 2 = {_fmt(t / 2)}.",
        ],
        _angle_at(pm, pa, pd),
        lo=0,
        hi=180,
    )


def _m_kite_angle(rng):
    nm, a, b, c, d = _names(rng)
    alpha, beta = rng.randint(40, 100), rng.randint(70, 130)
    gamma = 360 - alpha - 2 * beta
    text = (
        f"A kite has two pairs of equal adjacent sides, and the angles between the unequal sides are equal. "
        f"In kite {nm}, {a}{b} = {a}{d} and {c}{b} = {c}{d}. Angle {d}{a}{b} = {alpha} degrees and "
        f"angle {a}{b}{c} = {beta} degrees. Find angle {b}{c}{d} in degrees."
    )
    # Coordinates: A at the origin, AB = AD = 1, angle DAB = alpha, C on the axis of symmetry with angle ABC = beta.
    half = math.radians(alpha / 2)
    acb = math.radians(180 - alpha / 2 - beta)
    ac = math.sin(math.radians(beta)) / math.sin(acb)
    pa, pc = (0.0, 0.0), (ac, 0.0)
    pb, pd = (math.cos(half), math.sin(half)), (math.cos(half), -math.sin(half))
    check(abs(_angle_at(pa, pb, pc) - beta) < 1e-6, "f10c_quadrilateral_properties: kite model misses the given angle")
    return Case(
        text,
        gamma,
        "degrees",
        [(360 - alpha - beta, M_KITE), (360 - 2 * alpha - beta, M_KITE), (180 - alpha, M_RIGHT_ANGLE), (beta, M_COPY)],
        [
            f"Angle {a}{d}{c} = angle {a}{b}{c} = {beta} degrees (the equal kite angles).",
            f"The four angles add to 360: {alpha} + 2 x {beta} + angle {b}{c}{d} = 360, so {b}{c}{d} = {gamma}.",
        ],
        _angle_at(pb, pc, pd),
        lo=0,
        hi=180,
    )


def _m_para_two_angles(rng):
    nm, a, b, c, d = _names(rng)
    t = rng.choice([x for x in range(52, 129) if x != 90])
    text = (
        f"In a parallelogram, opposite angles are equal and consecutive angles add to 180 degrees. "
        f"In parallelogram {nm}, angle {a}{b}{c} = {t} degrees. Find the sum of angles {b}{a}{d} and {b}{c}{d} in degrees."
    )
    pa, pb, pc, pd = _parallelogram_from_sides(7, 5, 180 - t)
    return Case(
        text,
        360 - 2 * t,
        "degrees",
        [
            (180 - t, M_ONE_ANGLE),
            (2 * t, M_CONSEC_EQ),
            (360 - t, M_TOTAL_360),
            (t, M_CONSEC_EQ),
            ((180 - t) / 2, M_HALVE_ANGLE),
        ],
        [
            f"Angle {b}{a}{d} = 180 - {t} and angle {b}{c}{d} = 180 - {t} (both are next to angle {a}{b}{c}).",
            f"Sum = 2 x {180 - t} = {_fmt(360 - 2 * t)}.",
        ],
        _angle_at(pb, pa, pd) + _angle_at(pb, pc, pd),
        lo=0,
        hi=360,
    )


def _m_triangle_perim(rng):
    nm, a, b, c, d = _names(rng)
    m_half, n_half = rng.randint(5, 20), rng.randint(5, 20)
    if m_half == n_half:
        n_half += 1
    ac, bd = 2 * m_half, 2 * n_half
    ab = rng.randint(abs(m_half - n_half) + 2, m_half + n_half - 2)
    text = (
        f"The diagonals of a parallelogram bisect each other. In parallelogram {nm}, the diagonals meet at M, "
        f"{a}{c} = {ac} cm, {b}{d} = {bd} cm and {a}{b} = {ab} cm. Find the perimeter of triangle {a}{b}M in cm."
    )
    # Place A and C, then B from AB and BD (D = A + C - B), and measure the perimeter of ABM from the coordinates.
    bx = (ac * ac + 4 * ab * ab - bd * bd) / (4 * ac)
    by = math.sqrt(ab * ab - bx * bx)
    pa, pc, pb = (0.0, 0.0), (float(ac), 0.0), (bx, by)
    pd = (pa[0] + pc[0] - pb[0], pa[1] + pc[1] - pb[1])
    m = _cross(pa, pc, pb, pd)
    check(
        abs(_dist(pb, pd) - bd) < 1e-6, "f10c_quadrilateral_properties: parallelogram model misses the given diagonal"
    )
    return Case(
        text,
        ab + m_half + n_half,
        "cm",
        [(ab + ac + bd, M_WHOLE_DIAG), (ab + m_half, M_OMITS), (m_half + n_half, M_OMITS), (ab + m_half + bd, M_DIAG)],
        [
            f"{a}M = {ac} / 2 and {b}M = {bd} / 2.",
            f"Perimeter = {ab} + {m_half} + {n_half} = {_fmt(ab + m_half + n_half)} cm.",
        ],
        _dist(pa, pb) + _dist(pa, m) + _dist(pb, m),
    )


# -- Hard -------------------------------------------------------------------------


def _solve_linear_by_trial(lhs, rhs, lo=1, hi=60):
    """The whole value of x that makes lhs(x) == rhs(x), by trying each one."""
    hits = [x for x in range(lo, hi) if lhs(x) == rhs(x)]
    check(len(hits) == 1, "f10c_quadrilateral_properties: algebra check found no unique whole solution")
    return hits[0]


def _h_alg_sides(rng):
    nm, a, b, c, d = _names(rng)
    while True:
        x = rng.randint(3, 12)
        p = rng.randint(2, 5)
        r = p + rng.randint(2, 4)
        q = rng.randint(1, 9)
        s = (r - p) * x - q
        if s > 0:
            break
    text = (
        f"Opposite sides of a parallelogram are equal. In parallelogram {nm}, {a}{b} = ({p}x + {q}) cm and "
        f"{d}{c} = ({r}x - {s}) cm. Find the length of {a}{b} in cm."
    )
    xs = _solve_linear_by_trial(lambda v: p * v + q, lambda v: r * v - s)
    xslip = (s - q) / (r - p)
    cands = [(x, M_XVALUE), (2 * (p * x + q), M_ADD_OPP)]
    if _is_half(xslip) and xslip > 0:
        cands.append((p * xslip + q, M_SLIP))
    cands.append((p * x - q, M_SLIP))
    return Case(
        text,
        p * x + q,
        "cm",
        cands,
        [
            f"{a}{b} = {d}{c}, so {p}x + {q} = {r}x - {s}, giving x = {x}.",
            f"{a}{b} = {p} x {x} + {q} = {_fmt(p * x + q)} cm.",
        ],
        p * xs + q,
    )


def _h_alg_angles(rng):
    nm, a, b, c, d = _names(rng)
    while True:
        x = rng.randint(15, 40)
        p = rng.randint(2, 4)
        r = p + rng.randint(1, 3)
        q = rng.randint(5, 30)
        s = (r - p) * x - q
        angle = p * x + q
        if s > 0 and 20 < angle < 160 and angle != 90:
            break
    text = (
        f"In a parallelogram, opposite angles are equal and consecutive angles add to 180 degrees. In parallelogram {nm}, "
        f"angle {d}{a}{b} = ({p}x + {q}) degrees and angle {b}{c}{d} = ({r}x - {s}) degrees. Find angle {a}{b}{c} in degrees."
    )
    xs = _solve_linear_by_trial(lambda v: p * v + q, lambda v: r * v - s)
    xslip = (s + q) / (r + p)
    cands = [(angle, M_CONSEC_EQ), (x, M_XVALUE), (360 - angle, M_TOTAL_360), (angle / 2, M_HALVE_ANGLE)]
    if _is_half(xslip) and xslip > 0:
        cands.append((180 - (p * xslip + q), M_SLIP))
    return Case(
        text,
        180 - angle,
        "degrees",
        cands,
        [
            f"Opposite angles are equal: {p}x + {q} = {r}x - {s}, so x = {x} and the angle is {angle} degrees.",
            f"Angle {a}{b}{c} = 180 - {angle} = {180 - angle} degrees.",
        ],
        180 - (p * xs + q),
        lo=0,
        hi=180,
    )


def _h_rhombus_side(rng):
    nm, a, b, c, d = _names(rng)
    x, y, z = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        x, y = y, x
    d1, d2 = 2 * x, 2 * y
    text = (
        f"The diagonals of a rhombus bisect each other at right angles. In rhombus {nm}, the diagonals {a}{c} = {d1} cm "
        f"and {b}{d} = {d2} cm. Find the length of side {a}{b} in cm."
    )
    pa, pc = (-x, 0.0), (x, 0.0)
    pb, pd = (0.0, y), (0.0, -y)
    return Case(
        text,
        z,
        "cm",
        [(x + y, M_PYTH_ADD), (math.hypot(d1, d2), M_WHOLE_DIAG), (abs(y - x), M_PYTH_SUB), (z + x, M_OMITS)],
        [
            f"The diagonals bisect each other: half-diagonals {x} cm and {y} cm, at a right angle.",
            f"{a}{b} = sqrt({x}^2 + {y}^2) = {_fmt(z)} cm.",
        ],
        _dist(pa, pb),
    )


def _h_rect_centre(rng):
    nm, a, b, c, d = _names(rng)
    x, y, z = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        x, y = y, x
    text = (
        f"The diagonals of a rectangle are equal and bisect each other. Rectangle {nm} has {a}{b} = {x} cm and "
        f"{b}{c} = {y} cm, and its diagonals cross at M. Find the length {a}M in cm."
    )
    pa, pb, pc, pd = (0.0, 0.0), (x, 0.0), (x, y), (0.0, y)
    m = _cross(pa, pc, pb, pd)
    return Case(
        text,
        z / 2,
        "cm",
        [(z, M_DIAG), (x + y, M_PYTH_ADD), ((x + y) / 2, M_AVG), (abs(y - x), M_PYTH_SUB)],
        [f"{a}{c} = sqrt({x}^2 + {y}^2) = {z} cm (right angle at {b}).", f"{a}M is half of {a}{c}: {_fmt(z / 2)} cm."],
        _dist(pa, m),
    )


def _h_kite_axis(rng):
    nm, a, b, c, d = _names(rng)
    leg, am, ab = rng.choice(
        [
            (12, 5, 13),
            (15, 8, 17),
            (6, 8, 10),
            (8, 6, 10),
            (24, 7, 25),
            (9, 12, 15),
            (16, 12, 20),
            (20, 21, 29),
            (12, 35, 37),
            (15, 20, 25),
            (10, 24, 26),
        ]
    )
    text = (
        f"In a kite the diagonal joining the two vertices where equal sides meet is the perpendicular bisector of the other "
        f"diagonal. In kite {nm}, {a}{b} = {a}{d} = {ab} cm and {c}{b} = {c}{d}. The diagonals cross at M and {b}{d} = {2 * leg} cm. "
        f"Find the length {a}M in cm."
    )
    ax = _solve_linear_by_trial(lambda v: v * v + leg * leg, lambda v: ab * ab, 1, 60)
    return Case(
        text,
        am,
        "cm",
        [(leg, M_PYTH_ADD), (ab - leg, M_PYTH_SUB), (ab + leg, M_PYTH_ADD), (2 * leg, M_WHOLE_DIAG)],
        [
            f"{b}M = {2 * leg} / 2 = {leg} cm and angle {a}M{b} is a right angle.",
            f"{a}M = sqrt({ab}^2 - {leg}^2) = {_fmt(am)} cm.",
        ],
        float(ax),
    )


def _h_alg_diag(rng):
    nm, a, b, c, d = _names(rng)
    while True:
        x = rng.randint(3, 12)
        p = rng.randint(2, 4)
        r = p + rng.randint(1, 3)
        q = rng.randint(1, 8)
        s = (r - p) * x - q
        if s > 0:
            break
    text = (
        f"The diagonals of a parallelogram bisect each other. In parallelogram {nm}, the diagonals meet at M, "
        f"{a}M = ({p}x + {q}) cm and M{c} = ({r}x - {s}) cm. Find the length of the diagonal {a}{c} in cm."
    )
    xs = _solve_linear_by_trial(lambda v: p * v + q, lambda v: r * v - s)
    cands = [(p * x + q, M_DIAG), (x, M_XVALUE), ((p * x + q) * 4, M_HALVE_TWICE), (2 * x, M_XVALUE)]
    return Case(
        text,
        2 * (p * x + q),
        "cm",
        cands,
        [
            f"M is the midpoint of {a}{c}, so {p}x + {q} = {r}x - {s} and x = {x}.",
            f"{a}{c} = 2 × ({p} × {x} + {q}) = {_fmt(2 * (p * x + q))} cm.",
        ],
        2 * (p * xs + q),
    )


def _h_rect_exterior(rng):
    nm, a, b, c, d = _names(rng)
    u = rng.choice([x for x in range(20, 71) if x != 45])
    text = (
        f"The diagonals of a rectangle are equal and bisect each other, so the diagonals cross at M with M{b} = M{c}. "
        f"In rectangle {nm}, angle {a}{c}{b} = {u} degrees. Find angle {a}M{b} in degrees."
    )
    w = 5.0
    h = w * math.tan(math.radians(u))
    pb, pc, pa, pd = (0.0, 0.0), (w, 0.0), (0.0, h), (w, h)
    m = _cross(pa, pc, pb, pd)
    check(
        abs(_angle_at(pa, pc, pb) - u) < 1e-6, "f10c_quadrilateral_properties: rectangle model misses the given angle"
    )
    return Case(
        text,
        2 * u,
        "degrees",
        [(180 - 2 * u, M_SUPP), (u, M_COPY), (90 - u, M_RIGHT_ANGLE), (180 - u, M_RIGHT_ANGLE)],
        [
            f"M{b} = M{c}, so angle M{b}{c} = angle M{c}{b} = {u} degrees.",
            f"Angle {a}M{b} is the exterior angle of triangle M{b}{c}: {u} + {u} = {2 * u} degrees.",
        ],
        _angle_at(pa, m, pb),
        lo=0,
        hi=180,
    )


_CASES = {
    "easy": [_e_diag_half, _e_diag_double, _e_rhombus_perim, _e_rect_perim, _e_kite_perim],
    "medium": [
        _m_cointerior,
        _m_rhombus_base,
        _m_rhombus_perp,
        _m_rect_isosceles,
        _m_kite_angle,
        _m_para_two_angles,
        _m_triangle_perim,
    ],
    "hard": [_h_alg_sides, _h_alg_angles, _h_rhombus_side, _h_rect_centre, _h_kite_axis, _h_alg_diag, _h_rect_exterior],
}


@generator(
    "f10c_quadrilateral_properties",
    level=8,
    presets={t: dict(tier=t) for t in TIERS},
)
def f10c_quadrilateral_properties(rng, tier="easy"):
    """Apply a stated quadrilateral property to find an unknown length or angle (see the module docstring for the
    cases at each tier)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {list(TIERS)}")
    case = rng.choice(_CASES[tier])(rng)

    check(_is_half(case.value), "f10c_quadrilateral_properties: answer is not a multiple of 0.5")
    check(
        abs(case.truth - case.value) < 1e-6,
        f"f10c_quadrilateral_properties: independent route gives {case.truth}, answer is {case.value}",
    )

    answer_text = _fmt(case.value)
    pool = []
    notes = {}
    for v, note in case.cands:
        if not (case.lo < v < case.hi) or not _is_half(v):
            continue
        txt = _fmt(v)
        if txt not in notes:
            pool.append(txt)
            notes[txt] = note
    distractors = pick_distractors(answer_text, pool, n=3, tolerance=GRADER_TOLERANCE)
    check(len(distractors) >= 2, "f10c_quadrilateral_properties: fewer than two distractors survived")
    # Offset the pool's order so the same misconception is not always first.
    rng.shuffle(distractors)

    body = re.sub(r" in (cm|degrees)\.$", ".", case.text)
    unit_line = "in cm" if case.unit == "cm" else "in degrees"
    text = f"{body} Give your answer {unit_line}, to 1 decimal place."
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(case.value),
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        working=case.working,
        distractors=distractors,
        option_notes={d: notes[d] for d in distractors},
    )
