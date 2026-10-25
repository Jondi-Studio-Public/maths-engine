"""Generates `nx_complex_arg_ratio` (easy, medium, hard) and the exam chain `exd_nx_complex_arg_ratio`.

Skill ``complex-arg-ratio-geometry`` (``senior-mathematics``, answer ``expression``). The existing
``complex-rotation-and-vertex-completion`` covers rotating a point and completing a square or an equilateral triangle;
this one covers the rest of the geometric toolkit: the ratio (z1 - z2)/(z3 - z2) as the test for collinear and
perpendicular points and as the angle at a vertex, the image of a point under w = az + b, midpoint and distance.

* easy: the image of a point under w = az + b; the midpoint of two points; the distance between two points.
* medium: the ratio (C - B)/(A - B) as a + bi; the multiplier a from the images of two points under w = az + b; the
  real k that puts C = k + ci on the line through A and B.
* hard: the angle ABC in degrees from arg((C - B)/(A - B)) (right, 45 and 135 degree angles, and 30, 60, 120 and 150
  degrees with surd coordinates); the real k that makes the angle at B a right angle.
* exam (``exd_nx_complex_arg_ratio``): from A, B, C, (a) the ratio (C - B)/(A - B), (b) the angle ABC following through
  from (a), (c) the midpoint M of AC, (d) the point D that makes M the midpoint of BD, following through from (c).

Every key is re-derived by a second route in floating point (the angle from ``cmath.phase``, the collinearity and
perpendicularity from the cross and dot products of the sides).
"""

from __future__ import annotations

import cmath
import math

import sympy

from ..core import attempts, check, generator, paren_negative, phrase
from ._w6b_kit import I, cdist, cx_text, cx_value, dist, gaussian, need, plus, retry, want

_MULTIPLIERS = (2, 3, -1, -2, I, -I, 1 + I, 1 - I, 2 * I, 2 + I, 1 + 2 * I)
_PYTHAGOREAN = ((3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (8, 15), (15, 8), (9, 12), (12, 9))


def _g(rng, span, *, parts_nonzero=True):
    for _ in attempts(500, "a Gaussian integer"):
        a, b = rng.randint(-span, span), rng.randint(-span, span)
        if parts_nonzero and (a == 0 or b == 0):
            continue
        return gaussian(a, b)
    raise AssertionError  # pragma: no cover


def _cmp(z) -> complex:
    return complex(sympy.N(z))


def _num(value, tag):
    return dist(str(value), tag, value)


def _expr_fields(z):
    return dict(answer_kind="expression", answer=cx_value(z), answer_text=cx_text(z))


def _num_fields(v):
    return dict(answer_kind="number", answer=v, answer_text=str(v))


def _deg_fields(v):
    return dict(answer_kind="number", answer=v, answer_text=str(v))


def _kc(c: int) -> str:
    """The imaginary part of k + ci as it follows the k: ``+ 3i`` or ``- 7i``."""
    return f"{'+' if c > 0 else '-'} {abs(c)}i"


def _br(z) -> str:
    """The complex number as text, bracketed unless it is a non-negative whole number or i."""
    t = cx_text(z)
    return t if t == "i" or (sympy.im(z) == 0 and sympy.re(z) >= 0) else f"({t})"


# -- easy -----------------------------------------------------------------------------------------------------

_MAP_STEMS = (
    "A transformation of the complex plane is w = {a}z + {b}. Find the image of the point {z} under this transformation.",
    "The point P represents z = {z}. P is mapped to P' by w = {a}z + {b}. Which complex number does P' represent?",
    "Under the map z -> {a}z + {b} of the Argand diagram, where does the point {z} go? Give the complex number "
    "that its image represents.",
)
_MID_STEMS = (
    "On an Argand diagram, A represents {a} and B represents {b}. Find the complex number represented by the midpoint of AB.",
    "The points {a} and {b} are the ends of a line segment in the complex plane. What complex number marks the midpoint "
    "of the segment?",
    "M is the midpoint of the interval joining the points {a} and {b}. Write M as a complex number.",
)
_DIST_STEMS = (
    "Find the distance between the points {a} and {b} on the complex plane.",
    "A and B represent {a} and {b} on an Argand diagram. How far apart are A and B?",
    "Two points of the complex plane represent {a} and {b}. Find the length of the segment joining them.",
)


def _bracket_free(z) -> str:
    return cx_text(z)


def _easy_map(rng):
    a = rng.choice(_MULTIPLIERS)
    b = _g(rng, 6, parts_nonzero=False)
    z = _g(rng, 6)
    want(sympy.expand(a * z) != z and b != 0)
    w = sympy.expand(a * z + b)
    check(abs(_cmp(w) - (_cmp(a) * _cmp(z) + _cmp(b))) < 1e-9, "image differs")
    cands = [
        cdist(a * z, "nx-map-constant-dropped"),
        cdist(z + b, "nx-map-multiplier-dropped"),
        cdist(a * (z + b), "nx-map-adds-before-multiplying"),
        cdist(a * z - b, "nx-map-constant-subtracted"),
        cdist(
            sympy.expand(sympy.re(a) * sympy.re(z) + sympy.re(b))
            + I * sympy.expand(sympy.im(a) * sympy.im(z) + sympy.im(b)),
            "nx-map-multiplies-parts-separately",
        ),
    ]
    working = [
        f"Multiply: ({cx_text(a)})({cx_text(z)}) = {cx_text(a * z)}.",
        f"Add the constant: {cx_text(a * z)} {plus(cx_text(b))} = {cx_text(w)}." if b != 0 else "No constant to add.",
    ]
    return dict(
        text=phrase(rng, _MAP_STEMS, a=_br(a), b=_br(b), z=cx_text(z)),
        working=working,
        distractors=need(cx_text(w), cands, 4),
        **_expr_fields(w),
    )


def _easy_mid(rng):
    for _ in attempts(500, "two points of one parity"):
        a, b = _g(rng, 8), _g(rng, 8)
        if sympy.re(a - b) % 2 == 0 and sympy.im(a - b) % 2 == 0 and a != b:
            break
    m = sympy.expand((a + b) / 2)
    cands = [
        cdist(a + b, "nx-midpoint-sum-not-halved"),
        cdist(sympy.expand((a - b) / 2), "nx-midpoint-half-the-difference"),
        cdist(sympy.expand(sympy.re(a + b) / 2 + I * sympy.im(a + b)), "nx-midpoint-halves-real-part-only"),
        cdist(sympy.expand(sympy.im(a + b) / 2 + I * sympy.re(a + b) / 2), "nx-midpoint-parts-swapped"),
    ]
    working = [
        f"The midpoint is (a + b)/2 = ({cx_text(a)} {plus(cx_text(b))})/2.",
        f"= ({cx_text(a + b)})/2 = {cx_text(m)}.",
    ]
    return dict(
        text=phrase(rng, _MID_STEMS, a=cx_text(a), b=cx_text(b)),
        working=working,
        distractors=need(cx_text(m), cands, 4),
        **_expr_fields(m),
    )


def _easy_dist(rng):
    p, q = rng.choice(_PYTHAGOREAN)
    p, q = rng.choice([-1, 1]) * p, rng.choice([-1, 1]) * q
    a = _g(rng, 6)
    b = sympy.expand(a + gaussian(p, q))
    d = int(math.hypot(p, q))
    check(d * d == p * p + q * q and abs(abs(_cmp(a) - _cmp(b)) - d) < 1e-9, "distance differs")
    cands = [
        _num(abs(p) + abs(q), "nx-distance-adds-the-parts"),
        _num(p * p + q * q, "nx-distance-forgets-square-root"),
        _num(max(abs(p), abs(q)), "nx-distance-takes-the-larger-part"),
        _num(abs(p - q) if abs(p) != abs(q) else abs(p), "nx-distance-subtracts-the-parts"),
    ]
    working = [
        f"a - b = {cx_text(a - b)}.",
        f"|a - b| = √({paren_negative(p)}² + {paren_negative(q)}²) = √{p * p + q * q} = {d}.",
    ]
    return dict(
        text=phrase(rng, _DIST_STEMS, a=cx_text(a), b=cx_text(b)),
        working=working,
        distractors=need(str(d), cands, 4),
        **_num_fields(d),
    )


def _easy(rng):
    return retry(rng, rng.choice([_easy_map, _easy_map, _easy_mid, _easy_dist]))


# -- medium ---------------------------------------------------------------------------------------------------

_RATIO_STEMS = (
    "The points A, B and C on an Argand diagram represent {a}, {b} and {c}. Write (C - B)/(A - B) in the form x + yi.",
    "On the complex plane, A = {a}, B = {b} and C = {c}. Find the ratio (c - b)/(a - b) of the complex numbers "
    "represented, in the form x + yi.",
    "Let A, B and C represent {a}, {b} and {c}. Calculate the quotient of the vectors BC and BA, that is "
    "(c - b)/(a - b), giving x + yi.",
)
_MULT_STEMS = (
    "A transformation w = az + b, where a and b are complex constants, maps {z1} to {w1} and {z2} to {w2}. Find a.",
    "The map z -> az + b of the complex plane sends the point {z1} to {w1} and the point {z2} to {w2}. Find the "
    "complex number a.",
    "Suppose w = az + b sends {z1} to {w1} and {z2} to {w2}. Work out the constant a.",
)
_COLL_STEMS = (
    "The points A, B and C represent {a}, {b} and k {c}, where k is real. A, B and C lie on a straight line, so "
    "(C - A)/(B - A) is real. Find k.",
    "On an Argand diagram, A = {a}, B = {b} and C = k {c} for a real number k. The three points are collinear. Find k.",
    "Find the real number k so that the points {a}, {b} and k {c} are collinear on the complex plane.",
)


def _medium_ratio(rng):
    d = _g(rng, 4, parts_nonzero=False)
    want(d != 0)
    m = rng.choice([1 + I, -1 + I, 1 - I, -1 - I, 2 + I, 1 + 2 * I, 2 - I, -2 + I, 3 + I, 2 * I, -I, 1 - 2 * I, 3 - I])
    b = _g(rng, 5)
    a = sympy.expand(b + d)
    c = sympy.expand(b + d * m)
    q = sympy.expand(m)
    check(abs((_cmp(c) - _cmp(b)) / (_cmp(a) - _cmp(b)) - _cmp(q)) < 1e-9, "ratio differs")
    inv = sympy.expand(sympy.conjugate(m) / (sympy.re(m) ** 2 + sympy.im(m) ** 2))
    cands = [
        cdist(inv, "nx-ratio-inverted"),
        cdist(-q, "nx-ratio-difference-reversed"),
        cdist(sympy.expand((c - b) - (a - b)), "nx-ratio-subtracted-not-divided"),
        cdist(sympy.expand(sympy.conjugate(q)), "nx-ratio-conjugated"),
    ]
    den = sympy.expand(d * sympy.conjugate(d))
    working = [
        f"c - b = {cx_text(c - b)} and a - b = {cx_text(a - b)}.",
        f"({cx_text(c - b)})/({cx_text(a - b)}): multiply top and bottom by the conjugate {cx_text(sympy.conjugate(d))}"
        f" so the denominator becomes {den}.",
        f"The ratio is {cx_text(q)}.",
    ]
    return dict(
        text=phrase(rng, _RATIO_STEMS, a=cx_text(a), b=cx_text(b), c=cx_text(c)),
        working=working,
        distractors=need(cx_text(q), cands, 4),
        **_expr_fields(q),
    )


def _medium_multiplier(rng):
    a = rng.choice([2, 3, -1, -2, I, -I, 1 + I, 1 - I, 2 * I, 2 + I, 1 + 2 * I, 2 - I, -1 + I, 3 - I])
    b = _g(rng, 6, parts_nonzero=False)
    z1, z2 = _g(rng, 5), _g(rng, 5)
    want(z1 != z2)
    w1, w2 = sympy.expand(a * z1 + b), sympy.expand(a * z2 + b)
    check(abs((_cmp(w1) - _cmp(w2)) / (_cmp(z1) - _cmp(z2)) - _cmp(a)) < 1e-9, "multiplier differs")
    dz = z1 - z2
    dw = w1 - w2
    s = sympy.expand(z1 + z2)
    cands = [
        cdist(dw, "nx-multiplier-difference-not-divided"),
        cdist(sympy.expand(dz / dw) if dw != 0 else 0, "nx-multiplier-ratio-inverted"),
        cdist(sympy.expand(w1 / z1), "nx-multiplier-from-one-point-only"),
        cdist(sympy.expand((w1 + w2) / s) if s != 0 else 0, "nx-multiplier-sums-not-differences"),
        cdist(sympy.expand(w1 - w2 - dz), "nx-multiplier-difference-of-differences"),
    ]
    working = [
        f"Subtracting the two equations removes b: w1 - w2 = a(z1 - z2).",
        f"w1 - w2 = {cx_text(dw)} and z1 - z2 = {cx_text(dz)}.",
        f"a = ({cx_text(dw)})/({cx_text(dz)}) = {cx_text(a)}.",
    ]
    return dict(
        text=phrase(rng, _MULT_STEMS, z1=cx_text(z1), w1=cx_text(w1), z2=cx_text(z2), w2=cx_text(w2)),
        working=working,
        distractors=need(cx_text(a), cands, 4),
        **_expr_fields(a),
    )


def _medium_collinear(rng):
    for _ in attempts(500, "a collinear draw"):
        a = _g(rng, 6)
        u, v = rng.randint(-3, 3), rng.randint(-3, 3)
        if u == 0 or v == 0:
            continue
        t = rng.choice([2, 3, -1, -2, 4, -3])
        break
    d = gaussian(2 * u, 2 * v) if rng.random() < 0.3 else gaussian(u, v)
    dx, dy = int(sympy.re(d)), int(sympy.im(d))
    b = sympy.expand(a + d)
    ax, ay = int(sympy.re(a)), int(sympy.im(a))
    k = ax + t * dx
    c = ay + t * dy
    want(c != 0 and k != ax and k != int(sympy.re(b)))
    point = gaussian(k, c)
    # second route: the cross product of the sides vanishes
    cross = (_cmp(b) - _cmp(a)).real * (_cmp(point) - _cmp(a)).imag - (_cmp(b) - _cmp(a)).imag * (
        _cmp(point) - _cmp(a)
    ).real
    check(abs(cross) < 1e-9, "points are not collinear")
    cands = [
        _num(ax + t * dy, "nx-collinear-parts-swapped"),
        _num(int(sympy.re(b)) + t * dx, "nx-collinear-wrong-base-point"),
        _num(ax - t * dx, "nx-collinear-sign-slip"),
        _num(ax + (c * dx) // dy if (c * dx) % dy == 0 else ax + c + dx, "nx-collinear-imaginary-offset-ignored"),
    ]
    working = [
        f"(C - A)/(B - A) is real, where C - A = (k - ({ax})) {_kc(c - ay)} and B - A = {cx_text(d)}.",
        f"Use the cross product: ({dx})({c - ay}) = ({dy})(k - ({ax})), so k - ({ax}) = {t * dx}.",
        f"k = {k}.",
    ]
    return dict(
        text=phrase(rng, _COLL_STEMS, a=cx_text(a), b=cx_text(b), c=_kc(c)),
        working=working,
        distractors=need(str(k), cands, 4),
        answer_kind="number",
        answer=k,
        answer_text=str(k),
    )


def _medium(rng):
    return retry(
        rng, rng.choice([_medium_ratio, _medium_ratio, _medium_multiplier, _medium_multiplier, _medium_collinear])
    )


# -- hard -----------------------------------------------------------------------------------------------------

_ANGLE_STEMS = (
    "On an Argand diagram, A, B and C represent {a}, {b} and {c}. Use arg((c - b)/(a - b)) to find the size of angle "
    "ABC in degrees.",
    "The vertices of triangle ABC are A = {a}, B = {b} and C = {c} on the complex plane. By finding the argument of "
    "(c - b)/(a - b), state angle ABC in degrees.",
    "Triangle ABC has A({a}), B({b}) and C({c}) on an Argand diagram. Find the angle at B, in degrees, using the "
    "argument of the quotient of the vectors BC and BA.",
)
_PERP_STEMS = (
    "The points A, B and C represent {a}, {b} and k {c}, where k is real. Angle ABC is a right angle, so "
    "(A - B)/(C - B) is purely imaginary. Find k.",
    "On the complex plane, A = {a}, B = {b} and C = k {c}, with k real. BA is perpendicular to BC. Find k.",
    "Find the real number k for which the line from {b} to {a} is perpendicular to the line from {b} to k {c}.",
)

_GAUSS_ANGLES = ((1 + I, 45), (-1 + I, 135), (I, 90), (1 - I, 45), (-1 - I, 135), (-I, 90))
_SURD_ANGLES = (
    (1 + sympy.sqrt(3) * I, 60),
    (-1 + sympy.sqrt(3) * I, 120),
    (sympy.sqrt(3) + I, 30),
    (-sympy.sqrt(3) + I, 150),
    (1 - sympy.sqrt(3) * I, 60),
    (-1 - sympy.sqrt(3) * I, 120),
    (sympy.sqrt(3) - I, 30),
    (-sympy.sqrt(3) - I, 150),
)


def _hard_angle(rng):
    surd = rng.random() < 0.5
    if surd:
        m, theta = rng.choice(_SURD_ANGLES)
        scale = 1
    else:
        m, theta = rng.choice(_GAUSS_ANGLES)
        scale = rng.choice([1, 2, 3])
    if surd:
        d = rng.choice([1, 2, 3, -1, -2, -3, I, 2 * I, -I, -2 * I])
    else:
        d = _g(rng, 4, parts_nonzero=False)
        want(d != 0)
    b = _g(rng, 4)
    a = sympy.expand(b + d)
    c = sympy.expand(b + d * m * scale)
    q = sympy.expand(m * scale)
    # second route: the angle between the two sides from the dot product
    u, w = _cmp(a) - _cmp(b), _cmp(c) - _cmp(b)
    cos_angle = (u.real * w.real + u.imag * w.imag) / (abs(u) * abs(w))
    check(abs(math.degrees(math.acos(max(-1, min(1, cos_angle)))) - theta) < 1e-7, "angle differs")
    check(abs(abs(math.degrees(cmath.phase(_cmp(q)))) - theta) < 1e-7, "argument differs")
    cands = [
        _num(180 - theta, "nx-angle-supplement-taken"),
        _num(abs(90 - theta), "nx-angle-complement-taken") if theta != 90 else None,
        _num(theta // 2 if theta % 2 == 0 else theta + 15, "nx-angle-halved"),
        _num(min(2 * theta, 180) if 2 * theta != theta else theta + 30, "nx-angle-doubled"),
        _num(360 - theta, "nx-angle-reflex-taken"),
    ]
    working = [
        f"(c - b)/(a - b) = ({cx_text(c - b)})/({cx_text(a - b)}) = {cx_text(q)}.",
        f"The angle at B is |arg of this ratio| = {theta}°.",
    ]
    return dict(
        text=phrase(rng, _ANGLE_STEMS, a=cx_text(a), b=cx_text(b), c=cx_text(c)),
        working=working,
        distractors=need(str(theta), [c for c in cands if c], 4),
        **_deg_fields(theta),
    )


def _hard_perp(rng):
    for _ in attempts(500, "a perpendicular draw"):
        p, q = rng.randint(-4, 4), rng.randint(-4, 4)
        s = rng.choice([1, 2, 3, -1, -2, -3])
        if p == 0 or q == 0 or abs(p) == abs(q):
            continue
        break
    b = _g(rng, 5)
    bx, by = int(sympy.re(b)), int(sympy.im(b))
    a = gaussian(bx + p, by + q)
    k, c = bx - s * q, by + s * p
    point = gaussian(k, c)
    # second route: the dot product of the sides vanishes
    u, w = _cmp(a) - _cmp(b), _cmp(point) - _cmp(b)
    check(abs(u.real * w.real + u.imag * w.imag) < 1e-9, "sides are not perpendicular")
    want(c != 0 and k != 0)
    cands = [
        _num(bx + s * q, "nx-perp-uses-parallel-condition"),
        _num(bx - s * p, "nx-perp-parts-swapped"),
        _num(bx + p - s * q, "nx-perp-measured-from-the-wrong-vertex"),
        _num(-s * q, "nx-perp-base-point-ignored"),
    ]
    working = [
        f"BA = {cx_text(a - b)}. C - B = (k - ({bx})) {_kc(c - by)}.",
        f"BA is perpendicular to BC, so the dot product is zero: ({p})(k - ({bx})) + ({q})({c - by}) = 0.",
        f"k - ({bx}) = {-s * q}, so k = {k}.",
    ]
    return dict(
        text=phrase(rng, _PERP_STEMS, a=cx_text(a), b=cx_text(b), c=_kc(c)),
        working=working,
        distractors=need(str(k), cands, 4),
        answer_kind="number",
        answer=k,
        answer_text=str(k),
    )


def _hard(rng):
    return retry(rng, rng.choice([_hard_angle, _hard_angle, _hard_perp]))


@generator(
    "nx_complex_arg_ratio",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def nx_complex_arg_ratio(rng, tier="easy"):
    """Geometry of the complex plane by the ratio of two vectors, w = az + b, midpoints and distances."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -----------------------------------------------------------------------------------------------------

_EXAM_STEMS = (
    "The points A, B and C on an Argand diagram represent the complex numbers {a}, {b} and {c}.",
    "Triangle ABC is drawn on the complex plane with A at {a}, B at {b} and C at {c}.",
    "On an Argand diagram the vertices of triangle ABC are A = {a}, B = {b} and C = {c}.",
)


def _sym(z) -> str:
    return f"{int(sympy.re(z))} + ({int(sympy.im(z))})*i"


def _xfields(z, marks, **extra):
    return dict(answer_kind="expression", answer=_sym(z), answer_text=cx_text(z), marks=marks, **extra)


@generator("exd_nx_complex_arg_ratio", level=17, presets={"exam": dict(tier="exam")})
def exd_nx_complex_arg_ratio(rng, tier="exam"):
    """Exam chain: the ratio (C - B)/(A - B), the angle ABC from it, the midpoint of AC, the point D opposite B."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _exam)


def _exam(rng):
    m, theta = rng.choice(_GAUSS_ANGLES)
    scale = rng.choice([1, 2, 3])
    p, q = rng.randint(-3, 3), rng.randint(-3, 3)
    want((p, q) != (0, 0))
    d = gaussian(p, q)
    b = _g(rng, 4)
    q_ratio = sympy.expand(m * scale)
    a = sympy.expand(b + d)
    c = sympy.expand(b + d * q_ratio)
    want(sympy.re(a + c) % 2 == 0 and sympy.im(a + c) % 2 == 0)
    mid = sympy.expand((a + c) / 2)
    dd = sympy.expand(2 * mid - b)
    check(dd == sympy.expand(a + c - b), "opposite vertex differs")
    check(abs(abs(math.degrees(cmath.phase(_cmp(q_ratio)))) - theta) < 1e-7, "angle differs")
    bx, by = int(sympy.re(b)), int(sympy.im(b))
    inv = sympy.expand(sympy.conjugate(q_ratio) / (sympy.re(q_ratio) ** 2 + sympy.im(q_ratio) ** 2))
    ratio_cands = [
        cdist(inv, "nx-ratio-inverted"),
        cdist(-q_ratio, "nx-ratio-difference-reversed"),
        cdist(sympy.expand(c - a), "nx-ratio-subtracted-not-divided"),
        cdist(sympy.expand(sympy.conjugate(q_ratio)), "nx-ratio-conjugated"),
    ]
    angle_cands = [
        _num(180 - theta, "nx-angle-supplement-taken"),
        _num(abs(90 - theta), "nx-angle-complement-taken") if theta != 90 else None,
        _num(360 - theta, "nx-angle-reflex-taken"),
        _num(2 * theta if 2 * theta <= 180 else theta // 2 + 5, "nx-angle-doubled"),
    ]
    mid_cands = [
        cdist(a + c, "nx-midpoint-sum-not-halved"),
        cdist(
            sympy.expand((c - a) / 2) if (sympy.re(c - a) % 2 == 0 and sympy.im(c - a) % 2 == 0) else a + c - b,
            "nx-midpoint-half-the-difference",
        ),
        cdist(sympy.expand(sympy.re(a + c) / 2 + I * sympy.im(a + c)), "nx-midpoint-halves-real-part-only"),
    ]
    d_cands = [
        cdist(mid, "nx-opposite-vertex-midpoint-taken"),
        cdist(sympy.expand(mid - b), "nx-opposite-vertex-reflected-wrongly"),
        cdist(sympy.expand(2 * mid + b), "nx-opposite-vertex-adds-b"),
        cdist(sympy.expand(a + c), "nx-opposite-vertex-sum-of-a-and-c"),
    ]
    stem = phrase(rng, _EXAM_STEMS, a=cx_text(a), b=cx_text(b), c=cx_text(c))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find (c - b)/(a - b) in the form x + yi.",
                working=[
                    f"c - b = {cx_text(c - b)}, a - b = {cx_text(a - b)}.",
                    f"The ratio is ({cx_text(c - b)})/({cx_text(a - b)}) = {cx_text(q_ratio)}.",
                ],
                distractors=need(cx_text(q_ratio), ratio_cands, 3),
                **_xfields(q_ratio, 2),
            ),
            dict(
                prompt="Hence find the size of angle ABC, in degrees.",
                answer_kind="number",
                answer=theta,
                answer_text=str(theta),
                marks=1,
                follow_through="abs(arg(Subs(part_a,(i,),(I,))))*180/pi",
                sensible_range=(0, 180),
                working=[f"Angle ABC is the size of arg((c - b)/(a - b)), that is {theta}°."],
                distractors=need(str(theta), [c for c in angle_cands if c], 3),
            ),
            dict(
                prompt="Find the complex number that represents M, the midpoint of AC.",
                working=[f"M = (a + c)/2 = ({cx_text(a + c)})/2 = {cx_text(mid)}."],
                distractors=need(cx_text(mid), mid_cands, 3),
                **_xfields(mid, 1),
            ),
            dict(
                prompt="ABCD is a parallelogram, so M is also the midpoint of BD. Hence find the complex number that "
                "represents D.",
                follow_through=f"2*part_c - ({bx} + ({by})*i)",
                working=[
                    "M is the midpoint of BD, so d = 2m - b.",
                    f"d = 2({cx_text(mid)}) - ({cx_text(b)}) = {cx_text(dd)}.",
                ],
                distractors=need(cx_text(dd), d_cands, 3),
                **_xfields(dd, 2),
            ),
        ],
    )
