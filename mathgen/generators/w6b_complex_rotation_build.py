"""Generates `w6b_complex_rotation` (easy, medium, hard) and the exam chain `exd_x_w6b_complex_rotation`.

``complex-rotation-and-vertex-completion`` (``complex-analysis``, answer ``expression``): rotating a point of the
complex plane by multiplying by a unit complex number, and finding the missing vertices of squares and equilateral
triangles from the vertices that are given.

* easy: the image of a Gaussian integer under a rotation of 90 degrees either way, or 180 degrees, about the origin.
* medium: either a quarter turn or half turn about a centre that is not the origin, or a rotation of 30, 45, 60 or 120
  degrees about the origin with an exact surd answer.
* hard: the missing vertex of a square from two adjacent vertices, or from its centre and one vertex, or the third
  vertex of an equilateral triangle, with the labelling direction stated (anticlockwise or clockwise).
* exam (``exd_x_w6b_complex_rotation``): a square ABCD from A and B: the side AB as a complex number, its length, its
  area (following through from the length) and one of the two remaining vertices.

Every answer is re-derived by a second route: the image is checked to lie at the same distance from the centre as the
original and at the stated angle from it, using floating-point complex arithmetic.
"""

from __future__ import annotations

import cmath
import math

import sympy

from ..core import attempts, check, generator, phrase
from ._w6b_kit import I, cdist, cis, cx_text, cx_value, dist, gaussian, need, plus, retry

_PHRASES = {
    90: ("90° anticlockwise", "a quarter turn anticlockwise", "π/2 radians anticlockwise"),
    -90: ("90° clockwise", "a quarter turn clockwise", "π/2 radians clockwise"),
    180: ("180°", "a half turn", "π radians"),
    60: ("60° anticlockwise", "π/3 radians anticlockwise"),
    -60: ("60° clockwise", "π/3 radians clockwise"),
    45: ("45° anticlockwise", "π/4 radians anticlockwise"),
    -45: ("45° clockwise", "π/4 radians clockwise"),
    30: ("30° anticlockwise", "π/6 radians anticlockwise"),
    -30: ("30° clockwise", "π/6 radians clockwise"),
    120: ("120° anticlockwise", "2π/3 radians anticlockwise"),
    -120: ("120° clockwise", "2π/3 radians clockwise"),
}


def _angle_words(rng, deg):
    return rng.choice(_PHRASES[deg])


def _rotate(z, c, deg):
    """The image of z under a rotation of ``deg`` degrees about c, in exact form."""
    return sympy.expand(c + (z - c) * cis(deg))


def _verify_rotation(z, c, deg, w):
    """Second route: |w - c| = |z - c| and (w - c)/(z - c) has argument deg, in floating point."""
    zf, cf, wf = complex(z), complex(c), complex(w)
    check(abs(abs(wf - cf) - abs(zf - cf)) < 1e-9, "the image is at a different distance from the centre")
    turn = cmath.phase((wf - cf) / (zf - cf))
    expected = math.radians(deg)
    diff = (turn - expected + math.pi) % (2 * math.pi) - math.pi
    check(abs(diff) < 1e-9, "the image is turned through a different angle")


def _swapped(deg):
    """cos and sin swapped: the unit number sin + i cos in place of cos + i sin."""
    unit = cis(deg)
    return sympy.expand(sympy.im(unit) + I * sympy.re(unit))


def _pick_gaussian(rng, span, *, distinct_parts=True, nonzero_parts=True):
    for _ in attempts(500, "a Gaussian integer"):
        a, b = rng.randint(-span, span), rng.randint(-span, span)
        if nonzero_parts and (a == 0 or b == 0):
            continue
        if distinct_parts and abs(a) == abs(b):
            continue
        return a, b
    raise AssertionError  # pragma: no cover


# -- easy -----------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "On an Argand diagram, the point P represents z = {z}. P is rotated {ang} about the origin to give the point Q. "
    "Which complex number does Q represent?",
    "Find the complex number represented by the image of the point {z} after a rotation through {ang} about the origin.",
    "The complex number {z} is shown as a point on the complex plane. The point is turned through {ang} about the "
    "origin. What complex number does the new point show?",
)


def _easy(rng):
    a, b = _pick_gaussian(rng, 9)
    deg = rng.choice([90, 90, -90, -90, 180])
    z = gaussian(a, b)
    w = _rotate(z, 0, deg)
    _verify_rotation(z, 0, deg, w)
    opposite = _rotate(z, 0, -deg)
    candidates = []
    if deg != 180:
        candidates.append(cdist(opposite, "rotation-wrong-direction"))
        candidates.append(cdist(gaussian(b, a), "rotation-swaps-real-and-imaginary"))
        candidates.append(cdist(-z, "rotation-quarter-turn-as-half-turn"))
    else:
        candidates.append(cdist(sympy.conjugate(z), "rotation-half-turn-reflects"))
        candidates.append(cdist(gaussian(-a, b), "rotation-half-turn-negates-one-part"))
        candidates.append(cdist(I * z, "rotation-half-turn-as-quarter-turn"))
        candidates.append(cdist(gaussian(b, a), "rotation-swaps-real-and-imaginary"))
    answer = cx_text(w)
    answer_value = cx_value(w)
    working = [
        f"A rotation through {_PHRASES[deg][0]} about the origin multiplies z by {cx_text(cis(deg))}.",
        f"{cx_text(z)} × ({cx_text(cis(deg))}) = {answer}.",
    ]
    return dict(
        text=phrase(rng, _EASY_STEMS, z=cx_text(z), ang=_angle_words(rng, deg)),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


# -- medium ---------------------------------------------------------------------------------------------------

_CENTRE_STEMS = (
    "The point P represents z = {z}. P is rotated {ang} about the point C, which represents {c}, to give the point Q. "
    "Find the complex number represented by Q.",
    "Find the image of the point {z} under a rotation through {ang} about the point {c} of the complex plane.",
    "On an Argand diagram, A is the point {z} and C is the point {c}. A is turned through {ang} about C to give the point A'. "
    "State the complex number that A' represents.",
)
_ORIGIN_STEMS = (
    "The complex number {z} is rotated through {ang} about the origin. Write the image in exact form.",
    "Find, in exact form, the complex number represented by the image of {z} after a rotation of {ang} about the origin.",
    "A point representing {z} is rotated about the origin through {ang}. What complex number does the image represent? "
    "Leave surds in exact form.",
)


def _medium_centre(rng):
    deg = rng.choice([90, -90, 180, 90, -90])
    for _ in attempts(500, "a point and a centre"):
        a, b = rng.randint(-8, 8), rng.randint(-8, 8)
        ca, cb = rng.randint(-5, 5), rng.randint(-5, 5)
        if (a, b) == (ca, cb) or (ca, cb) == (0, 0) or abs(a - ca) == abs(b - cb) or a - ca == 0 or b - cb == 0:
            continue
        break
    z, c = gaussian(a, b), gaussian(ca, cb)
    w = _rotate(z, c, deg)
    _verify_rotation(z, c, deg, w)
    about_origin = _rotate(z, 0, deg)
    candidates = [
        cdist(about_origin, "rotation-centre-ignored"),
        cdist(c + (z - c) * cis(-deg), "rotation-wrong-direction"),
        cdist((z - c) * cis(deg), "rotation-centre-not-added-back"),
        cdist(z * cis(deg) + c, "rotation-centre-added-to-rotated-point"),
    ]
    if deg == 180:
        candidates.append(cdist(c + sympy.conjugate(z - c), "rotation-half-turn-reflects"))
    answer = cx_text(w)
    answer_value = cx_value(w)
    working = [
        f"Move the centre to the origin: z - c = {cx_text(z - c)}.",
        f"Rotate: ({cx_text(z - c)}) × ({cx_text(cis(deg))}) = {cx_text((z - c) * cis(deg))}.",
        f"Move back: add c = {cx_text(c)} to get {answer}.",
    ]
    return dict(
        text=phrase(rng, _CENTRE_STEMS, z=cx_text(z), c=cx_text(c), ang=_angle_words(rng, deg)),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


def _medium_surd(rng):
    deg = rng.choice([60, -60, 30, -30, 45, -45, 120, 60, 45])
    if abs(deg) == 45:
        a, b = _pick_gaussian(rng, 7)
    else:
        for _ in attempts(500, "an even Gaussian integer"):
            a, b = 2 * rng.randint(-4, 4), 2 * rng.randint(-4, 4)
            if a != 0 and b != 0 and abs(a) != abs(b):
                break
    z = gaussian(a, b)
    w = _rotate(z, 0, deg)
    _verify_rotation(z, 0, deg, w)
    candidates = [
        cdist(z * cis(-deg), "rotation-wrong-direction"),
        cdist(z * _swapped(deg), "rotation-sine-and-cosine-swapped"),
        cdist(z * sympy.expand(sympy.re(cis(deg)) + sympy.im(cis(deg))), "rotation-multiplies-by-cos-plus-sin"),
        cdist(z + cis(deg), "rotation-adds-instead-of-multiplies"),
    ]
    answer = cx_text(w)
    answer_value = cx_value(w)
    working = [
        f"Multiply by cos {abs(deg)}° {'+' if deg > 0 else '-'} i sin {abs(deg)}° = {cx_text(cis(deg))}.",
        f"({cx_text(z)}) × ({cx_text(cis(deg))}) = {answer}.",
    ]
    return dict(
        text=phrase(rng, _ORIGIN_STEMS, z=cx_text(z), ang=_angle_words(rng, deg)),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


def _medium(rng):
    return rng.choice([_medium_centre, _medium_surd, _medium_centre, _medium_surd])(rng)


# -- hard -----------------------------------------------------------------------------------------------------

_SQUARE_STEMS = (
    "ABCD is a square whose vertices are labelled in {order} order on an Argand diagram. A represents {a} and B "
    "represents {b}. Find the complex number that {target} represents.",
    "On the complex plane, the points A = {a} and B = {b} are adjacent vertices of a square ABCD, with the letters "
    "running {order} around it. What complex number is vertex {target}?",
    "Two adjacent vertices of the square ABCD are A({a}) and B({b}), and A, B, C, D follow one another {order}. "
    "Give vertex {target} as a complex number.",
)
_CENTRE_VERTEX_STEMS = (
    "A square has its centre at {c} on the complex plane and one vertex at {a}. Going {order} around the square, the "
    "next vertex after {a} is B. Find the complex number that B represents.",
    "The point {a} is a vertex of a square whose centre is {c}. Find the vertex that comes next when the square is "
    "traversed {order} starting from {a}.",
    "On an Argand diagram a square is centred at {c} and has a vertex at {a}. State the complex number of the "
    "neighbouring vertex reached by turning {order} about the centre.",
)
_TRIANGLE_STEMS = (
    "ABC is an equilateral triangle with its vertices labelled {order} on an Argand diagram. A represents {a} and B "
    "represents {b}. Find the complex number that C represents.",
    "The points {a} and {b} are two vertices A and B of an equilateral triangle ABC, and A, B, C run {order}. "
    "Write the third vertex C as a complex number.",
    "A and B are the points {a} and {b} of the complex plane. Find vertex C of the equilateral triangle ABC, given "
    "that the letters A, B, C go {order} around it.",
)


def _order_phrase(rng, anticlockwise):
    return rng.choice(["anticlockwise", "counter-clockwise"]) if anticlockwise else "clockwise"


def _square_data(rng):
    for _ in attempts(500, "two adjacent vertices"):
        a1, a2 = rng.randint(-6, 6), rng.randint(-6, 6)
        p, q = rng.randint(-5, 5), rng.randint(-5, 5)
        if p == 0 or q == 0 or abs(p) == abs(q):
            continue
        return gaussian(a1, a2), gaussian(a1 + p, a2 + q)
    raise AssertionError  # pragma: no cover


def _hard_square(rng):
    a, b = _square_data(rng)
    anti = rng.random() < 0.65
    turn = I if anti else -I
    u = b - a
    target = rng.choice(["C", "D"])
    c_pt, d_pt = sympy.expand(b + turn * u), sympy.expand(a + turn * u)
    answer_pt = c_pt if target == "C" else d_pt
    # second route: the four points close up into a square (equal sides, equal diagonals)
    pts = [complex(p) for p in (a, b, c_pt, d_pt)]
    sides = [abs(pts[(k + 1) % 4] - pts[k]) for k in range(4)]
    check(max(sides) - min(sides) < 1e-9 and abs(abs(pts[2] - pts[0]) - abs(pts[3] - pts[1])) < 1e-9, "not a square")
    other = d_pt if target == "C" else c_pt
    candidates = [
        cdist(sympy.expand(b - turn * u) if target == "C" else sympy.expand(a - turn * u), "square-wrong-orientation"),
        cdist(other, "square-vertices-swapped"),
        cdist(sympy.expand(turn * b) if target == "C" else sympy.expand(turn * a), "square-rotates-position-not-side"),
        cdist(sympy.expand(b + u) if target == "C" else sympy.expand(a - u), "square-extends-the-side"),
    ]
    answer = cx_text(answer_pt)
    answer_value = cx_value(answer_pt)
    working = [
        f"The side AB is B - A = {cx_text(u)}.",
        f"Going {'anticlockwise' if anti else 'clockwise'}, the next side is AB turned through 90°: {cx_text(turn * u)}.",
        f"{'C = B' if target == 'C' else 'D = A'} {plus(cx_text(turn * u))} = {answer}.",
    ]
    return dict(
        text=phrase(rng, _SQUARE_STEMS, a=cx_text(a), b=cx_text(b), order=_order_phrase(rng, anti), target=target),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


def _hard_centre(rng):
    for _ in attempts(500, "a centre and a vertex"):
        ca, cb = rng.randint(-5, 5), rng.randint(-5, 5)
        p, q = rng.randint(-5, 5), rng.randint(-5, 5)
        if p == 0 or q == 0 or abs(p) == abs(q) or (ca, cb) == (0, 0):
            continue
        break
    c, a = gaussian(ca, cb), gaussian(ca + p, cb + q)
    anti = rng.random() < 0.6
    turn = I if anti else -I
    b = sympy.expand(c + turn * (a - c))
    _verify_rotation(a, c, 90 if anti else -90, b)
    candidates = [
        cdist(sympy.expand(c - turn * (a - c)), "square-wrong-orientation"),
        cdist(sympy.expand(turn * (a - c)), "rotation-centre-not-added-back"),
        cdist(sympy.expand(2 * c - a), "square-opposite-vertex"),
        cdist(sympy.expand(c + turn * a), "square-rotates-position-not-side"),
    ]
    answer = cx_text(b)
    answer_value = cx_value(b)
    working = [
        f"The vector from the centre to A is A - c = {cx_text(a - c)}.",
        f"The next vertex {'anticlockwise' if anti else 'clockwise'} is that vector turned through 90°: {cx_text(turn * (a - c))}.",
        f"Add the centre back: B = {answer}.",
    ]
    return dict(
        text=phrase(rng, _CENTRE_VERTEX_STEMS, c=cx_text(c), a=cx_text(a), order=_order_phrase(rng, anti)),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


def _hard_triangle(rng):
    for _ in attempts(500, "an equilateral triangle base"):
        a1, a2 = rng.randint(-5, 5), rng.randint(-5, 5)
        p, q = rng.randint(-3, 3), rng.randint(-3, 3)
        if p == 0 or q == 0 or abs(p) == abs(q):
            continue
        break
    a = gaussian(a1, a2)
    b = a + 2 * gaussian(p, q)
    anti = rng.random() < 0.6
    deg = 60 if anti else -60
    c = _rotate(b, a, deg)
    _verify_rotation(b, a, deg, c)
    pts = [complex(x) for x in (a, b, c)]
    sides = [abs(pts[(k + 1) % 3] - pts[k]) for k in range(3)]
    check(max(sides) - min(sides) < 1e-9, "not equilateral")
    candidates = [
        cdist(_rotate(b, a, -deg), "triangle-wrong-orientation"),
        cdist(sympy.expand(b * cis(deg)), "rotation-centre-ignored"),
        cdist(sympy.expand(a + (b - a) * _swapped(deg)), "rotation-sine-and-cosine-swapped"),
        cdist(sympy.expand((a + b) / 2 + (I if anti else -I) * (b - a)), "triangle-height-equals-side"),
    ]
    answer = cx_text(c)
    answer_value = cx_value(c)
    working = [
        f"The side AB is B - A = {cx_text(b - a)}.",
        f"Turn AB through 60° {'anticlockwise' if anti else 'clockwise'} about A: multiply by {cx_text(cis(deg))}.",
        f"C = A + ({cx_text(b - a)})({cx_text(cis(deg))}) = {answer}.",
    ]
    return dict(
        text=phrase(rng, _TRIANGLE_STEMS, a=cx_text(a), b=cx_text(b), order=_order_phrase(rng, anti)),
        answer_kind="expression",
        answer=answer_value,
        answer_text=answer,
        working=working,
        distractors=need(answer, candidates, 4),
    )


def _hard(rng):
    return rng.choice([_hard_square, _hard_square, _hard_centre, _hard_triangle])(rng)


@generator(
    "w6b_complex_rotation",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6b_complex_rotation(rng, tier="easy"):
    """Rotations in the complex plane and the vertices of squares and equilateral triangles; see the module docstring."""
    if tier == "easy":
        return retry(rng, _easy)
    if tier == "medium":
        return retry(rng, _medium)
    check(tier == "hard", f"unknown tier {tier!r}")
    return retry(rng, _hard)


# -- exam -----------------------------------------------------------------------------------------------------


def _num(value, tag):
    return dist(str(value), tag, value)


_EXAM_STEMS = (
    "ABCD is a square whose vertices are labelled {order} on an Argand diagram. A represents {a} and B represents {b}.",
    "On the complex plane, the square ABCD has its vertices in {order} order. The adjacent vertices A and B are the points {a} and {b}.",
    "A and B are adjacent vertices of a square ABCD, and A, B, C, D run {order} around it. A is the point {a} and B is the point {b}.",
)

_PYTHAGOREAN = ((3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (8, 15), (15, 8))


@generator("exd_x_w6b_complex_rotation", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w6b_complex_rotation(rng, tier="exam"):
    """Exam chain: the side, its length, the area (following through) and a missing vertex of a square ABCD."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry(rng, _exam)


def _exam(rng):
    p, q = rng.choice(_PYTHAGOREAN)
    p, q = rng.choice([-1, 1]) * p, rng.choice([-1, 1]) * q
    if rng.random() < 0.5:
        p, q = q, p
    for _ in attempts(200, "an anchor vertex"):
        a1, a2 = rng.randint(-6, 6), rng.randint(-6, 6)
        if (a1, a2) != (0, 0):
            break
    a, u = gaussian(a1, a2), gaussian(p, q)
    b = a + u
    anti = rng.random() < 0.6
    turn = I if anti else -I
    c_pt, d_pt = sympy.expand(b + turn * u), sympy.expand(a + turn * u)
    side = int(math.hypot(p, q))
    check(side * side == p * p + q * q, "side is not a whole number")
    target = rng.choice(["C", "D"])
    vertex = c_pt if target == "C" else d_pt
    wrong_side = sympy.expand(b - turn * u) if target == "C" else sympy.expand(a - turn * u)
    order = "anticlockwise" if anti else "clockwise"

    ab_cands = [
        cdist(a - b, "side-vector-difference-reversed"),
        cdist(b + a, "side-vector-sum-for-difference"),
        cdist(gaussian(q, p), "side-vector-parts-swapped"),
    ]
    side_cands = [
        _num(abs(p) + abs(q), "side-length-adds-the-parts"),
        _num(abs(p * p + q * q), "side-length-forgets-square-root"),
        _num(max(abs(p), abs(q)), "side-length-takes-the-larger-part"),
    ]
    area_cands = [
        _num(4 * side, "square-area-as-perimeter"),
        _num(2 * side, "square-area-doubles-side"),
        _num(side * side // 2, "square-area-halved"),
        _num(abs(p) * abs(q), "square-area-from-the-parts"),
    ]
    vertex_cands = [
        cdist(wrong_side, "square-wrong-orientation"),
        cdist(d_pt if target == "C" else c_pt, "square-vertices-swapped"),
        cdist(sympy.expand(turn * (b if target == "C" else a)), "square-rotates-position-not-side"),
    ]
    return dict(
        text=phrase(rng, _EXAM_STEMS, order=order, a=cx_text(a), b=cx_text(b)),
        parts=[
            dict(
                prompt="Find the complex number that represents the side AB, that is, the vector from A to B.",
                answer_kind="expression",
                answer=cx_value(u),
                answer_text=cx_text(u),
                marks=1,
                working=[f"B - A = {cx_text(b)} - ({cx_text(a)}) = {cx_text(u)}."],
                distractors=need(cx_text(u), ab_cands, 3),
            ),
            dict(
                prompt="Find the length of the side AB.",
                answer_kind="number",
                answer=side,
                answer_text=str(side),
                marks=1,
                sensible_range=(0, None),
                working=[f"|AB| = √({abs(p)}² + {abs(q)}²) = {side}."],
                distractors=need(str(side), side_cands, 3),
            ),
            dict(
                prompt="Hence find the area of the square.",
                answer_kind="number",
                answer=side * side,
                answer_text=str(side * side),
                marks=1,
                follow_through="part_b**2",
                sensible_range=(0, None),
                working=[f"Area = {side}² = {side * side}."],
                distractors=need(str(side * side), area_cands, 3),
            ),
            dict(
                prompt=f"Find the complex number that represents the vertex {target}.",
                answer_kind="expression",
                answer=cx_value(vertex),
                answer_text=cx_text(vertex),
                marks=2,
                working=[
                    f"Turn AB through 90° {order}: {cx_text(turn * u)}.",
                    f"{'C = B' if target == 'C' else 'D = A'} {plus(cx_text(turn * u))} = {cx_text(vertex)}.",
                ],
                distractors=need(cx_text(vertex), vertex_cands, 3),
            ),
        ],
    )
