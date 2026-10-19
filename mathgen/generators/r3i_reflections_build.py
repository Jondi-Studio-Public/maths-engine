"""Generates `r3i_reflect_rotate` questions (easy, medium, hard) and the exam chain `r3i_exam_reflect_rotate`.

Skill ``reflections-compose-to-rotation`` (``matrices``, answer ``matrix``): the product of two reflection matrices is a
rotation through twice the angle between the mirror lines, and the product of three is a reflection. Every mirror passes
through the origin at an angle that is a multiple of 15° to the positive x-axis, so every entry is 0, ±1, ±1/2 or
±√3/2 (written √3/2).

* easy: the matrices of two reflections in the axes or the lines y = ±x are stated; the matrix of one reflection followed
  by the other. The pairs are chosen so that the order matters.
* medium: two mirror lines given only by their angles; the matrix of reflecting in the first and then in the second.
* hard: either three mirror lines (the result is a single reflection, whose matrix is asked for), or the matrix of the
  second mirror when the first mirror and the rotation produced by the pair are stated.
* exam (``r3i_exam_reflect_rotate``): two mirror lines; (a) the angle of the rotation they produce, (b) its matrix, (c) the
  number of times it must be applied to bring every point back (follows through from (a)), (d) the image of a point.

The key is built from the angle rule and checked a second way: the two reflections are applied one after the other to
the points (1, 0) and (0, 1) using complex numbers (the reflection in the line at angle θ sends z to e^(2iθ) times
the conjugate of z), and the columns of the matrix must be the images.
"""

from __future__ import annotations

import cmath
import functools
import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from ._r3i_kit import matrix_dist, matrix_fields, num_cands, num_fields, pick, pick_text, retry_short

PI = sympy.pi


@functools.lru_cache(maxsize=None)
def refl(deg: int) -> sympy.ImmutableMatrix:
    """Reflection in the line through the origin at `deg` degrees (a multiple of 15) to the positive x-axis."""
    t = PI * 2 * deg / 180
    return sympy.ImmutableMatrix([[sympy.cos(t), sympy.sin(t)], [sympy.sin(t), -sympy.cos(t)]])


@functools.lru_cache(maxsize=None)
def rot(deg: int) -> sympy.ImmutableMatrix:
    """Anticlockwise rotation about the origin through `deg` degrees (a multiple of 30)."""
    t = PI * deg / 180
    return sympy.ImmutableMatrix([[sympy.cos(t), -sympy.sin(t)], [sympy.sin(t), sympy.cos(t)]])


@functools.lru_cache(maxsize=None)
def _clean(value):
    """An exact entry in its simplest closed form (the entries come from a small set, so this is cached)."""
    return sympy.nsimplify(value)


def _rows(m: sympy.Matrix):
    return [[_clean(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def _apply(deg_list, z: complex) -> complex:
    """Reflect z in each mirror (angles in degrees) in turn."""
    for deg in deg_list:
        z = cmath.exp(2j * math.radians(deg)) * z.conjugate()
    return z


def _check_matrix(mat: sympy.Matrix, mirrors) -> None:
    """The columns of `mat` are the images of (1, 0) and (0, 1) under the mirrors applied in order."""
    for col, start in enumerate((1 + 0j, 0 + 1j)):
        image = _apply(mirrors, start)
        check(
            abs(complex(mat[0, col]) - image.real) < 1e-9 and abs(complex(mat[1, col]) - image.imag) < 1e-9,
            "the matrix differs from reflecting the points one mirror after the other",
        )


def _mod(deg: int, base: int) -> int:
    return deg % base


# the four standard mirrors and the three quarter-turn rotations of the easy tier: name, matrix, mirror angle
_MIRRORS = {
    "the x-axis": (sympy.Matrix([[1, 0], [0, -1]]), 0),
    "the y-axis": (sympy.Matrix([[-1, 0], [0, 1]]), 90),
    "the line y = x": (sympy.Matrix([[0, 1], [1, 0]]), 45),
    "the line y = -x": (sympy.Matrix([[0, -1], [-1, 0]]), 135),
}
_TURNS = {"a quarter turn (90°)": 90, "a half turn (180°)": 180, "a three-quarter turn (270°)": 270}


def _mtext(m: sympy.Matrix) -> str:
    return "[" + ", ".join("[" + ", ".join(str(m[i, j]) for j in range(m.cols)) + "]" for i in range(m.rows)) + "]"


def _angle_text(deg: int) -> str:
    return f"{deg}°"


# -- easy ---------------------------------------------------------------------------------------------------


def _easy(rng):
    shape = rng.choice(["two_mirrors", "mirror_then_turn", "turn_then_mirror"])
    if shape == "two_mirrors":
        names = list(_MIRRORS)
        for _ in attempts(200, "two reflections whose product depends on the order"):
            first, second = rng.sample(names, 2)
            if _MIRRORS[second][0] * _MIRRORS[first][0] != _MIRRORS[first][0] * _MIRRORS[second][0]:
                break
        (m1, d1), (m2, d2) = _MIRRORS[first], _MIRRORS[second]
        key, mirrors = m2 * m1, [d1, d2]
        given = (
            f"The matrix of the reflection in {first} is P = {_mtext(m1)} and the matrix of the reflection in {second} is "
            f"Q = {_mtext(m2)}."
        )
        ask = f"Find the matrix of the single transformation that reflects a point in {first} and then in {second}."
        reversed_product, left, right = m1 * m2, "Q", "P"
    else:
        mirror = rng.choice(list(_MIRRORS))
        turn = rng.choice(list(_TURNS))
        m, d = _MIRRORS[mirror]
        t = rot(_TURNS[turn])
        if shape == "mirror_then_turn":
            key, mirrors = t * m, None
            ask = f"Find the matrix of the single transformation that reflects a point in {mirror} and then rotates it through {turn[turn.index('(') + 1 : -1]} anticlockwise about the origin."
            reversed_product, left, right = m * t, "R", "P"
        else:
            key, mirrors = m * t, None
            ask = f"Find the matrix of the single transformation that rotates a point through {turn[turn.index('(') + 1 : -1]} anticlockwise about the origin and then reflects it in {mirror}."
            reversed_product, left, right = t * m, "P", "R"
        given = f"The matrix of the reflection in {mirror} is P = {_mtext(m)} and the matrix of the anticlockwise rotation through {turn[turn.index('(') + 1 : -1]} about the origin is R = {_mtext(t)}."
        m1, m2 = (m, t) if shape == "mirror_then_turn" else (t, m)
    if mirrors:
        _check_matrix(key, mirrors)
    fields = matrix_fields(_rows(key))
    cands = [
        matrix_dist(_rows(reversed_product), "r3i-matrix-product-order-reversed"),
        matrix_dist(_rows(m1 + m2), "r3i-matrix-added-not-multiplied"),
        matrix_dist(_rows(-key), "r3i-matrix-product-sign-slip"),
    ]
    product = m1.multiply_elementwise(m2)
    if any(v != 0 for v in product):
        cands.insert(2, matrix_dist(_rows(product), "r3i-matrix-entrywise-product"))
    return dict(
        text=f"{given} {ask}",
        working=[
            f"The transformation applied first has its matrix on the right, so the combined matrix is {left}{right}.",
            f"{left}{right} = {fields['answer_text']}.",
        ],
        distractors=pick_matrix(fields, cands),
        **fields,
    )


def pick_matrix(fields, cands):
    return pick_text(fields, cands)


# -- medium -------------------------------------------------------------------------------------------------


def _angles(rng, count: int):
    for _ in attempts(200, "distinct mirror angles"):
        values = [rng.randrange(0, 180, 15) for _ in range(count)]
        if all(values[i] != values[i + 1] for i in range(count - 1)):
            return values
    raise AssertionError  # pragma: no cover


def _medium(rng):
    a, b = _angles(rng, 2)
    phi = (2 * (b - a)) % 360
    key = rot(phi)
    _check_matrix(key, [a, b])
    check(key == refl(b) * refl(a), "the rotation differs from the product of the reflection matrices")
    fields = matrix_fields(_rows(key))
    cands = [
        matrix_dist(_rows(rot((2 * (a - b)) % 360)), "r3i-reflections-order-reversed"),
        matrix_dist(_rows(rot((b - a) % 360)), "r3i-reflections-angle-not-doubled") if (b - a) % 30 == 0 else None,
        matrix_dist(_rows(refl((b - a) % 180)), "r3i-reflections-product-taken-as-reflection"),
        matrix_dist(_rows(rot((2 * (a + b)) % 360)), "r3i-reflections-angles-added"),
    ]
    stem = phrase(
        rng,
        [
            "A point is reflected in the line through the origin at an angle of {a} to the positive x-axis, and then "
            "in the line through the origin at an angle of {b} to the positive x-axis (both angles measured "
            "anticlockwise). Find the matrix of the single transformation that does the same job.",
            "Two mirrors pass through the origin, the first at an angle of {a} and the second at an angle of {b} to "
            "the positive x-axis (both angles measured anticlockwise). A point is reflected in the first mirror and "
            "then in the second. Find the matrix of the single transformation that has the same effect.",
            "The reflection in the line through the origin at an angle of {a} to the positive x-axis is followed by "
            "the reflection in the line through the origin at an angle of {b} (both angles measured anticlockwise). "
            "Find the matrix of the one transformation equivalent to the pair.",
        ],
        a=_angle_text(a),
        b=_angle_text(b),
    )
    return dict(
        text=stem,
        working=[
            f"Two reflections in lines {abs(b - a)}° apart combine to a rotation through twice that angle, taken from the first line towards the second.",
            f"Here the rotation is through {phi}° anticlockwise, with matrix {fields['answer_text']}.",
        ],
        distractors=pick_matrix(fields, cands),
        **fields,
    )


# -- hard ---------------------------------------------------------------------------------------------------


def _hard_three(rng):
    for _ in attempts(500, "three mirrors with a result that differs from each of them"):
        a, b, c = _angles(rng, 3)
        net = (a - b + c) % 180
        if net not in (a, b, c) and (a - b + c) % 180 != 0:
            break
    key = refl(net)
    check(key == refl(c) * refl(b) * refl(a), "the reflection differs from the product of the three matrices")
    _check_matrix(key, [a, b, c])
    fields = matrix_fields(_rows(key))
    cands = [
        matrix_dist(_rows(refl((a + b - c) % 180)), "r3i-reflections-three-angles-signs-wrong"),
        matrix_dist(_rows(refl((c - b - a) % 180)), "r3i-reflections-three-angles-signs-wrong"),
        matrix_dist(_rows(rot((2 * (c - b + a)) % 360)), "r3i-reflections-three-taken-as-rotation"),
        matrix_dist(_rows(refl((2 * (c - b + a)) % 180)), "r3i-reflections-angle-doubled-twice"),
    ]
    stem = phrase(
        rng,
        [
            "A point is reflected in three lines through the origin, one after the other. The lines make angles of "
            "{a}, {b} and {c} with the positive x-axis (anticlockwise), and the reflections are applied in that "
            "order. The result is a single reflection. Find its matrix.",
            "A point is reflected in turn in three lines through the origin, at angles of {a}, {b} and {c} to the "
            "positive x-axis (anticlockwise). The combined effect is a single reflection. Find the matrix of that "
            "reflection.",
        ],
        a=_angle_text(a),
        b=_angle_text(b),
        c=_angle_text(c),
    )
    return dict(
        text=stem,
        working=[
            "Two of the reflections combine into a rotation; reflecting once more gives a reflection.",
            f"The mirror of the combined reflection makes an angle of {a} - {b} + {c} = {a - b + c}° (that is, {net}° after removing multiples of 180°) with the x-axis.",
            f"Its matrix is {fields['answer_text']}.",
        ],
        distractors=pick_matrix(fields, cands),
        **fields,
    )


def _hard_second_mirror(rng):
    for _ in attempts(200, "a rotation that is not a half-turn multiple"):
        a = rng.randrange(0, 180, 15)
        phi = rng.choice([60, 90, 120, 150, 210, 240, 270, 300, 330, 30])
        b = (a + phi // 2) % 180
        if b != a:
            break
    check(rot(phi) == refl(b) * refl(a), "the rotation differs from the product")
    _check_matrix(refl(b), [b])
    fields = matrix_fields(_rows(refl(b)))
    cands = [
        matrix_dist(_rows(refl((a - phi // 2) % 180)), "r3i-reflections-second-mirror-on-wrong-side"),
        matrix_dist(_rows(refl((a + phi) % 180)), "r3i-reflections-angle-not-halved"),
        matrix_dist(_rows(refl((phi // 2) % 180)), "r3i-reflections-first-mirror-forgotten"),
        matrix_dist(_rows(rot(phi)), "r3i-reflections-gave-the-rotation"),
    ]
    stem = (
        f"A point is reflected in a line L1 through the origin at an angle of {_angle_text(a)} to the positive x-axis, and then in a "
        f"second line L2 through the origin. Together the two reflections are equivalent to an anticlockwise rotation about the origin "
        f"through {_angle_text(phi)}. Find the matrix of the reflection in L2 (the line makes an angle between 0° and 180° with the "
        "positive x-axis)."
    )
    return dict(
        text=stem,
        working=[
            f"A rotation through {phi}° comes from mirrors {phi // 2}° apart, with the second mirror {phi // 2}° anticlockwise of the first.",
            f"So L2 is at {a} + {phi // 2} = {a + phi // 2}° (that is, {b}°), and its matrix is {fields['answer_text']}.",
        ],
        distractors=pick_matrix(fields, cands),
        **fields,
    )


def _hard(rng):
    return _hard_three(rng) if rng.random() < 0.5 else _hard_second_mirror(rng)


@generator(
    "r3i_reflect_rotate",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_reflect_rotate(rng, tier="easy"):
    """Composing reflection matrices; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------


@generator("r3i_exam_reflect_rotate", level=14, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_reflect_rotate(rng, tier="exam"):
    """Exam chain: rotation angle of two reflections, its matrix, the order of the rotation, the image of a point."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "mirrors whose rotation angle divides 360"):
        a, b = _angles(rng, 2)
        phi = (2 * (b - a)) % 360
        if 360 % phi == 0:
            break
    n = 360 // phi
    px, py = rng.choice([(2, 0), (4, 0), (0, 2), (0, 4), (6, 0), (0, 6), (2, 2), (4, 2), (2, 4), (-2, 4), (4, -2)])
    mat = rot(phi)
    check(mat == refl(b) * refl(a), "the rotation differs from the product")
    _check_matrix(mat, [a, b])
    image = mat * sympy.Matrix([px, py])
    check(
        abs(complex(image[0]) - _apply([a, b], complex(px, py)).real) < 1e-9,
        "the image differs from reflecting the point",
    )
    check(
        (mat**n) == sympy.eye(2) and all((mat**k) != sympy.eye(2) for k in range(1, n)),
        "the order of the rotation is wrong",
    )
    f_a, f_c = num_fields(phi), num_fields(n)
    f_b = matrix_fields(_rows(mat))
    f_d = matrix_fields(_rows(image.T))
    return dict(
        text=phrase(
            rng,
            [
                "A point is reflected in the line through the origin at an angle of {a} to the positive x-axis, and "
                "then in the line through the origin at an angle of {b} to the positive x-axis (both anticlockwise). "
                "The two reflections are together equivalent to a rotation about the origin.",
                "Two mirrors pass through the origin, at angles of {a} and {b} to the positive x-axis (both "
                "anticlockwise). Reflecting a point in the first mirror and then in the second has the same effect "
                "as a rotation about the origin.",
                "The reflection in the line through the origin at an angle of {a} to the positive x-axis is followed "
                "by the reflection in the line at an angle of {b} (both anticlockwise). Together they are equivalent "
                "to a rotation about the origin.",
            ],
            a=_angle_text(a),
            b=_angle_text(b),
        ),
        parts=[
            dict(
                prompt="Find the angle of this rotation, in degrees anticlockwise, between 0 and 360.",
                marks=2,
                sensible_range=(0, 360),
                working=[
                    f"The mirrors are {abs(b - a)}° apart, so the rotation is through 2 × ({b} - {a}) = {2 * (b - a)}°, that is, {phi}°."
                ],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            ((b - a) % 360, "r3i-reflections-angle-not-doubled"),
                            ((2 * (a - b)) % 360, "r3i-reflections-order-reversed"),
                            ((2 * (a + b)) % 360, "r3i-reflections-angles-added"),
                            (abs(2 * (b - a)), "r3i-reflections-angle-not-reduced"),
                        ],
                        phi,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="Write down the matrix of this rotation.",
                marks=2,
                working=[
                    f"The matrix of an anticlockwise rotation through {phi}° is [[cos {phi}°, -sin {phi}°], [sin {phi}°, cos {phi}°]]."
                ],
                distractors=pick_matrix(
                    f_b,
                    [
                        matrix_dist(_rows(rot((-phi) % 360)), "r3i-reflections-rotation-direction-reversed"),
                        matrix_dist(_rows(refl((phi // 2) % 180)), "r3i-reflections-product-taken-as-reflection"),
                        matrix_dist(_rows(mat.T * -1), "r3i-matrix-product-sign-slip"),
                    ],
                ),
                **f_b,
            ),
            dict(
                prompt="What is the smallest number of times this rotation must be applied to bring every point back to where it started?",
                marks=1,
                follow_through="360/part_a",
                sensible_range=(0, 400),
                working=[
                    f"Every {phi}° turn adds up to a whole number of full turns after {n} applications, since {n} × {phi} = {n * phi} = {n * phi // 360} × 360."
                ],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (phi, "r3i-reflections-order-taken-as-angle"),
                            (n * 2, "r3i-reflections-order-doubled"),
                            (n + 1, "r3i-reflections-order-off-by-one"),
                            (max(n - 1, 1), "r3i-reflections-order-off-by-one"),
                        ],
                        n,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt=f"Find the image of the point ({px}, {py}) under the two reflections (reflect in the first line, then in the second). Give the coordinates as an ordered pair.",
                marks=3,
                working=[f"Multiply the rotation matrix by the column ({px}, {py}) to get {f_d['answer_text']}."],
                distractors=pick_matrix(
                    f_d,
                    [
                        matrix_dist(
                            _rows((rot((-phi) % 360) * sympy.Matrix([px, py])).T),
                            "r3i-reflections-rotation-direction-reversed",
                        ),
                        *(
                            [
                                matrix_dist(
                                    _rows((rot((b - a) % 360) * sympy.Matrix([px, py])).T),
                                    "r3i-reflections-angle-not-doubled",
                                )
                            ]
                            if (b - a) % 30 == 0
                            else []
                        ),
                        matrix_dist(_rows((refl(b) * sympy.Matrix([px, py])).T), "r3i-reflections-second-mirror-only"),
                    ],
                ),
                **f_d,
            ),
        ],
    )
