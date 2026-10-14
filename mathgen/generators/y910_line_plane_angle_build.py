"""The angle between a line and a plane, in a cuboid or a pyramid (AC: Year 10A, three-dimensional trigonometry).

The learner is given the dimensions of a solid in words (nothing is drawn) and a named line and plane, and types
the angle between them in degrees, to one decimal place. The skill is choosing the right right-angled triangle:
the angle between a line and a plane is the angle between the line and its own shadow (projection) on the plane,
so the triangle is "the line, its shadow, and the perpendicular dropped from the line's far end to the plane".

Ladder (reasoning escalation, not size):

* easy: a face diagonal and the base. The shadow of a face diagonal on the base is a base edge, so the triangle is
  already a face of the box.
* medium: a space diagonal and the base (the shadow is the base diagonal, so Pythagoras comes first), or a slant
  edge of a square-based pyramid and its base (the shadow is half the base diagonal).
* hard: a space diagonal and a vertical side face (the plane is not horizontal, the shadow is a face diagonal), or a
  slant edge of a rectangular pyramid whose slant-edge LENGTH is given, not its height, so the shadow is half a
  rectangle's diagonal and the angle comes from a cosine.
* exam: a four-part chain on one cuboid (base diagonal, space diagonal, angle with the base, angle with a side face).

Every key is computed by right-triangle trigonometry and checked a second way, with the vector formula
``sin(theta) = |d . n| / (|d| |n|)`` for a direction ``d`` and a plane normal ``n``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

TOPIC = "y910_line_plane_angle"
TIERS = ("easy", "medium", "hard")
DP = 1

# base vertices in order A B C D (anticlockwise from above); E F G H are the vertices directly above them
_BASE = "ABCD"
_TOP = "EFGH"
_NOUNS = (
    "A storage crate",
    "A rectangular fish tank",
    "A packing box",
    "A concrete block",
    "A shipping container model",
    "A rectangular water tank",
)


def _deg(x: float) -> float:
    return math.degrees(x)


def _shown(value: float) -> Fraction:
    """The value to one decimal place, as an exact fraction."""
    return round_half_up(Fraction(value), DP)


def _text(value: Fraction) -> str:
    return stated_precision_string(value, DP)


def _plane_angle_by_vectors(direction, normal) -> float:
    """Second route: the angle between a line and a plane from sin(theta) = |d . n| / (|d| |n|)."""
    dot = abs(sum(d * n for d, n in zip(direction, normal)))
    norm = math.sqrt(sum(d * d for d in direction)) * math.sqrt(sum(n * n for n in normal))
    return _deg(math.asin(dot / norm))


def _tagged(key: Fraction, wrong: list[tuple[float, str]], n: int = 3) -> list[Distractor]:
    """Tagged wrong options at the key's own number of decimal places, each clear of the grader's tolerance."""
    margin = 2 * stated_precision_tolerance(DP)
    by_text: dict[str, tuple[Fraction, str]] = {}
    for value, note in wrong:
        if not (0 < value < 90):
            continue
        shown = _shown(value)
        by_text.setdefault(_text(shown), (shown, note))
    texts = pick_distractors(_text(key), list(by_text), n, tolerance=margin)
    return [Distractor(t, by_text[t][1], numeric_answer(by_text[t][0])) for t in texts]


def _part_fields(prompt, key: Fraction, wrong, marks, working, *, follow_through=None, low=0, high=90):
    out = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=_text(key),
        answer_latex=latex_answer(_text(key)),
        marks=marks,
        sensible_range=(low, high),
        working=working,
        distractors=_tagged(key, wrong),
    )
    if follow_through:
        out["follow_through"] = follow_through
    return out


# ---------------------------------------------------------------------------------------------------- cuboids


def _cuboid_intro(rng, a, b, h) -> str:
    noun = rng.choice(_NOUNS)
    return (
        f"{noun} is the cuboid ABCDEFGH. Its base ABCD is horizontal with AB = {a} cm and BC = {b} cm, and the "
        f"vertical edges AE, BF, CG and DH are each {h} cm long."
    )


def _cuboid_dims(rng):
    def draw():
        return rng.randint(4, 15), rng.randint(3, 12), rng.randint(3, 14)

    for _ in attempts(500, "cuboid dimensions"):
        a, b, h = draw()
        if a != b and len({a, b, h}) == 3:
            return a, b, h
    raise AssertionError  # pragma: no cover


# vertex coordinates for a cuboid with AB = a (x), BC = b (y), AE = h (z)
def _coords(a, b, h):
    return {
        "A": (0, 0, 0),
        "B": (a, 0, 0),
        "C": (a, b, 0),
        "D": (0, b, 0),
        "E": (0, 0, h),
        "F": (a, 0, h),
        "G": (a, b, h),
        "H": (0, b, h),
    }


def _vec(p, q):
    return tuple(qi - pi for pi, qi in zip(p, q))


# vertical faces: name, a vertex pair identifying it (any two opposite corners), the axis of its normal and
# the face's x or y position
_SIDE_FACES = {
    "ABFE": ("y", 0),
    "BCGF": ("x", "a"),
    "CDHG": ("y", "b"),
    "DAEH": ("x", 0),
}


def _easy_case(rng):
    """A face diagonal and the base: the shadow is a base edge."""
    for _ in attempts(500, "an easy cuboid"):
        a, b, h = _cuboid_dims(rng)
        # face diagonals from a base vertex up to the far top vertex of a vertical face
        start, end, edge_len, other_len = rng.choice(
            [
                ("A", "F", a, b),
                ("B", "E", a, b),
                ("D", "G", a, b),
                ("C", "H", a, b),
                ("A", "H", b, a),
                ("D", "E", b, a),
                ("B", "G", b, a),
                ("C", "F", b, a),
            ]
        )
        theta = math.atan2(h, edge_len)
        deg = _deg(theta)
        if 15 <= deg <= 75 and rounds_unambiguously(deg, DP):
            break
    pts = _coords(a, b, h)
    d = _vec(pts[start], pts[end])
    vec = _plane_angle_by_vectors(d, (0, 0, 1))
    check(abs(vec - deg) < 1e-9, "the vector route disagrees with the triangle route")
    key = _shown(deg)
    # name the shadow of the diagonal on the base: it runs from `start` to the base vertex under `end`
    under = _BASE[_TOP.index(end)]
    shadow = f"{start}{under}"
    wrong = [
        (_deg(math.atan2(h, other_len)), "used-the-other-base-edge-as-the-shadow"),
        (90 - deg, "tangent-ratio-upside-down"),
        (_deg(math.atan2(h, math.hypot(a, b))), "used-the-base-diagonal-as-the-shadow"),
        (_deg(math.atan2(edge_len, h)), "tangent-ratio-upside-down"),
        (_deg(math.atan2(h, edge_len + other_len)), "added-the-base-edges-for-the-shadow"),
    ]
    text = (
        f"{_cuboid_intro(rng, a, b, h)} The face diagonal {start}{end} is drawn. Find the angle between "
        f"{start}{end} and the base ABCD, correct to one decimal place."
    )
    working = [
        f"The shadow of {start}{end} on the base is {shadow}, which is {edge_len} cm.",
        f"The perpendicular from {end} to the base is {end}{under} = {h} cm.",
        f"tan(angle) = {h} / {edge_len}, so the angle is {_text(key)} degrees.",
    ]
    return text, key, wrong, working


def _space_case(rng, plane: str):
    diagonals = [("A", "G"), ("B", "H"), ("C", "E"), ("D", "F")]
    for _ in attempts(500, "a space-diagonal cuboid"):
        a, b, h = _cuboid_dims(rng)
        start, end = rng.choice(diagonals)
        pts = _coords(a, b, h)
        d = _vec(pts[start], pts[end])
        if plane == "base":
            normal_axis, perp, para = 2, h, math.hypot(a, b)
            face = None
        else:
            # a vertical face that contains one end of the diagonal
            options = [f for f in _SIDE_FACES if start in f or end in f]
            face = rng.choice(options)
            axis, _ = _SIDE_FACES[face]
            normal_axis = 0 if axis == "x" else 1
            perp = a if axis == "x" else b
            para = math.hypot(*(v for i, v in enumerate((a, b, h)) if i != normal_axis))
        deg = _deg(math.atan2(perp, para))
        if 12 <= deg <= 75 and rounds_unambiguously(deg, DP):
            break
    normal = tuple(1 if i == normal_axis else 0 for i in range(3))
    check(abs(_plane_angle_by_vectors(d, normal) - deg) < 1e-9, "the vector route disagrees with the triangle route")
    key = _shown(deg)
    diag = f"{start}{end}"
    if plane == "base":
        plane_name, plane_words = "ABCD", "the base ABCD"
        shadow = f"{start}{_BASE[_TOP.index(end)]}"
        wrong = [
            (_deg(math.atan2(h, a)), "used-a-base-edge-as-the-shadow"),
            (_deg(math.atan2(h, b)), "used-a-base-edge-as-the-shadow"),
            (90 - deg, "tangent-ratio-upside-down"),
            (_deg(math.atan2(h, a + b)), "added-the-base-edges-for-the-shadow"),
            (_deg(math.atan2(h, a * a + b * b)), "forgot-the-square-root-in-the-shadow"),
            (
                _deg(math.asin(min(1.0, h / math.hypot(a, b)))) if h < math.hypot(a, b) else 0,
                "base-diagonal-as-hypotenuse",
            ),
        ]
        working = [
            f"The shadow of {diag} on the base is the base diagonal {shadow} = sqrt({a}^2 + {b}^2) = {math.hypot(a, b):.2f} cm.",
            f"The perpendicular from the top end of {diag} to the base is {h} cm.",
            f"tan(angle) = {h} / {math.hypot(a, b):.2f}, so the angle is {_text(key)} degrees.",
        ]
    else:
        plane_name, plane_words = face, f"the vertical face {face}"
        base_deg = _deg(math.atan2(h, math.hypot(a, b)))
        wrong = [
            (base_deg, "measured-to-the-base-instead-of-the-face"),
            (90 - deg, "tangent-ratio-upside-down"),
            (_deg(math.atan2(perp, perp + para)), "added-instead-of-using-pythagoras"),
            (_deg(math.atan2(perp, max(a, b, h))), "used-one-edge-as-the-shadow"),
            (
                _deg(math.atan2(perp, min(x for i, x in enumerate((a, b, h)) if i != normal_axis))),
                "used-one-edge-as-the-shadow",
            ),
        ]
        working = [
            f"The shadow of {diag} on {plane_words} is a face diagonal of length {para:.2f} cm.",
            f"The perpendicular from the far end of {diag} to the face is the edge of length {perp} cm.",
            f"tan(angle) = {perp} / {para:.2f}, so the angle is {_text(key)} degrees.",
        ]
    text = (
        f"{_cuboid_intro(rng, a, b, h)} The space diagonal {diag} is drawn. Find the angle between {diag} and "
        f"{plane_words}, correct to one decimal place."
    )
    return text, key, wrong, working


# ---------------------------------------------------------------------------------------------------- pyramids


def _pyramid_height_case(rng, rectangle: bool):
    """A pyramid VABCD, apex V above the centre of the base. Angle between VA and the base from the height."""
    for _ in attempts(500, "a pyramid"):
        a = rng.randint(6, 20)
        b = rng.randint(5, 18) if rectangle else a
        if rectangle and a == b:
            continue
        h = rng.randint(4, 16)
        half = math.hypot(a, b) / 2
        deg = _deg(math.atan2(h, half))
        if 15 <= deg <= 75 and rounds_unambiguously(deg, DP):
            break
    d = (a / 2, b / 2, -h)  # from the apex down to base vertex A when the centre is the origin
    check(abs(_plane_angle_by_vectors(d, (0, 0, 1)) - deg) < 1e-9, "the vector route disagrees")
    key = _shown(deg)
    base = (
        f"a square base ABCD of side {a} m"
        if not rectangle
        else f"a rectangular base ABCD with AB = {a} m and BC = {b} m"
    )
    full = math.hypot(a, b)
    wrong = [
        (_deg(math.atan2(h, full)), "used-the-whole-diagonal-instead-of-half"),
        (_deg(math.atan2(h, a / 2)), "used-half-a-base-edge-as-the-shadow"),
        (90 - deg, "tangent-ratio-upside-down"),
        (_deg(math.atan2(h, a)), "used-a-base-edge-as-the-shadow"),
        (_deg(math.atan2(h, b / 2)), "used-half-a-base-edge-as-the-shadow"),
    ]
    text = (
        f"A pyramid-shaped tent VABCD stands on {base}. The apex V is {h} m directly above the centre of the base. "
        f"Find the angle between the sloping edge VA and the base, correct to one decimal place."
    )
    working = [
        f"The shadow of VA on the base runs from A to the centre: half the base diagonal, {half:.2f} m.",
        f"tan(angle) = {h} / {half:.2f}, so the angle is {_text(key)} degrees.",
    ]
    return text, key, wrong, working


def _pyramid_edge_case(rng):
    """A rectangular pyramid where the slant edge LENGTH is given: the angle comes from a cosine."""
    for _ in attempts(1000, "a pyramid with a given edge"):
        a, b = rng.randint(6, 20), rng.randint(5, 18)
        if a == b:
            continue
        half = math.hypot(a, b) / 2
        e = rng.randint(int(half) + 2, int(half) + 14)
        if e <= half:
            continue
        deg = _deg(math.acos(half / e))
        if 20 <= deg <= 75 and rounds_unambiguously(deg, DP):
            break
    h = math.sqrt(e * e - half * half)
    d = (half, 0.0, -h)
    check(abs(_plane_angle_by_vectors(d, (0, 0, 1)) - deg) < 1e-6, "the vector route disagrees")
    key = _shown(deg)
    full = math.hypot(a, b)
    wrong = [
        (_deg(math.acos(min(1.0, full / e))) if full < e else 0, "used-the-whole-diagonal-instead-of-half"),
        (_deg(math.acos(a / e)) if a < e else 0, "used-a-base-edge-as-the-shadow"),
        (90 - deg, "cosine-and-sine-swapped"),
        (_deg(math.atan2(half, e)), "used-the-edge-as-the-opposite-side"),
        (_deg(math.acos(min(1.0, (a / 2) / e))), "used-half-a-base-edge-as-the-shadow"),
    ]
    text = (
        f"A pyramid VABCD has a horizontal rectangular base ABCD with AB = {a} cm and BC = {b} cm. The apex V is "
        f"directly above the centre of the base, and each sloping edge (VA, VB, VC and VD) is {e} cm long. "
        f"Find the angle between a sloping edge and the base, correct to one decimal place."
    )
    working = [
        f"The shadow of VA on the base is half the base diagonal: sqrt({a}^2 + {b}^2) / 2 = {half:.2f} cm.",
        f"The shadow is adjacent to the angle and VA is the hypotenuse: cos(angle) = {half:.2f} / {e}.",
        f"The angle is {_text(key)} degrees.",
    ]
    return text, key, wrong, working


@generator(TOPIC, level=10, presets={t: {"tier": t} for t in TIERS})
def y910_line_plane_angle(rng, tier="easy"):
    """The angle between a line and a plane in a cuboid or a pyramid; see the module docstring."""
    check(tier in TIERS, f"unknown tier {tier!r}")
    for _ in attempts(50, "a question with three distinct wrong angles"):
        if tier == "easy":
            text, key, wrong, working = _easy_case(rng)
        elif tier == "medium":
            if rng.random() < 0.5:
                text, key, wrong, working = _space_case(rng, "base")
            else:
                text, key, wrong, working = _pyramid_height_case(rng, rectangle=rng.random() < 0.3)
        else:
            if rng.random() < 0.5:
                text, key, wrong, working = _space_case(rng, "face")
            else:
                text, key, wrong, working = _pyramid_edge_case(rng)
        picked = _tagged(key, wrong)
        if len(picked) >= 3:
            break
    return dict(
        text=text,
        answer=numeric_answer(key),
        answer_text=_text(key),
        answer_latex=latex_answer(_text(key)),
        working=working,
        distractors=picked,
    )


# ---------------------------------------------------------------------------------------------------- exam


# base rectangles whose diagonal is a whole number, so the first part is exact
_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (6, 8, 10), (9, 12, 15), (12, 16, 20), (7, 24, 25), (20, 21, 29)]


@generator("exd_x_y910_line_plane_angle", level=10, presets={"exam": dict(tier="exam")})
def exd_x_y910_line_plane_angle(rng, tier="exam"):
    """Exam tier: base diagonal, angle of the space diagonal with the base, then with a vertical face."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam cuboid"):
        a, b, c = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            a, b = b, a
        scale = rng.choice([1, 1, 2])
        a, b, c = a * scale, b * scale, c * scale
        if c > 40:
            continue
        h = rng.randint(4, 3 * c // 2)
        space = math.sqrt(c * c + h * h)
        base_deg = _deg(math.atan2(h, c))
        face_deg = _deg(math.atan2(a, math.hypot(b, h)))
        if (
            h != a
            and h != b
            and 15 <= base_deg <= 70
            and 10 <= face_deg <= 70
            and rounds_unambiguously(space, 1)
            and rounds_unambiguously(base_deg, DP)
            and rounds_unambiguously(face_deg, DP)
        ):
            break
    check(abs(_plane_angle_by_vectors((a, b, h), (0, 0, 1)) - base_deg) < 1e-9, "base angle disagrees")
    check(abs(_plane_angle_by_vectors((a, b, h), (1, 0, 0)) - face_deg) < 1e-9, "face angle disagrees")
    space_key = round_half_up(Fraction(space), 1)
    base_key = _shown(base_deg)
    face_key = _shown(face_deg)
    text = phrase(
        rng,
        (
            "A rectangular steel frame is the cuboid ABCDEFGH. Its base ABCD is horizontal with AB = {a} cm and "
            "BC = {b} cm, and the vertical edges (AE, BF, CG and DH) are {h} cm long. "
            "The diagonal AG joins corner A to the opposite top corner G.",
            "A wooden crate is the cuboid ABCDEFGH, standing on its horizontal base ABCD with AB = {a} cm and "
            "BC = {b} cm, and the vertical edges (AE, BF, CG and DH) are {h} cm long. "
            "A diagonal AG runs from corner A to the opposite top corner G.",
            "The glass display case is the cuboid ABCDEFGH. Its floor ABCD is level, with AB = {a} cm and "
            "BC = {b} cm, and the vertical edges (AE, BF, CG and DH) are {h} cm long. "
            "The line AG joins corner A to the opposite top corner G.",
        ),
        a=a,
        b=b,
        h=h,
    )
    margin1 = 2 * stated_precision_tolerance(1)

    def one_dp(v):
        return stated_precision_string(round_half_up(Fraction(v), 1), 1)

    space_wrong = [
        Distractor(
            one_dp(a + b + h), "added-the-edges-for-the-diagonal", numeric_answer(round_half_up(Fraction(a + b + h), 1))
        ),
        Distractor(
            one_dp(math.hypot(a, h)),
            "left-out-an-edge-in-pythagoras",
            numeric_answer(round_half_up(Fraction(math.hypot(a, h)), 1)),
        ),
        Distractor(
            one_dp(math.hypot(c, 0) + h),
            "added-instead-of-using-pythagoras",
            numeric_answer(round_half_up(Fraction(c + h), 1)),
        ),
        Distractor(
            one_dp(math.sqrt(a * a + b * b + h * h) / 2),
            "halved-the-diagonal",
            numeric_answer(round_half_up(Fraction(space / 2), 1)),
        ),
    ]
    space_d = pick_distractors(one_dp(space), space_wrong, 3, tolerance=margin1)
    parts = [
        dict(
            prompt="Find the length of the base diagonal AC, in centimetres.",
            answer_kind="number",
            answer=c,
            answer_text=str(c),
            answer_latex=latex_answer(str(c)),
            marks=1,
            sensible_range=(0, None),
            working=[f"AC = sqrt({a}^2 + {b}^2) = sqrt({a * a + b * b}) = {c} cm."],
            distractors=pick_distractors(
                str(c),
                [
                    Distractor(str(a + b), "added-the-edges-for-the-diagonal", a + b),
                    Distractor(str(a * a + b * b), "forgot-the-square-root-in-the-shadow", a * a + b * b),
                    Distractor(str(abs(a - b)), "subtracted-the-edges-for-the-diagonal", abs(a - b)),
                ],
                3,
            ),
        ),
        dict(
            prompt="Find the length of the space diagonal AG, in centimetres, correct to one decimal place.",
            answer_kind="number",
            answer=numeric_answer(space_key),
            answer_text=one_dp(space),
            answer_latex=latex_answer(one_dp(space)),
            marks=2,
            sensible_range=(0, None),
            follow_through="sqrt(part_a**2 + %d)" % (h * h),
            working=[f"AG = sqrt(AC^2 + CG^2) = sqrt({c}^2 + {h}^2) = {space:.2f}, about {one_dp(space)} cm."],
            distractors=space_d,
        ),
        _part_fields(
            "Find the angle between AG and the base ABCD, correct to one decimal place.",
            base_key,
            [
                (_deg(math.atan2(h, a)), "used-a-base-edge-as-the-shadow"),
                (_deg(math.atan2(h, b)), "used-a-base-edge-as-the-shadow"),
                (90 - base_deg, "tangent-ratio-upside-down"),
                (_deg(math.atan2(c, h)), "tangent-ratio-upside-down"),
                (_deg(math.atan2(h, a + b)), "added-the-base-edges-for-the-shadow"),
            ],
            2,
            [
                f"The shadow of AG on the base is AC = {c} cm.",
                f"tan(angle) = {h} / {c}, so the angle is {_text(base_key)} degrees.",
            ],
            follow_through="atan(%d/part_a)*180/pi" % h,
        ),
        _part_fields(
            "Find the angle between AG and the vertical face BCGF, correct to one decimal place.",
            face_key,
            [
                (base_deg, "measured-to-the-base-instead-of-the-face"),
                (90 - face_deg, "tangent-ratio-upside-down"),
                (_deg(math.atan2(a, b + h)), "added-instead-of-using-pythagoras"),
                (_deg(math.atan2(a, max(b, h))), "used-one-edge-as-the-shadow"),
                (_deg(math.atan2(a, space)), "used-the-space-diagonal-as-the-shadow"),
            ],
            3,
            [
                f"G is on the face BCGF. The shadow of AG on that face is BG, the face diagonal sqrt({b}^2 + {h}^2) = {math.hypot(b, h):.2f} cm.",
                f"The perpendicular from A to the face is AB = {a} cm.",
                f"tan(angle) = {a} / {math.hypot(b, h):.2f}, so the angle is {_text(face_key)} degrees.",
            ],
        ),
    ]
    return dict(text=text, parts=parts)
