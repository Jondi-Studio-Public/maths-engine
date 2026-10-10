"""Exam-tier sources for three linear-transformation skills (answer: an image or original point).

The practice tiers use only the textbook special cases: dilations, rotations
through multiples of 90 degrees and reflections in the axes or ``y = +-x``,
applied one at a time and described in words. The exam tier leaves that menu.

``exd_x_g4_lt_basic`` (basic transformations)
    * reflection in an arbitrary line ``y = (p/q)x`` through the origin, so
      the matrix has to be derived (it is ``1/(p^2+q^2) [[q^2-p^2, 2pq], [2pq, p^2-q^2]]``);
    * the same reflection with the line given by a second point on it;
    * a rotation through an angle given only by ``cos`` and ``sin``.

``exd_x_g4_lt_composition`` (composition)
    * two maps in function notation ``G o F``, at least one of them a
      non-axis reflection or a general-angle rotation (apply the right-hand
      map first);
    * three maps ``H o G o F``;
    * a composite whose final image is given and whose original point is asked.

``exd_x_g4_lt_inverse`` (inverses)
    * a transformation given only by an arbitrary invertible matrix (the
      determinant is not +-1, so the inverse needs the adjugate and a division);
    * the inverse of a composite (undo the maps in reverse order);
    * the inverse of a general-angle rotation.

Every question is built backwards from an integer answer point. The maps are
held as exact ``Fraction`` matrices; a draw is kept only if every intermediate
point is an integer, so the stated numbers are the ones a learner can actually
reach by hand. Distractors are the exact answers of named wrong methods
(reversed order, dropped map, forward map instead of inverse, transposed
matrix, and so on) and are removed if the pair grader would accept them.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors, rational_text, xy_solution

Mat = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
Pt = tuple[Fraction, Fraction]

# Pythagorean triples (cos numerator, sin numerator, hypotenuse).
_TRIPLES = ((3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (8, 15, 17), (15, 8, 17))
# Gradients p/q of the non-special mirror lines (never +-1, never 0 or vertical).
_GRADIENTS = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (-1, 2),
    (-2, 1),
    (-1, 3),
    (-2, 3),
    (-3, 4),
)
_DILATION_FACTORS = (-3, -2, 2, 3, 4)
_SPECIAL_LINES = ("the x-axis", "the y-axis", "the line y = x", "the line y = -x")


def _F(*v) -> tuple:
    return tuple(Fraction(a) for a in v)


def _mat(a, b, c, d) -> Mat:
    return (_F(a, b), _F(c, d))


def _apply(M: Mat, P: Pt) -> Pt:
    return (M[0][0] * P[0] + M[0][1] * P[1], M[1][0] * P[0] + M[1][1] * P[1])


def _inverse(M: Mat) -> Mat:
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return ((M[1][1] / det, -M[0][1] / det), (-M[1][0] / det, M[0][0] / det))


def _is_int(P: Pt) -> bool:
    return all(c.denominator == 1 for c in P)


def _pt_text(P) -> str:
    return f"x = {rational_text(Fraction(P[0]))}, y = {rational_text(Fraction(P[1]))}"


def _finish(text: str, answer: Pt, wrong: list, working: list[str]) -> dict | None:
    """The question dict, or None when fewer than two distractors survive."""
    x0, y0 = int(answer[0]), int(answer[1])
    solved = xy_solution(x0, y0)
    cands = [_pt_text(w) for w in wrong if w is not None and tuple(w) != tuple(answer)]
    solved["distractors"] = pick_distractors(solved["answer_text"], cands, tolerance=GRADER_TOLERANCE)
    if len(solved["distractors"]) < 2:
        return None
    return dict(text=text, latex=rf"\text{{{text}}}", working=working, **solved)


# -- the maps -------------------------------------------------------------------------------


class _Map:
    """A named linear map: its description for the prompt and its exact matrix."""

    def __init__(self, kind: str, desc: str, M: Mat, special: bool):
        self.kind, self.desc, self.M, self.special = kind, desc, M, special


def _dilation(rng: random.Random) -> _Map:
    while True:
        l1, l2 = rng.choice(_DILATION_FACTORS), rng.choice(_DILATION_FACTORS)
        if l1 != l2:
            return _Map("dilation", f"the dilation (x, y) -> ({l1}x, {l2}y)", _mat(l1, 0, 0, l2), True)


def _rotation_special(rng: random.Random) -> _Map:
    theta = rng.choice((90, 270))
    M = {90: _mat(0, -1, 1, 0), 270: _mat(0, 1, -1, 0)}[theta]
    return _Map("rotation", f"a rotation of {theta} degrees anticlockwise about the origin", M, True)


def _rotation_general_desc(a: int, b: int, c: int) -> str:
    return f"a rotation about the origin through an angle theta, where cos(theta) = {a}/{c} and sin(theta) = {b}/{c}"


def _rotation_general(rng: random.Random) -> _Map:
    a, b, c = rng.choice(_TRIPLES)
    b *= rng.choice((-1, 1))
    M = (_F(Fraction(a, c), Fraction(-b, c)), _F(Fraction(b, c), Fraction(a, c)))
    return _Map("rotation", _rotation_general_desc(a, b, c), M, False)


def _reflection_special(rng: random.Random) -> _Map:
    line = rng.choice(_SPECIAL_LINES)
    M = {
        "the x-axis": _mat(1, 0, 0, -1),
        "the y-axis": _mat(-1, 0, 0, 1),
        "the line y = x": _mat(0, 1, 1, 0),
        "the line y = -x": _mat(0, -1, -1, 0),
    }[line]
    return _Map("reflection", f"a reflection in {line}", M, True)


def _line_text(p: int, q: int) -> str:
    coeff = {1: "", -1: "-"}.get(p, str(p))  # y = -x/2, not y = -1x/2
    if q == 1:
        return f"y = {coeff}x"
    return f"y = {coeff}x/{q}"


def _reflection_matrix(p: int, q: int) -> Mat:
    h = p * p + q * q
    return (
        _F(Fraction(q * q - p * p, h), Fraction(2 * p * q, h)),
        _F(Fraction(2 * p * q, h), Fraction(p * p - q * q, h)),
    )


def _reflection_general(rng: random.Random) -> _Map:
    p, q = rng.choice(_GRADIENTS)
    return _Map("reflection", f"a reflection in the line {_line_text(p, q)}", _reflection_matrix(p, q), False)


_SPECIAL_MAKERS = (_dilation, _rotation_special, _reflection_special)
_GENERAL_MAKERS = (_rotation_general, _reflection_general)


def _small(P: Pt, bound: int = 60) -> bool:
    return _is_int(P) and all(abs(c) <= bound for c in P)


def _chain(maps: list, P: Pt) -> list:
    """The points P, F1(P), F2(F1(P)), ... for maps listed in the order they act."""
    pts = [P]
    for m in maps:
        pts.append(_apply(m.M, pts[-1]))
    return pts


def _point(rng: random.Random, bound: int = 6) -> Pt:
    while True:
        P = _F(rng.randint(-bound, bound), rng.randint(-bound, bound))
        if P != _F(0, 0):
            return P


def _undo_text(maps: list) -> str:
    return " and then ".join(m.desc for m in maps)


# -- exd_x_g4_lt_basic ----------------------------------------------------------------------


def _basic_reflection(rng: random.Random) -> dict:
    """Reflect in y = (p/q)x. P = s*(q,p) + t*(-p,q) (along the mirror plus across it), image s*(q,p) - t*(-p,q)."""
    while True:
        p, q = rng.choice(_GRADIENTS)
        s, t = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
        P = _F(s * q - t * p, s * p + t * q)
        ans = _F(s * q + t * p, s * p - t * q)
        M = _reflection_matrix(p, q)
        check(_apply(M, P) == ans, "reflection matrix disagrees with the along/across construction")
        if max(abs(c) for c in (*P, *ans)) > 30:
            continue
        gradient_inverted = _apply(_reflection_matrix(q, p), P)
        along_only = _F(s * q, s * p)  # projected instead of reflected
        opposite = (-ans[0], -ans[1])  # reflected in the perpendicular line
        text = f"Find the image of the point ({int(P[0])}, {int(P[1])}) when it is reflected in the line {_line_text(p, q)}."
        working = [
            f"The reflection matrix for the line is 1/{p * p + q * q} [[{q * q - p * p}, {2 * p * q}], [{2 * p * q}, {p * p - q * q}]].",
            f"Image = ({int(ans[0])}, {int(ans[1])}).",
        ]
        out = _finish(text, ans, [gradient_inverted, along_only, opposite, P], working)
        if out:
            return out


def _basic_reflection_two_point(rng: random.Random) -> dict:
    """Same reflection, with the mirror given as the line through the origin and a stated point."""
    while True:
        q, p = rng.choice(((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (3, 4), (4, 3), (1, 4), (2, 5)))
        p *= rng.choice((-1, 1))
        s, t = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
        P = _F(s * q - t * p, s * p + t * q)
        ans = _F(s * q + t * p, s * p - t * q)
        if _apply(_reflection_matrix(p, q), P) != ans:
            raise AssertionError("two-point reflection construction disagrees with the matrix")
        if max(abs(c) for c in (*P, *ans)) > 30:
            continue
        swapped_direction = _apply(_reflection_matrix(q, p), P)
        along_only = _F(s * q, s * p)
        opposite = (-ans[0], -ans[1])
        text = (
            f"The line L passes through the origin and the point ({q}, {p}). "
            f"Find the image of the point ({int(P[0])}, {int(P[1])}) when it is reflected in L."
        )
        working = [f"L has gradient {p}/{q}.", f"Image = ({int(ans[0])}, {int(ans[1])})."]
        out = _finish(text, ans, [swapped_direction, along_only, opposite, P], working)
        if out:
            return out


def _basic_rotation(rng: random.Random) -> dict:
    """Rotate through an angle given by cos and sin. P = (a - ib)(u + iv), so the image is c(u + iv)."""
    while True:
        a, b, c = rng.choice(_TRIPLES)
        b *= rng.choice((-1, 1))
        u, v = rng.randint(-2, 2), rng.randint(-2, 2)
        if (u, v) == (0, 0):
            continue
        P = _F(a * u + b * v, a * v - b * u)
        ans = _F(c * u, c * v)
        M = (_F(Fraction(a, c), Fraction(-b, c)), _F(Fraction(b, c), Fraction(a, c)))
        check(_apply(M, P) == ans, "rotation matrix disagrees with the Gaussian-integer construction")
        clockwise = _apply((_F(Fraction(a, c), Fraction(b, c)), _F(Fraction(-b, c), Fraction(a, c))), P)
        swapped = _apply((_F(Fraction(b, c), Fraction(-a, c)), _F(Fraction(a, c), Fraction(b, c))), P)
        as_reflection = _apply((_F(Fraction(a, c), Fraction(b, c)), _F(Fraction(b, c), Fraction(-a, c))), P)
        text = f"Find the image of the point ({int(P[0])}, {int(P[1])}) under {_rotation_general_desc(a, b, c)}."
        working = [
            f"The rotation matrix is [[{a}/{c}, {-b}/{c}], [{b}/{c}, {a}/{c}]].",
            f"Image = ({int(ans[0])}, {int(ans[1])}).",
        ]
        out = _finish(text, ans, [clockwise, swapped, as_reflection, P], working)
        if out:
            return out


@generator(
    "exd_x_g4_lt_basic",
    level=13,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_g4_lt_basic(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier basic transformation; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_basic_reflection, _basic_reflection_two_point, _basic_rotation))(rng)


# -- exd_x_g4_lt_composition ----------------------------------------------------------------


_NAMES = ("F", "G", "H")


def _definitions(named: list) -> str:
    parts = [f"{n} be {m.desc}" for n, m in named]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _composition_forward(rng: random.Random, n_maps: int) -> dict:
    """Image of P under the composite written right-to-left (the right-hand map acts first)."""
    while True:
        if n_maps == 2:
            maps = [rng.choice(_GENERAL_MAKERS)(rng), rng.choice(_SPECIAL_MAKERS)(rng)]
            rng.shuffle(maps)
        else:
            maps = [rng.choice(_SPECIAL_MAKERS)(rng) for _ in range(3)]
        kinds = [m.kind for m in maps]
        if len(set(kinds)) < len(kinds) and n_maps == 3:
            continue
        P = _point(rng)
        pts = _chain(maps, P)
        if not all(_small(p) for p in pts):
            continue
        ans = pts[-1]
        names = _NAMES[:n_maps]
        named = list(zip(names, maps))  # names[0] acts first
        notation = " o ".join(reversed(names))
        reversed_order = _chain(list(reversed(maps)), P)[-1]
        drop_first = _chain(maps[1:], P)[-1]
        drop_last = _chain(maps[:-1], P)[-1]
        text = (
            f"Let {_definitions(named)}. "
            f"Find the image of the point ({int(P[0])}, {int(P[1])}) under the composite {notation}, "
            f"where (G o F)(P) means G(F(P))."
        )
        working = [
            f"{names[i]} acts at step {i + 1}: ({int(pts[i + 1][0])}, {int(pts[i + 1][1])})." for i in range(n_maps)
        ]
        out = _finish(text, ans, [reversed_order, drop_first, drop_last, P], working)
        if out:
            return out


def _composition_find_original(rng: random.Random) -> dict:
    """The composite 'first A, then B' sends an unknown P to Q; find P."""
    while True:
        maps = [rng.choice(_SPECIAL_MAKERS)(rng), rng.choice(_SPECIAL_MAKERS)(rng)]
        if maps[0].kind == maps[1].kind:
            continue
        P = _point(rng)
        pts = _chain(maps, P)
        if not all(_small(p) for p in pts):
            continue
        Q = pts[-1]
        undo_wrong_order = _chain(
            [_Map("inv", "", _inverse(maps[0].M), True), _Map("inv", "", _inverse(maps[1].M), True)], Q
        )[-1]
        only_second_undone = _apply(_inverse(maps[1].M), Q)
        forward_again = _chain(list(reversed(maps)), Q)[-1]
        text = (
            f"A transformation T is {maps[0].desc}, followed by {maps[1].desc}. "
            f"T maps a point P to the image ({int(Q[0])}, {int(Q[1])}). Find P."
        )
        working = [
            "Undo the maps in the reverse order.",
            f"After undoing the second map: ({int(pts[1][0])}, {int(pts[1][1])}).",
            f"P = ({int(P[0])}, {int(P[1])}).",
        ]
        out = _finish(text, P, [undo_wrong_order, only_second_undone, forward_again, Q], working)
        if out:
            return out


@generator(
    "exd_x_g4_lt_composition",
    level=13,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_g4_lt_composition(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier composition; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    pick = rng.randrange(3)
    if pick == 0:
        return _composition_forward(rng, 2)
    if pick == 1:
        return _composition_forward(rng, 3)
    return _composition_find_original(rng)


# -- exd_x_g4_lt_inverse --------------------------------------------------------------------


def _inverse_matrix_question(rng: random.Random) -> dict:
    """An arbitrary invertible matrix (|det| >= 2) maps P to Q; find P."""
    while True:
        M = _mat(*[rng.randint(-4, 4) for _ in range(4)])
        det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        if abs(det) < 2:
            continue
        P = _point(rng, 5)
        Q = _apply(M, P)
        if not _small(Q):
            continue
        forward_again = _apply(M, Q)
        adjugate_only = _apply(((M[1][1], -M[0][1]), (-M[1][0], M[0][0])), Q)
        transposed = _apply(_inverse(((M[0][0], M[1][0]), (M[0][1], M[1][1]))), Q)
        offdiag_kept = _apply(((M[1][1] / det, M[0][1] / det), (M[1][0] / det, M[0][0] / det)), Q)
        text = (
            f"A linear transformation has matrix [{M[0][0]} {M[0][1]}; {M[1][0]} {M[1][1]}] "
            f"and maps a point P to the image ({int(Q[0])}, {int(Q[1])}). Find P."
        )
        working = [
            f"The determinant is {det}.",
            f"M^-1 = 1/{det} [[{M[1][1]}, {-M[0][1]}], [{-M[1][0]}, {M[0][0]}]].",
            f"P = ({int(P[0])}, {int(P[1])}).",
        ]
        out = _finish(text, P, [forward_again, adjugate_only, transposed, offdiag_kept], working)
        if out:
            return out


def _inverse_composite_question(rng: random.Random) -> dict:
    """First map A then map B sends P to Q; the inverse undoes B first, then A."""
    while True:
        maps = [rng.choice(_SPECIAL_MAKERS)(rng), rng.choice(_SPECIAL_MAKERS)(rng)]
        if maps[0].kind == maps[1].kind:
            continue
        P = _point(rng)
        pts = _chain(maps, P)
        if not all(_small(p) for p in pts):
            continue
        Q = pts[-1]
        inv = [_Map("inv", "", _inverse(m.M), True) for m in maps]
        wrong_order = _chain(inv, Q)[-1]
        forward_reversed = _chain(list(reversed(maps)), Q)[-1]
        one_undone = _apply(inv[1].M, Q)
        text = (
            f"The transformation T is {maps[0].desc}, followed by {maps[1].desc}. "
            f"Find the point that T sends to ({int(Q[0])}, {int(Q[1])})."
        )
        working = [
            "T^-1 undoes the second map first, then the first.",
            f"T^-1 ({int(Q[0])}, {int(Q[1])}) = ({int(P[0])}, {int(P[1])}).",
        ]
        out = _finish(text, P, [wrong_order, forward_reversed, one_undone, Q], working)
        if out:
            return out


def _inverse_rotation_question(rng: random.Random) -> dict:
    """A rotation given by cos and sin maps P to Q = c(u + iv); find P by rotating back."""
    while True:
        a, b, c = rng.choice(_TRIPLES)
        b *= rng.choice((-1, 1))
        u, v = rng.randint(-2, 2), rng.randint(-2, 2)
        if (u, v) == (0, 0):
            continue
        P = _F(a * u + b * v, a * v - b * u)
        Q = _F(c * u, c * v)
        M = (_F(Fraction(a, c), Fraction(-b, c)), _F(Fraction(b, c), Fraction(a, c)))
        check(_apply(M, P) == Q, "rotation matrix disagrees with the Gaussian-integer construction")
        forward_again = _apply(M, Q)
        swapped = _apply((_F(Fraction(b, c), Fraction(-a, c)), _F(Fraction(a, c), Fraction(b, c))), Q)
        sign_flipped = (-P[0], -P[1])
        text = f"{_rotation_general_desc(a, b, c).capitalize()}, maps a point P to the image ({int(Q[0])}, {int(Q[1])}). Find P."
        working = ["Rotate the image back through -theta.", f"P = ({int(P[0])}, {int(P[1])})."]
        out = _finish(text, P, [forward_again, swapped, sign_flipped, Q], working)
        if out:
            return out


@generator(
    "exd_x_g4_lt_inverse",
    level=13,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_g4_lt_inverse(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier inverse of a transformation; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_inverse_matrix_question, _inverse_composite_question, _inverse_rotation_question))(rng)
