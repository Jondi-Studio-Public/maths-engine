"""Tests for the multi-part exam source of ucv1-coordinate-basis: determinant, two coordinates in a basis, and the
sum of the entries of T(v) for a map diagonal in that basis. Keys are re-derived from the rendered stem with sympy."""

from __future__ import annotations

import re

import sympy

import mathgen.generators.exd_x_r2g_basis_map  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_basis_map"
SKILL = "ucv1-coordinate-basis"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]


def _text(v) -> str:
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def _facts(q):
    b1 = [int(x) for x in re.search(r"b1 = \((-?\d+), (-?\d+)\)", q.stem).groups()]
    b2 = [int(x) for x in re.search(r"b2 = \((-?\d+), (-?\d+)\)", q.stem).groups()]
    v = [int(x) for x in re.search(r"v = \((-?\d+), (-?\d+)\)", q.stem).groups()]
    l1 = int(re.search(r"T\(b1\) = (-?\d+)b1", q.stem).group(1))
    l2 = int(re.search(r"T\(b2\) = (-?\d+)b2", q.stem).group(1))
    return b1, b2, v, l1, l2


def _solve(q):
    b1, b2, v, l1, l2 = _facts(q)
    basis = sympy.Matrix([[b1[0], b2[0]], [b1[1], b2[1]]])
    c = basis.LUsolve(sympy.Matrix(v))
    image = basis * sympy.Matrix([l1 * c[0], l2 * c[1]])
    return basis.det(), c, image[0] + image[1], (b1, b2, v, l1, l2)


def oracle(q):
    det, c, total, _ = _solve(q)
    return {"a": _text(det), "b": _text(c[0]), "c": _text(c[1]), "d": _text(total)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"b", "c", "d"}, "p13f_coordinate_basis")


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_follow_through_from_wrong_determinant():
    """A determinant of the wrong sign flips both coordinates and is carried through (b), (c) and (d)."""

    def wrong_chain(q):
        det, c, _total, (b1, b2, _v, l1, l2) = _solve(q)
        c1, c2 = -c[0], -c[1]
        image = sympy.Matrix([[b1[0], b2[0]], [b1[1], b2[1]]]) * sympy.Matrix([l1 * c1, l2 * c2])
        replies = {"a": _text(-det), "b": _text(c1), "c": _text(c2), "d": _text(image[0] + image[1])}
        return "a", replies, 5

    follow_through_check(TOPIC, SKILL, wrong_chain)


def test_follow_through_into_image_only():
    """A wrong c1 in (b) is carried into (d) as written, with the right c2."""

    def wrong_chain(q):
        det, c, _total, (b1, b2, _v, l1, l2) = _solve(q)
        c1 = c[0] + 1
        image = sympy.Matrix([[b1[0], b2[0]], [b1[1], b2[1]]]) * sympy.Matrix([l1 * c1, l2 * c[1]])
        replies = {"a": _text(det), "b": _text(c1), "c": _text(c[1]), "d": _text(image[0] + image[1])}
        return "b", replies, 4

    follow_through_check(TOPIC, SKILL, wrong_chain)
