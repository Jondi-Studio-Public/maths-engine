"""Tests `uc2-least-squares-normal-equations` (topic `s2b3_least_squares`) and its exam chain.

The oracle reads the data points (or the matrix and right-hand side) back out of the question TEXT and re-derives the
least-squares solution with SymPy's pseudo-inverse, and the residual with b - Ax.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.s2b3_least_squares_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.s2b3_support import (
    answer_vector,
    check_exam_grading,
    check_single_tier,
    check_wiring,
    int_vectors,
    matrix_in,
    skill,
    stem,
)

TOPIC = "s2b3_least_squares"
EXAM = "exd_x_s2b3_least_squares"
SK = skill("uc2-least-squares-normal-equations")
TIERS = ("easy", "medium", "hard")
_POINT_RE = re.compile(r"\((-?\d+), (-?\d+)\)")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _num(value):
    return sympy.Rational(value["p"], value["q"]) if isinstance(value, dict) else sympy.Integer(value)


def _points(text):
    return [(int(x), int(y)) for x, y in _POINT_RE.findall(text)]


def _fit(points, degree):
    a = sympy.Matrix([[x**k for k in range(degree + 1)] for x, _y in points])
    b = sympy.Matrix([y for _x, y in points])
    return a, b, a.pinv() * b


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["uc2-projection-onto-subspace", "matrix-solve-system"])


def test_easy_key_is_the_least_squares_line():
    for seed in range(120):
        q = _q("easy", seed)
        pts = _points(q.text)
        assert len(pts) == 3
        _a, _b, x = _fit(pts, 1)
        assert answer_vector(q) == list(x), (seed, q.text)
        assert len({p[0] for p in pts}) == 3


def test_medium_key_is_the_solution_or_the_residual():
    seen = Counter()
    for seed in range(120):
        q = _q("medium", seed)
        a = matrix_in(q.text)
        b = sympy.Matrix(int_vectors(q.text.split("b = ")[1])[0])
        x = a.pinv() * b
        if "residual" in q.text:
            assert answer_vector(q) == list(b - a * x), (seed, q.text)
            seen["residual"] += 1
        else:
            assert answer_vector(q) == list(x), (seed, q.text)
            seen["solution"] += 1
        assert b != a * x, "the system must be inconsistent"
    assert seen["residual"] >= 30 and seen["solution"] >= 30


def test_hard_key_is_the_least_squares_parabola():
    sizes = Counter()
    for seed in range(120):
        q = _q("hard", seed)
        pts = _points(q.text)
        _a, _b, x = _fit(pts, 2)
        assert answer_vector(q) == list(x), (seed, q.text)
        sizes[len(pts)] += 1
    assert set(sizes) <= {4, 5} and len(sizes) == 2


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=80)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 45, (tier, Counter(answers).most_common(3))


def test_contexts_are_australian():
    texts = " ".join(_q("easy", s).text for s in range(60))
    assert any(place in texts for place in ("Port Augusta", "Toowoomba", "Queensland", "Ningaloo", "Adelaide"))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "expression"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, False]


def test_exam_keys_match_sympy():
    for seed in range(120):
        q = _exam(seed)
        pts = _points(stem(q))
        a, b, x = _fit(pts, 1)
        assert len(pts) == 4
        assert q.parts[0].answer == (a.T * a).det()
        assert _num(q.parts[1].answer) == x[1]
        assert _num(q.parts[2].answer) == x[0]
        residual = b - a * x
        assert sympy.simplify(sympy.sympify(q.parts[3].answer) - sympy.sqrt(residual.dot(residual))) == 0
        assert a.T * residual == sympy.zeros(2, 1)


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=40)


def test_exam_follow_through_from_a_wrong_determinant():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        pts = _points(stem(q))
        n, sx = len(pts), sum(x for x, _ in pts)
        sy, sxy = sum(y for _, y in pts), sum(x * y for x, y in pts)
        wrong_det = (q.parts[0].answer) + 3
        slope = sympy.Rational(n * sxy - sx * sy, wrong_det)
        result = grade(SK, q, {**canon, "a": str(wrong_det), "b": str(slope)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result)
        checked += 1
    assert checked == 40
