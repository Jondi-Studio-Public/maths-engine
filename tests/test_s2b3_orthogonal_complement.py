"""Tests `uc2-orthogonal-complement` (topic `s2b3_orthogonal_complement`) and its exam chain.

The oracle reads the spanning vectors, or the equations of the system, back out of the question TEXT and re-derives the
reduced row echelon basis of the complement with SymPy's `nullspace` and `rref`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2b3_orthogonal_complement_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.s2b3_support import (
    answer_matrix,
    check_exam_grading,
    check_single_tier,
    check_wiring,
    int_vectors,
    same_matrix,
    skill,
    stem,
)

TOPIC = "s2b3_orthogonal_complement"
EXAM = "exd_x_s2b3_orthogonal_complement"
SK = skill("uc2-orthogonal-complement")
TIERS = ("easy", "medium", "hard")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_NAMES = sympy.symbols("x y z w v")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _rref_rows(rows):
    reduced, pivots = sympy.Matrix(rows).rref()
    return reduced[: len(pivots), :]


def _complement_of_span(vectors):
    basis = sympy.Matrix(vectors).nullspace()
    return _rref_rows(sympy.Matrix.hstack(*basis).T.tolist())


def _equations(text):
    """The coefficient rows of the equations in `text` (everything after the opening brace or 'system')."""
    body = re.split(r"system |satisfy |: ", text, maxsplit=1)[1]
    body = re.split(r"\}\. |\. Find|\. Write", body)[0]
    rows = []
    for eq in body.split(", "):
        left = eq.replace(" = 0", "").replace("−", "-")
        expr = parse_expr(left, transformations=_TRANSFORMS, local_dict={str(n): n for n in _NAMES})
        rows.append([expr.coeff(n) for n in _NAMES[: 4 if "ℝ⁴" in text else 5]])
    return rows


def _oracle(text):
    if "span" in text:
        return _complement_of_span([v for v in int_vectors(text)])
    return _rref_rows(_equations(text))


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["ucv1-null-space", "vector-dot-product"])


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_reduced_basis(tier):
    shapes = Counter()
    systems = 0
    for seed in range(120):
        q = _q(tier, seed)
        key = answer_matrix(q)
        expected = _oracle(q.text)
        assert same_matrix(key, expected), (seed, q.text, key, expected)
        vs = int_vectors(q.text) if "span" in q.text else None
        if vs:
            assert all(sum(a * b for a, b in zip(row, v)) == 0 for row in key.tolist() for v in vs)
        systems += "span" not in q.text
        shapes[key.shape] += 1
    if tier == "hard":
        assert systems >= 20
    assert len(shapes) >= 2, shapes


def test_tiers_differ_in_dimension():
    easy = {int_vectors(_q("easy", s).text).__len__() for s in range(40)}
    assert {1, 2} <= easy
    medium_dims = {len(int_vectors(_q("medium", s).text)[0]) for s in range(40)}
    assert medium_dims == {4}
    hard = [_q("hard", s).text for s in range(60)]
    assert any("span" not in t for t in hard) and any("span" in t for t in hard)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=80)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 50, (tier, Counter(answers).most_common(3))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 3, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "matrix", "number"]
        assert [p.follow_through for p in q.parts] == [None, "5 - part_a", None, None]


def test_exam_keys_match_sympy():
    for seed in range(120):
        q = _exam(seed)
        vs = int_vectors(stem(q))
        assert len(vs) == 3 and len(vs[0]) == 5
        w = sympy.Matrix(vs)
        assert q.parts[0].answer == w.rank() == 2
        assert q.parts[1].answer == 5 - w.rank()
        assert same_matrix(answer_matrix(q.parts[2]), _complement_of_span(vs))
        z_text = re.search(r"z = \(([^)]*)\)", q.parts[3].prompt).group(1).split(", ")
        k = sympy.symbols("k")
        z = [k if t == "k" else sympy.Integer(int(t)) for t in z_text]
        roots = sympy.solve(list(w * sympy.Matrix(z)), k, dict=True)
        assert roots and roots[0][k] == q.parts[3].answer, (seed, q.parts[3].prompt)


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=40)


def test_exam_follow_through_from_a_wrong_dimension():
    for seed in range(30):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in (1, 3):
            result = grade(SK, q, {**canon, "a": str(wrong), "b": str(5 - wrong)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, wrong)
