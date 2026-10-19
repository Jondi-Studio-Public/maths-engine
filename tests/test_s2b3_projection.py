"""Tests `uc2-projection-onto-subspace` (topic `s2b3_projection`) and its exam chain `exd_x_s2b3_projection`.

The oracle reads the spanning vectors, the vector to project or the equation of the hyperplane back out of the question
TEXT and re-derives the projection with SymPy as A (A^T A)^-1 A^T.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.s2b3_projection_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.s2b3_support import (
    answer_matrix,
    answer_vector,
    check_exam_grading,
    check_single_tier,
    check_wiring,
    int_vectors,
    named_vector,
    same_matrix,
    skill,
    stem,
)

TOPIC = "s2b3_projection"
EXAM = "exd_x_s2b3_projection"
SK = skill("uc2-projection-onto-subspace")
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _num(value):
    return sympy.Rational(value["p"], value["q"]) if isinstance(value, dict) else sympy.Integer(value)


def _projector(columns):
    a = sympy.Matrix(columns).T
    return a * (a.T * a).inv() * a.T


def _hyperplane_normal(text):
    body = re.search(r"(?:ℝ[³⁴] with |given by )(.*?) = 0", text).group(1)
    n = [0] * 4
    for sign, coef, idx in re.findall(r"([+-]?)\s*(\d*)x(\d)", body.replace(" ", "")):
        n[int(idx) - 1] = (-1 if sign == "-" else 1) * int(coef or 1)
    return n[: 3 if "ℝ³" in text else 4]


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["vector-projection", "ucv1-null-space"])


def test_easy_key_is_the_line_projection():
    dims = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        a, v = sympy.Matrix(named_vector(q.text, "a")), sympy.Matrix(named_vector(q.text, "v"))
        assert answer_vector(q) == list(v.dot(a) / a.dot(a) * a), (seed, q.text)
        dims[len(a)] += 1
    assert set(dims) == {3, 4}


def test_medium_key_is_the_plane_projection():
    for seed in range(120):
        q = _q("medium", seed)
        a1, a2, v = (sympy.Matrix(named_vector(q.text, n)) for n in ("a1", "a2", "v"))
        expected = _projector([list(a1), list(a2)]) * v
        assert answer_vector(q) == list(expected), (seed, q.text)
        assert a1.dot(a2) != 0, "the spanning vectors must not be orthogonal"
        assert (v - expected).dot(a1) == 0 and (v - expected).dot(a2) == 0


def test_hard_key_is_the_projection_matrix():
    seen = Counter()
    for seed in range(120):
        q = _q("hard", seed)
        key = answer_matrix(q)
        if "span" in q.text or "a1 =" in q.text:
            vs = int_vectors(q.text)
            assert len(vs) == 2 and len(vs[0]) == 3
            expected = _projector(vs)
            seen["plane"] += 1
        else:
            n = _hyperplane_normal(q.text)
            expected = sympy.eye(len(n)) - sympy.Matrix(n) * sympy.Matrix(n).T / sympy.Matrix(n).dot(sympy.Matrix(n))
            seen[f"hyperplane{len(n)}"] += 1
        assert same_matrix(key, expected), (seed, q.text)
        assert key * key == key and key == key.T
    assert seen["plane"] >= 30 and seen["hyperplane3"] + seen["hyperplane4"] >= 30, seen


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
        assert [p.marks for p in q.parts] == [3, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["matrix", "fraction", "expression", "expression"]
        assert q.parts[2].follow_through == "sqrt(part_b)"
        assert q.parts[3].follow_through.startswith("sqrt(part_b/")


def test_exam_keys_match_sympy():
    for seed in range(120):
        q = _exam(seed)
        a1, a2, v = (sympy.Matrix(named_vector(stem(q), n)) for n in ("a1", "a2", "v"))
        p = _projector([list(a1), list(a2)]) * v
        d2 = (v - p).dot(v - p)
        assert answer_vector(q.parts[0]) == list(p)
        assert _num(q.parts[1].answer) == d2
        assert sympy.simplify(sympy.sympify(q.parts[2].answer) - sympy.sqrt(d2)) == 0
        assert sympy.simplify(sympy.sympify(q.parts[3].answer) - sympy.sqrt(d2 / v.dot(v))) == 0
        # the distance also equals |n . v| / |n| for the normal n = a1 x a2
        n = a1.cross(a2)
        assert sympy.simplify(sympy.sqrt(d2) - abs(n.dot(v)) / sympy.sqrt(n.dot(n))) == 0


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=40)


def test_exam_surd_replies_in_other_forms_are_accepted():
    for seed in range(30):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        alt = {**canon, "c": str(round(float(sympy.sympify(q.parts[2].answer)), 2))}
        assert grade(SK, q, alt).parts["c"].marks == 1, (seed, alt["c"])


def test_exam_follow_through_from_a_wrong_squared_distance():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        v_sq = sum(x * x for x in named_vector(stem(q), "v"))
        wrong = _num(q.parts[1].answer) / 2
        if wrong >= v_sq or wrong in (0, 1) or wrong == _num(q.parts[1].answer):
            continue
        reply_c = str(sympy.sqrt(wrong))
        reply_d = str(sympy.sqrt(wrong / v_sq))
        result = grade(SK, q, {**canon, "b": str(wrong), "c": reply_c, "d": reply_d})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, result)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 20
