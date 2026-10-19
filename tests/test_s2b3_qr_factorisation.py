"""Tests `uc2-qr-factorisation` (topic `s2b3_qr`) and its exam chain `exd_x_s2b3_qr`.

The oracle reads the matrix (and the vector b of the exam) back out of the question TEXT and re-derives Q and R with
SymPy's `QRdecomposition`, whose R has a positive diagonal.
"""

from __future__ import annotations

from collections import Counter

import pytest
import sympy

import mathgen.generators.s2b3_qr_factorisation_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.s2b3_support import (
    answer_matrix,
    answer_vector,
    check_exam_grading,
    check_single_tier,
    check_wiring,
    int_vectors,
    matrix_in,
    same_matrix,
    skill,
    stem,
)

TOPIC = "s2b3_qr"
EXAM = "exd_x_s2b3_qr"
SK = skill("uc2-qr-factorisation")
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _num(value):
    return sympy.Rational(value["p"], value["q"]) if isinstance(value, dict) else sympy.Integer(value)


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["uc2-gram-schmidt", "matrix-multiplication"])


def _check_factor(q, which):
    a = matrix_in(q.text)
    qq, rr = a.QRdecomposition()
    key = answer_matrix(q)
    assert same_matrix(key, qq if which == "Q" else rr), (q.text, key)
    # whichever factor is asked for, the pair rebuilds A
    assert (qq * rr - a).applyfunc(sympy.simplify) == sympy.zeros(*a.shape)
    assert all(rr[i, i].is_positive for i in range(rr.rows))
    return a


def test_easy_key_is_r_for_a_3x2_matrix():
    for seed in range(120):
        q = _q("easy", seed)
        assert "Find R" in q.text or "matrix R" in q.text or "What is R" in q.text
        a = _check_factor(q, "R")
        assert a.shape == (3, 2)
        assert all(sympy.sympify(e).is_Integer for e in answer_matrix(q)), "easy R has whole-number entries"


def test_medium_key_is_q_for_a_3x2_matrix():
    for seed in range(120):
        q = _q("medium", seed)
        a = _check_factor(q, "Q")
        assert a.shape == (3, 2)
        assert all(sympy.sympify(e).is_Rational for e in answer_matrix(q))


def test_hard_covers_three_columns_and_square_roots():
    seen = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        which = "Q" if "Find Q" in q.text or "matrix Q" in q.text or "What is Q" in q.text else "R"
        a = _check_factor(q, which)
        irrational = any(not sympy.sympify(e).is_Rational for e in answer_matrix(q))
        seen[(a.shape, which, irrational)] += 1
    assert any(shape == (3, 3) for shape, _w, _i in seen)
    assert any(irr for _s, _w, irr in seen)
    assert any(not irr for _s, _w, irr in seen)
    assert any(w == "Q" for _s, w, _i in seen) and any(w == "R" for _s, w, _i in seen)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=60)


def test_surd_answers_accept_the_common_spellings():
    checked = 0
    for seed in range(160):
        q = _q("hard", seed)
        if all(sympy.sympify(e).is_Rational for e in answer_matrix(q)):
            continue
        rows = answer_matrix(q).tolist()
        plain = "[" + ", ".join("[" + ", ".join(str(sympy.sympify(e)) for e in row) + "]" for row in rows) + "]"
        assert grade(SK, q, plain).correct, (seed, plain)
        checked += 1
    assert checked >= 20


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 30, (tier, Counter(answers).most_common(3))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 3]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "matrix"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, False]


def test_exam_keys_match_sympy():
    for seed in range(120):
        q = _exam(seed)
        a = matrix_in(stem(q))
        b = sympy.Matrix(int_vectors(stem(q).split("b = ")[1])[0])
        qq, rr = a.QRdecomposition()
        assert [_num(q.parts[i].answer) for i in range(3)] == [rr[0, 0], rr[0, 1], rr[1, 1]]
        assert answer_vector(q.parts[3]) == list(a.pinv() * b)
        assert list(rr.inv() * qq.T * b) == answer_vector(q.parts[3])


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=40)


def test_exam_follow_through_chain():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        a = matrix_in(stem(q))
        canon = {p.label: p.answer_text for p in q.parts}
        n12, n22 = (a[:, 0].dot(a[:, 1])), a[:, 1].dot(a[:, 1])
        wrong_r11 = q.parts[0].answer + 1
        wrong_r12 = sympy.Rational(n12, wrong_r11)
        result = grade(SK, q, {**canon, "a": str(wrong_r11), "b": str(wrong_r12)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result)
        wrong_r22 = sympy.sqrt(n22 - wrong_r12**2)
        if wrong_r22.is_real and wrong_r22.is_Rational and wrong_r22 > 0:
            result = grade(SK, q, {**canon, "a": str(wrong_r11), "b": str(wrong_r12), "c": str(wrong_r22)})
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result)
        checked += 1
    assert checked == 40
