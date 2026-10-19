"""Tests `uc2-svd-small-matrices` (topic `w6c_svd`) and its exam chain `exd_x_w6c_svd`.

The oracle reads the matrix back out of the question TEXT and re-derives each key with SymPy: the eigenvalues of AᵀA
(non-zero ones), the Frobenius norm, the rank-1 truncation of the SVD, and the null space of AᵀA − σ₁²I.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction

import pytest
import sympy

import mathgen.generators.w6c_svd_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests import w6c_support as support

TOPIC = "w6c_svd"
EXAM = "exd_x_w6c_svd"
SKILL = support.skill_named("uc2-svd-small-matrices")
_MATRIX = re.compile(r"\[\[[^A-Za-zλσ]*?\]\]")


def _matrix(text):
    return sympy.Matrix(ast.literal_eval(_MATRIX.search(text).group(0)))


def _singular(m):
    """Positive singular values, largest first, from the eigenvalues of the smaller Gram matrix."""
    gram = m * m.T if m.rows <= m.cols else m.T * m
    return sorted((sympy.sqrt(v) for v in gram.eigenvals() if v != 0), reverse=True)


def _value(text):
    return sympy.Rational(Fraction(text).numerator, Fraction(text).denominator)


def test_skill_wiring():
    support.check_wiring(SKILL, TOPIC, EXAM, ["ucv1-eigenvectors"])


@pytest.mark.parametrize("tier", support.TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    support.check_tier(SKILL, TOPIC, tier)


def test_easy_key_is_the_largest_singular_value_of_a_2x2():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = _matrix(q.text)
        assert m.shape == (2, 2) and m != m.T and not m.is_diagonal()
        assert _value(q.answer_text) == _singular(m)[0], (seed, q.text)
        assert _singular(m)[0].is_integer and m.det() != 0


def test_medium_key_is_the_smaller_singular_value_of_a_2x3():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = _matrix(q.text)
        assert m.shape == (2, 3) and m.rank() == 2
        values = _singular(m)
        assert _value(q.answer_text) == values[1] and values[0] > values[1], (seed, q.text)
        assert "0" in q.distractors  # the zero eigenvalue of the 3×3 AᵀA is the named trap


def test_hard_keys():
    kinds = set()
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _matrix(q.text)
        s1, s2 = _singular(m)
        key = _value(q.answer_text)
        if "‖A − A₁‖_F" in q.text:
            # Eckart-Young checked directly: the truncated SVD's Frobenius distance.
            u, sigma, vt = _svd_exact(m)
            a1 = sigma[0] * u[:, 0] * vt[0, :]
            assert sympy.sqrt(sum(x * x for x in (m - a1))) == key == s2
            kinds.add("error")
        elif "σ₁/σ₂" in q.text or "‖A⁻¹‖₂" in q.text:
            assert m.shape == (2, 2) and key == s1 / s2 and key.q != 1
            if "‖A⁻¹‖₂" in q.text:
                assert max(abs(v) for v in m.inv().singular_values()) * s1 == key
            kinds.add("condition")
        else:
            assert key == s1**2 / sum(x * x for x in m), (seed, q.text)
            kinds.add("share")
    assert kinds == {"error", "condition", "share"}


def _svd_exact(m):
    """(U, sigma, Vᵀ) with orthonormal exact columns, from the eigenvectors of the Gram matrix."""
    gram = m.T * m
    pairs = sorted(gram.eigenvects(), key=lambda t: -t[0])
    vs = []
    for value, _mult, vecs in pairs:
        if value != 0:
            v = vecs[0]
            vs.append(v / v.norm())
    sigma = [sympy.sqrt(p[0]) for p in pairs if p[0] != 0]
    us = [m * v / s for v, s in zip(vs, sigma)]
    u = sympy.Matrix.hstack(*us)
    vt = sympy.Matrix.hstack(*vs).T
    return u, sigma, vt


def test_answer_space_is_wide():
    for tier in support.TIERS:
        support.check_answer_space(TOPIC, tier, distinct=5, share=0.45)


def test_tier_shapes():
    support.check_tier_shapes_differ(TOPIC)


def test_exam_shape_and_marks():
    q = make(EXAM, seed=3, difficulty="exam")
    assert [p.answer_kind for p in q.parts] == ["number", "number", "fraction", "matrix"]
    assert [p.follow_through for p in q.parts] == [None, None, "part_a**2/(part_a**2 + part_b**2)", None]
    assert [p.answer_form for p in q.parts] == [None, None, None, "scalar_multiple"]


def test_exam_keys_match_independent_derivation():
    for seed in range(120):
        q = make(EXAM, seed=seed, difficulty="exam")
        m = _matrix(q.stem)
        assert m.shape == (2, 3) and m.rank() == 2
        s1, s2 = _singular(m)
        assert _value(q.parts[0].answer_text) == s1 and _value(q.parts[1].answer_text) == s2
        assert _value(q.parts[2].answer_text) == s1**2 / (s1**2 + s2**2) == s1**2 / sum(x * x for x in m)
        v = sympy.Matrix([row[0] for row in q.parts[3].answer])
        assert m.T * m * v == s1**2 * v and any(v)


def test_exam_grading_and_follow_through():
    support.check_exam_grading(SKILL, EXAM, [2, 2, 3, 3])
    support.check_follow_through(SKILL, EXAM)


def test_exam_vector_accepts_every_multiple_and_rejects_zero():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        vec = [row[0] for row in q.parts[3].answer]
        canon = {p.label: p.answer_text for p in q.parts}
        for scale in (2, -1, 5):
            reply = "(" + ", ".join(str(scale * x) for x in vec) + ")"
            from mathgen.engine import grade

            assert grade(SKILL, q, {**canon, "d": reply}).parts["d"].marks == 3
        assert grade(SKILL, q, {**canon, "d": "(0, 0, 0)"}).parts["d"].marks == 0


def test_exam_answer_space():
    answers = {make(EXAM, seed=s, difficulty="exam").answer_text for s in range(200)}
    assert len(answers) >= 80
