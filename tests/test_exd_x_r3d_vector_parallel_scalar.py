"""Exam chain of `senior-represent-a-scalar-multiple-of-a-vector`: k, the multiplier m, then the squared length."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_vector_parallel_scalar"
SKILL = S.load_skill("senior-represent-a-scalar-multiple-of-a-vector")


def _read(q):
    u = tuple(int(v) for v in re.search(r"u = \((-?\d+), (-?\d+)\)", q.stem).groups())
    w = tuple(int(v) for v in re.search(r"w = \((-?\d+), (-?\d+)\)", q.stem).groups())
    e = tuple(int(v) for v in re.search(r"e = \((-?\d+), (-?\d+)\)", q.stem).groups())
    sign = -1 if "w - ku" in q.stem else 1
    return u, w, e, sign


def oracle(q):
    u, w, e, sign = _read(q)
    k = sympy.Symbol("k")
    vec = [sign * (k * u[i] - w[i]) for i in range(2)]
    (root,) = sympy.solve(vec[0] * e[1] - vec[1] * e[0], k)
    v = [c.subs(k, root) for c in vec]
    m = sympy.Rational(v[0], e[0])
    assert v[1] == m * e[1]
    return {"a": str(root), "b": str(m), "c": str(v[0] ** 2 + v[1] ** 2)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_both_orders_occur(questions):
    assert {"w - ku" in q.stem for q in questions} == {True, False}


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_k(questions):
    def chain(q):
        u, w, e, sign = _read(q)
        k = int(q.parts[0].distractors[0])
        vec = [sign * (k * u[i] - w[i]) for i in range(2)]
        if vec[0] % e[0]:
            return None
        m = vec[0] // e[0]
        length2 = m * m * (e[0] ** 2 + e[1] ** 2)
        if m == 0 or m in (k,) or length2 in (k, m):
            return None
        return {"a": str(k), "b": str(m), "c": str(length2)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=120)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "99", "b": "98", "c": "97"})


def test_deterministic():
    S.check_deterministic(TOPIC)
