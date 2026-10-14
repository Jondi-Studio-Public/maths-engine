"""Exam chain of `matrix-determinant-2x2`: det A, det B from det(AB), the unknown entry, det(cA^-1 B)."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_determinant_chain"
SKILL = S.load_skill("matrix-determinant-2x2")


def _matrices(q):
    a = re.search(r"A = \[(-?\d+) (-?\d+); (-?\d+) (-?\d+)\]", q.stem)
    b = re.search(r"B = \[x (-?\d+); (-?\d+) (-?\d+)\]", q.stem)
    d = re.search(r"(?:det\(AB\) = |determinant )(-?\d+)", q.stem)
    return tuple(int(g) for g in (*a.groups(), *b.groups(), d.group(1)))


def oracle(q):
    a1, a2, a3, a4, b2, b3, b4, d = _matrices(q)
    x = sympy.Symbol("x")
    big_a, big_b = sympy.Matrix([[a1, a2], [a3, a4]]), sympy.Matrix([[x, b2], [b3, b4]])
    (xv,) = sympy.solve((big_a * big_b).det() - d, x)
    c = int(re.search(r"find det\((\d)[AB]", q.text).group(1))
    final = (c * big_a.inv() * big_b.subs(x, xv)).det()
    return {"a": str(big_a.det()), "b": str(big_b.subs(x, xv).det()), "c": str(xv), "d": str(final)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_determinant(questions):
    def chain(q):
        a1, a2, a3, a4, b2, b3, b4, d = _matrices(q)
        c = int(re.search(r"find det\((\d)[AB]", q.text).group(1))
        for wrong in q.parts[0].distractors:
            wrong = Fraction(wrong)
            if wrong == 0 or Fraction(d) / wrong % 1:
                continue
            det_b = Fraction(d) / wrong
            x = (det_b + b2 * b3) / b4
            final = c * c * det_b / wrong
            if x % 1 or final % 1 or len({wrong, det_b, x, final}) < 4:
                continue  # a reply that repeats an earlier one earns no follow-through
            return {"a": str(wrong), "b": str(det_b), "c": str(x), "d": str(final)}, {"b", "c", "d"}
        return None

    S.check_follow_through(SKILL, questions, chain, n=300, min_seen=3)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999", "b": "8888", "c": "7777", "d": "6666"})


def test_deterministic():
    S.check_deterministic(TOPIC)
