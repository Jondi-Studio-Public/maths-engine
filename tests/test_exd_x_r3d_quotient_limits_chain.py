"""Exam chain of `senior-examine-the-behaviour-of-the-difference-quotient-frac`: a, b, then x for a stated limit."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_quotient_limits_chain"
SKILL = S.load_skill("senior-examine-the-behaviour-of-the-difference-quotient-frac")
_PATTERNS = (
    r"approaches (-?\d+) as h approaches 0 when x = (-?\d+), and approaches (-?\d+) when x = (-?\d+)",
    r"has limit (-?\d+) at x = (-?\d+) and limit (-?\d+) at x = (-?\d+)",
    r"is (-?\d+) when x = (-?\d+) and (-?\d+) when x = (-?\d+)",
)


def _read(q):
    for pattern in _PATTERNS:
        found = re.search(pattern, q.stem)
        if found:
            l1, x1, l2, x2 = (int(v) for v in found.groups())
            m = int(re.search(r"(?:approach|equals) (-?\d+)", q.parts[2].prompt).group(1))
            return l1, x1, l2, x2, m
    raise AssertionError(q.stem)


def oracle(q):
    l1, x1, l2, x2, m = _read(q)
    # The limit is 2 a x + b: solve the two stated limits, then invert for x.
    a = Fraction(l2 - l1, 2 * (x2 - x1))
    b = l1 - 2 * a * x1
    x = (m - b) / (2 * a)
    return {"a": str(a), "b": str(b), "c": str(x)}


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


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_oracle_by_numeric_difference_quotient(questions):
    # Independent of the 2ax + b formula: difference quotients at a tiny h for the recovered a and b.
    for q in questions[:60]:
        l1, x1, l2, x2, m = _read(q)
        a, b, x3 = (Fraction(q.parts[i].answer_text) for i in range(3))
        h = Fraction(1, 10**9)

        def quotient(x):
            return (a * (x + h) ** 2 + b * (x + h) - a * x**2 - b * x) / h

        for x, want in ((x1, l1), (x2, l2), (x3, m)):
            assert abs(quotient(x) - want) < Fraction(1, 10**6)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_a(questions):
    def chain(q):
        l1, x1, l2, x2, m = _read(q)
        a = int(q.parts[0].distractors[0])
        b = l1 - 2 * a * x1
        if a == 0 or b == 0:
            return None  # a zero earlier reply carries no follow-through
        x3 = Fraction(m - b, 2 * a)
        if x3.denominator != 1 or len({a, b, int(x3)}) < 3:
            return None
        return {"a": str(a), "b": str(b), "c": str(int(x3))}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=300, minimum=25)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "99", "b": "98", "c": "97"})


def test_deterministic():
    S.check_deterministic(TOPIC)
