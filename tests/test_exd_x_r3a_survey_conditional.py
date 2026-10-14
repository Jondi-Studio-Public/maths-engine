"""Exam chain of `conditional-probability-counts`: overlap, exactly one, P(A | exactly one), independence check."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_survey_conditional"
SKILL = S.load_skill("conditional-probability-counts")


def _show(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def oracle(q):
    n, a, b, neither = map(int, re.match(r"Of (\d+) .+?, (\d+) .*?, (\d+) .*? and (\d+) do neither\.", q.stem).groups())
    both = a + b - (n - neither)
    only_a, only_b = a - both, b - both
    exactly = only_a + only_b
    return {
        "a": str(both),
        "b": str(exactly),
        "c": _show(Fraction(only_a, exactly)),
        "d": _show(Fraction(both, n) - Fraction(a, n) * Fraction(b, n)),
    }


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_overlap(questions):
    def chain(q):
        n, a, b, neither = map(
            int, re.match(r"Of (\d+) .+?, (\d+) .*?, (\d+) .*? and (\d+) do neither\.", q.stem).groups()
        )
        for wrong in map(int, q.parts[0].distractors):
            exactly = (n - neither) - wrong
            if exactly <= 0 or a - wrong < 0 or b - wrong < 0:
                continue
            p = Fraction(a - wrong, exactly)
            d = Fraction(wrong, n) - Fraction(a, n) * Fraction(b, n)
            if not 0 < p < 1 or wrong == exactly:
                continue
            return {"a": str(wrong), "b": str(exactly), "c": _show(p), "d": _show(d)}, {"b", "c", "d"}
        return None

    S.check_follow_through(SKILL, questions, chain, n=300, min_seen=10)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999", "b": "8888", "c": "7/3", "d": "9/7"})


def test_deterministic():
    S.check_deterministic(TOPIC)
