"""Exam source of `ucv1-inclusion-exclusion-two-sets`: overlap, "exactly one" and its probability."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_two_set_survey"
SKILL_ID = "ucv1-inclusion-exclusion-two-sets"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


def oracle(q):
    """Re-derive the answers from the stem by enumerating the people by region."""
    text = re.sub(r"Year \d+", "Year", q.stem)
    n, a, b, neither = (int(v) for v in re.findall(r"\d+", text))
    at_least_one = n - neither
    both = a + b - at_least_one
    people = ["A&B"] * both + ["A"] * (a - both) + ["B"] * (b - both) + ["none"] * neither
    assert len(people) == n
    exactly_one = sum(1 for p in people if p in ("A", "B"))
    probability = Fraction(exactly_one, n)
    text_p = (
        str(probability.numerator)
        if probability.denominator == 1
        else f"{probability.numerator}/{probability.denominator}"
    )
    return {"a": str(both), "b": str(exactly_one), "c": text_p}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_overlap(questions):
    """The 'forgot the neither group' overlap carried through (b) and (c) earns those marks."""
    checked = 0
    for q in questions:
        text = re.sub(r"Year \d+", "Year", q.stem)
        n, a, b, neither = (int(v) for v in re.findall(r"\d+", text))
        wrong_both = a + b - n  # the overlap if the people in neither are forgotten
        wrong_one = a + b - 2 * wrong_both
        wrong_p = Fraction(wrong_one, n)
        reply = {
            "a": str(wrong_both),
            "b": str(wrong_one),
            "c": f"{wrong_p.numerator}/{wrong_p.denominator}" if wrong_p.denominator != 1 else str(wrong_p.numerator),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True
        checked += 1
    assert checked == len(questions)
