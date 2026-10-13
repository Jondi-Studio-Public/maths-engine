"""Exam source of `hookes-law`: a spring measured under two loads, natural length unknown."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import round_half_up
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_spring_two_loads"
SKILL_ID = "hookes-law"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


def _read(q):
    l1, l2 = (Fraction(v) for v in re.findall(r"(\d+\.\d\d) m\b", q.stem))
    f1, f2 = (Fraction(v) for v in re.findall(r"(\d+\.\d) N", q.stem))
    l3 = Fraction(re.search(r"(\d+\.\d\d) m\b", q.parts[2].prompt).group(1))
    return l1, f1, l2, f2, l3


def _fixed(value, places):
    scaled = value * 10**places
    assert scaled.denominator == 1
    digits = str(abs(scaled.numerator)).rjust(places + 1, "0")
    return f"{digits[:-places]}.{digits[-places:]}"


def oracle(q):
    """Fit the line F = k (l - l0) through the two measurements, then read it at the third length."""
    l1, f1, l2, f2, l3 = _read(q)
    k = (f2 - f1) / (l2 - l1)
    l0 = l1 - f1 / k
    return {"a": str(int(k)), "b": _fixed(l0, 2), "c": _fixed(k * (l3 - l0), 1)}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert [s.stages for s in exam] == [["exam"]]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    existing = {prompt_shape(make("engineering", seed=s, kinds=["hookes_law"])) for s in range(60)}
    assert len(exam) >= 3
    assert exam - existing


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * len(answers)


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_stiffness(questions):
    """A stiffness found from the lengths instead of the extensions, carried through (b) and (c), earns marks."""
    checked = 0
    for q in questions:
        l1, f1, l2, f2, l3 = _read(q)
        k_wrong = int(q.parts[0].answer_text) + 10
        l0 = l1 - f1 / k_wrong
        b_wrong = _fixed(round_half_up(l0, 2), 2)
        l0_shown = Fraction(b_wrong)
        c_exact = k_wrong * (l3 - l0_shown)
        c_wrong = _fixed(round_half_up(c_exact, 1), 1)
        if l0_shown <= 0 or c_exact <= 0:
            continue
        result = grade(_SKILL, q, {"a": str(k_wrong), "b": b_wrong, "c": c_wrong})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2
        assert result.parts["b"].follow_through is (b_wrong != q.parts[1].answer_text)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is (c_wrong != q.parts[2].answer_text)
        checked += 1
    assert checked >= 100


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
