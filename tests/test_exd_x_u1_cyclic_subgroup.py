"""Exam source of `cyclic-group-element-order`: order, index and number of generators of a cyclic subgroup."""

from __future__ import annotations

import re
from math import gcd

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_cyclic_subgroup"
SKILL_ID = "cyclic-group-element-order"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


def oracle(q):
    """Re-derive all three answers from the rendered stem alone, by counting."""
    n = int(re.search(r"(?:Z/|modulo )(\d+)", q.stem).group(1))
    a = int(re.search(r"\[(\d+)\]", q.stem).group(1))
    order = next(k for k in range(1, n + 1) if (k * a) % n == 0)
    index = n // order
    generators = sum(1 for k in range(1, order + 1) if gcd(k, order) == 1)
    return {"a": str(order), "b": str(index), "c": str(generators)}


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
    hard = {prompt_shape(make("cyclic_group_element_order", seed=s, difficulty="hard", kind="sum")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False]


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
        assert result.correct and result.marks == result.max_marks == 5


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_index_from_wrong_order(questions):
    checked = 0
    for q in questions:
        n = int(re.search(r"(?:Z/|modulo )(\d+)", q.stem).group(1))
        wrong = int(q.parts[0].answer_text) * 2
        if n % wrong:
            continue
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "a": str(wrong), "b": str(n // wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True
        checked += 1
    assert checked >= 20


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
