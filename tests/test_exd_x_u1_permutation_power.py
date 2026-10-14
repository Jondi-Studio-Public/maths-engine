"""Exam source of `permutation-composition`: the composite, its order, and the image under a high power."""

from __future__ import annotations

import re

import pytest
from sympy.combinatorics import Permutation

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_permutation_power"
SKILL_ID = "permutation-composition"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


def oracle(q):
    """Re-derive the three answers from the rendered text with SymPy's permutation group code."""
    p_text, q_text = re.search(r"p = \(([\d, ]+)\).*?q = \(([\d, ]+)\)", q.stem).groups()
    p = [int(v) - 1 for v in p_text.split(",")]
    qq = [int(v) - 1 for v in q_text.split(",")]
    # SymPy multiplies left to right ((a*b)(i) = b(a(i))), so "p after q" is q*p.
    r = Permutation(qq) * Permutation(p)
    image = "(" + ", ".join(str(r(i) + 1) for i in range(len(p))) + ")"
    c_prompt = q.parts[2].prompt
    j = int(re.search(r"image of (\d+)", c_prompt).group(1))
    power = int(re.search(r"\^(\d+)", c_prompt).group(1))
    return {"a": image, "b": str(r.order()), "c": str((r**power)(j - 1) + 1)}


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
    hard = {prompt_shape(make("permutation_composition", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["ordered_list", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]  # from a wrong (a)


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


def test_parts_marked_independently(questions):
    """With no follow-through, a wrong (a) does not cost the marks for a correct (b) and (c)."""
    for q in questions[:60]:
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "a": q.parts[0].distractors[0]})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["c"].marks == 2


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
