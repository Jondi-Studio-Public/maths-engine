"""Tests the multi-part exam tier of the equal-chords skill: angle from a chord, angle for the run, joining chord."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_equal_chords_chain"
SKILL_ID = "senior-chords-of-equal-length-subtend-equal-angles-at"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _one_dp(x: float) -> str:
    return f"{math.floor(x * 10 + 0.5) / 10:.1f}"


def _chords_and_inputs(stem: str):
    radius = float(re.search(r"radius (\d+(?:\.\d)?)", stem).group(1))
    length = float(re.search(r"(\d+\.\d\d) (?:cm|m)", stem).group(1))
    n = 4 if "DE" in stem else 3 if "CD" in stem else 2
    return radius, length, n


def derive(q) -> dict[str, str]:
    radius, length, n = _chords_and_inputs(q.stem)
    theta = round(2 * math.degrees(math.asin(length / (2 * radius))))
    total = n * theta
    return {"a": str(theta), "b": str(total), "c": _one_dp(2 * radius * math.sin(math.radians(total) / 2))}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        assert derive(q) == _canonical(q), q.text


def test_answer_space(questions):
    values = [q.parts[2].answer_text for q in questions]
    assert len(set(values)) >= 100
    assert max(values.count(v) for v in set(values)) / len(values) <= 0.05


def test_all_run_lengths_occur(questions):
    assert {_chords_and_inputs(q.stem)[2] for q in questions} == {2, 3, 4}


def test_full_marks_and_distractors_rejected(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_angle(questions):
    for q in questions[:100]:
        radius, _length, n = _chords_and_inputs(q.stem)
        theta = int(q.parts[0].answer_text)
        wrong = theta + 3
        total = n * wrong
        reply = {"a": str(wrong), "b": str(total), "c": _one_dp(2 * radius * math.sin(math.radians(total) / 2))}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
            if reply[label] != q.parts["abc".index(label)].answer_text:
                assert result.parts[label].follow_through is True
        assert result.marks == 3 and not result.correct


def test_prompt_shapes_new():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    hard = {prompt_shape(make(hard_src.topic, seed=s, **hard_src.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard
