"""Tests the multi-part exam tier of the rate-units skill: speed in km/h, speed in m/s, time for a further distance."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_rate_units_chain"
SKILL_ID = "senior-convert-between-units-for-rates-for-example-km"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _half_up(x: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(int(x * scale + Fraction(1, 2)), scale)


def _inputs(q):
    km = Fraction(re.search(r"(\d+(?:\.\d+)?) km", q.stem).group(1))
    minutes = int(re.search(r"in (\d+) minutes", q.stem).group(1))
    further = int(re.search(r"further (\d+) m", q.parts[2].prompt).group(1))
    return km, minutes, further


def derive(q) -> dict[str, str]:
    km, minutes, further = _inputs(q)
    kmh = _half_up(km / Fraction(minutes, 60), 1)
    ms = _half_up(kmh * 1000 / 3600, 1)
    return {"a": f"{float(kmh):.1f}", "b": f"{float(ms):.1f}", "c": str(int(_half_up(further / ms, 0)))}


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


def test_follow_through_from_wrong_speed(questions):
    for q in questions[:100]:
        _km, _minutes, further = _inputs(q)
        wrong = Fraction(q.parts[0].answer_text) + 3
        ms = _half_up(wrong * 1000 / 3600, 1)
        reply = {"a": f"{float(wrong):.1f}", "b": f"{float(ms):.1f}", "c": str(int(_half_up(further / ms, 0)))}
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
