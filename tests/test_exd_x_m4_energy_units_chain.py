"""Tests the multi-part exam tier of the energy-units skill: kWh from watts and time, kilojoules, cost."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_energy_units_chain"
SKILL_ID = "senior-convert-from-one-unit-of-energy-to-another"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _half_up(x: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(int(x * scale + Fraction(1, 2)), scale)


def _inputs(stem: str):
    watts = int(re.search(r"(\d+) W", stem).group(1))
    hours = re.search(r"(\d+) hours?", stem)
    mins = re.search(r"(\d+) minutes", stem)
    minutes = (int(hours.group(1)) * 60 if hours else 0) + (int(mins.group(1)) if mins else 0)
    rate = int(re.search(r"(\d+) (?:cents|c) ", stem).group(1))
    return watts, minutes, rate


def derive(q) -> dict[str, str]:
    watts, minutes, rate = _inputs(q.stem)
    kwh = Fraction(watts, 1000) * Fraction(minutes, 60)
    cost = _half_up(kwh * rate / 100, 2)
    return {"a": f"{float(kwh):.2f}", "b": str(int(kwh * 3600)), "c": f"{float(cost):.2f}"}


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


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


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


def test_follow_through_from_wrong_energy(questions):
    for q in questions[:100]:
        _watts, _minutes, rate = _inputs(q.stem)
        wrong = Fraction(q.parts[0].answer_text) + Fraction(37, 10)
        reply = {
            "a": f"{float(wrong):.2f}",
            "b": str(int(wrong * 3600)),
            "c": f"{float(_half_up(wrong * rate / 100, 2)):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        assert result.marks == 3 and not result.correct


def test_prompt_shapes_new():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    hard = {prompt_shape(make(hard_src.topic, seed=s, **hard_src.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()
