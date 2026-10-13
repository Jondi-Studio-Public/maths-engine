"""Tests the multi-part exam tier of the metric-area-units skill: hectares, cost per hectare, percentage of a farm."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_area_units_land_chain"
SKILL_ID = "senior-convert-between-metric-units-of-area-and-other"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _inputs(stem: str):
    km = Fraction(re.search(r"(\d+(?:\.\d)?) km(?!²)", stem).group(1))
    metres = int(re.search(r"(\d+) m(?![²a-z])", stem).group(1))
    farm_km2 = Fraction(re.search(r"(\d+(?:\.\d)?) km²", stem).group(1))
    price = int(re.search(r"\$(\d+) per hectare", stem).group(1))
    return km, metres, farm_km2, price


def _half_up(x: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(int(x * scale + Fraction(1, 2)), scale)


def derive(q) -> dict[str, str]:
    km, metres, farm, price = _inputs(q.stem)
    square_metres = km * 1000 * metres
    hectares = square_metres / 10000
    farm_hectares = farm * 100
    percent = _half_up(hectares / farm_hectares * 100, 1)
    ha_text = str(int(hectares)) if hectares.denominator == 1 else f"{float(hectares):.1f}"
    return {"a": ha_text, "b": str(int(hectares * price)), "c": f"{float(percent):.1f}"}


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


def test_follow_through_from_wrong_area(questions):
    for q in questions[:100]:
        _km, _m, farm, price = _inputs(q.stem)
        wrong = Fraction(q.parts[0].answer_text) + Fraction(7, 2)
        percent = _half_up(wrong / (farm * 100) * 100, 1)
        reply = {"a": f"{float(wrong):.1f}", "b": f"{float(wrong * price):.2f}", "c": f"{float(percent):.1f}"}
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
