"""Tests the multi-part exam tier of the roots-of-unity skill: angle, distance and triangle area for two roots."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_roots_of_unity_chain"
SKILL_ID = "senior-determine-and-examine-the-n-th-roots-of"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _half_up(x: float, places: int) -> float:
    return math.floor(x * 10**places + 0.5) / 10**places


def _inputs(stem: str):
    n = int(re.search(r"z\^(\d+) = 1", stem).group(1))
    j = int(re.search(r"(?:represent|P marks|P =) z_(\d+)", stem).group(1))
    k = int(re.search(r"(?:and z_|Q marks z_|Q = z_)(\d+)", stem).group(1))
    u = int(
        re.search(r"(?:(\d+) cm to 1 unit|1 unit (?:represents|is) (\d+) cm)", stem).group(1)
        or re.search(r"1 unit (?:represents|is) (\d+) cm", stem).group(1)
    )
    return n, j, k, u


def derive(q) -> dict[str, str]:
    n, j, k, u = _inputs(q.stem)
    step = min((k - j) % n, (j - k) % n)
    angle = _half_up(360 * step / n, 1)
    chord = 2 * u * math.sin(math.radians(360 * step / n) / 2)
    area = u * u * math.sin(math.radians(360 * step / n)) / 2
    return {"a": f"{angle:.1f}", "b": f"{_half_up(chord, 2):.2f}", "c": f"{_half_up(area, 2):.2f}"}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
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
        assert result.correct and result.marks == result.max_marks == 6
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_angle(questions):
    for q in questions[:100]:
        _n, _j, _k, u = _inputs(q.stem)
        wrong = float(q.parts[0].answer_text) - 6.0
        rad = math.radians(wrong)
        reply = {
            "a": f"{wrong:.1f}",
            "b": f"{_half_up(2 * u * math.sin(rad / 2), 2):.2f}",
            "c": f"{_half_up(u * u * math.sin(rad) / 2, 2):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
            if reply[label] != q.parts["abc".index(label)].answer_text:
                assert result.parts[label].follow_through is True
        assert not result.correct


def test_prompt_shapes_new():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    hard = {prompt_shape(make(hard_src.topic, seed=s, **hard_src.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard
