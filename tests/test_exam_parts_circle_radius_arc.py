"""Tests the multi-part exam tier of `circle-area-to-diameter-and-circumference`: radius, circumference, arc."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills

TOPIC = "exam_parts_circle_radius_arc"
SKILL_ID = "circle-area-to-diameter-and-circumference"
SEEDS = range(120)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _dp1(x: float) -> str:
    return f"{math.floor(x * 10 + 0.5) / 10:.1f}"


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem alone."""
    area = float(re.search(r"(\d+\.\d\d) cm²", q.stem).group(1))
    theta = int(re.search(r"(\d+)°", q.stem).group(1))
    r = float(_dp1(math.sqrt(area / math.pi)))
    return {
        "a": _dp1(r),
        "b": _dp1(2 * math.pi * r),
        "c": _dp1(theta / 360 * 2 * math.pi * r),
    }


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


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_distractors_rejected(questions):
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                reply = {**canon, p.label: d}
                result = grade(_SKILL, q, reply)
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_radius(questions):
    checked = 0
    for q in questions:
        a_wrong = q.parts[0].distractors[0]
        r = float(a_wrong)
        b_wrong = _dp1(2 * math.pi * r)
        theta = int(re.search(r"(\d+)°", q.stem).group(1))
        c_wrong = _dp1(theta / 360 * 2 * math.pi * r)
        result = grade(_SKILL, q, {"a": a_wrong, "b": b_wrong, "c": c_wrong})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True
        assert result.marks == 3 and result.marks < result.max_marks
        assert not result.correct
        checked += 1
    assert checked == len(SEEDS)


def test_follow_through_offset_radius(questions):
    for q in questions[:40]:
        r = float(q.parts[0].answer_text) + 0.7
        theta = int(re.search(r"(\d+)°", q.stem).group(1))
        result = grade(
            _SKILL,
            q,
            {"a": _dp1(r), "b": _dp1(2 * math.pi * r), "c": _dp1(theta / 360 * 2 * math.pi * r)},
        )
        assert result.marks == 3
        assert result.parts["b"].follow_through and result.parts["c"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions:
        result = grade(_SKILL, q, {"a": "999.9", "b": "1.1", "c": "2.2"})
        assert result.marks == 0 and not result.correct


def test_variety():
    stems = {make(TOPIC, seed=s).stem for s in range(60)}
    assert len(stems) >= 40
    assert len({make(TOPIC, seed=s).parts[0].answer_text for s in range(60)}) >= 30


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()
