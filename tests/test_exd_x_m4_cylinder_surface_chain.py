"""Tests the multi-part exam tier of the surface-area skill: radius from volume, surface area, curved share."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_cylinder_surface_chain"
SKILL_ID = "senior-calculate-the-surface-areas-of-standard-three-dimensional"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _dp1(x: float) -> str:
    return f"{math.floor(x * 10 + 0.5) / 10:.1f}"


def _surface_and_share(r: float, h: int, is_open: bool) -> tuple[float, float]:
    sa = math.pi * r * (r + 2 * h) if is_open else 2 * math.pi * r * (r + h)
    return sa, 100 * 2 * math.pi * r * h / sa


def derive(q) -> dict[str, str]:
    """Re-derive every part from the rendered stem alone."""
    volume, height = (int(n) for n in re.findall(r"\d+", q.stem))
    is_open = "open" in q.stem or "no lid" in q.stem
    r = float(_dp1(math.sqrt(volume / (math.pi * height))))
    sa, share = _surface_and_share(r, height, is_open)
    return {"a": _dp1(r), "b": _dp1(sa), "c": _dp1(share)}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        assert derive(q) == _canonical(q), q.text


def test_both_containers_occur(questions):
    assert {"open" in q.stem or "no lid" in q.stem for q in questions} == {True, False}


def test_answer_space(questions):
    values = [q.parts[1].answer_text for q in questions]
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


def test_follow_through_from_wrong_radius(questions):
    for q in questions[:100]:
        height = int(re.findall(r"\d+", q.stem)[1])
        is_open = "open" in q.stem or "no lid" in q.stem
        r = float(q.parts[0].answer_text) + 0.7
        sa, share = _surface_and_share(r, height, is_open)
        result = grade(_SKILL, q, {"a": _dp1(r), "b": _dp1(sa), "c": _dp1(share)})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True
        assert result.marks == 4 and not result.correct


def test_prompt_shapes_new():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    hard = {prompt_shape(make(hard_src.topic, seed=s, **hard_src.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard
