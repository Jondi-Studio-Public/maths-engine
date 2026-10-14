"""Tests the exam source of `circle-reverse-from-area`: radius, circumference and a post count, as a chain."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_m1_circle_posts"
SKILL_ID = "circle-reverse-from-area"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_PI = Fraction(314159, 100000)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _dp1(x: float) -> str:
    return f"{math.floor(x * 10 + 0.5) / 10:.1f}"


def _oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered prompt alone, working as a learner does."""
    area = float(re.search(r"(\d+\.\d\d) m²", q.text).group(1))
    spacing = float(re.search(r"every ([\d.]+) m", q.text).group(1))
    r = float(_dp1(math.sqrt(area / float(_PI))))
    c = float(_dp1(2 * float(_PI) * r))
    return {"a": _dp1(r), "b": _dp1(2 * float(_PI) * r), "c": str(math.ceil(c / spacing))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape():
    q = _exam(1)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [p.marks for p in q.parts] == [2, 1, 2]
    assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert _oracle(q) == _canonical(q), (seed, q.text)


def test_prompt_does_not_contain_a_key():
    for seed in range(300):
        q = _exam(seed)
        numbers = set(re.findall(r"\d+(?:\.\d+)?", q.stem + " ".join(p.prompt for p in q.parts)))
        assert not numbers & {p.answer_text for p in q.parts}, (seed, q.text)


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(120)]


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_distractors_rejected(questions):
    shown = 0
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                shown += 1
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 300


def test_follow_through_from_wrong_radius(questions):
    for q in questions:
        r = float(q.parts[0].distractors[0])
        spacing = float(re.search(r"every ([\d.]+) m", q.text).group(1))
        b = _dp1(2 * float(_PI) * r)
        c = str(math.ceil(float(b) / spacing))
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": b, "c": c})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, r)
        assert result.parts["b"].follow_through is True
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, r, b, c)
        assert not result.correct


def test_follow_through_from_wrong_circumference(questions):
    for q in questions[:60]:
        spacing = float(re.search(r"every ([\d.]+) m", q.text).group(1))
        b = float(q.parts[1].answer_text) + 3.0
        result = grade(_SKILL, q, {"a": q.parts[0].answer_text, "b": f"{b:.1f}", "c": str(math.ceil(b / spacing))})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks
