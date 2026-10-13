"""Tests the multi-part exam source of `estimate-financial-total`: an estimate from rounded unit prices, the exact total, and the size of the gap.

The oracle re-derives every part's answer from the rendered stem alone, with exact arithmetic and no code shared
with the generator.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import PRESETS
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr3_estimate_total"
SKILL_ID = "estimate-financial-total"
LABELS = ["a", "b", "c"]
MARKS = [2, 2, 1]
FOLLOWS = [False, False, True]

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def half_up(x) -> int:
    return math.floor(Fraction(x) + Fraction(1, 2))


def cents(x) -> str:
    """Two decimal places, ties away from zero, from an exact fraction."""
    scaled = Fraction(x) * 100
    return f"{half_up(scaled) / 100:.2f}"


def parse(text):
    return [(int(q), Fraction(p)) for q, p in re.findall(r"(\d+) [a-z\- ]+? at \$(\d+\.\d\d) each", text)]


def oracle(text):
    lines = parse(text)
    assert len(lines) == 3
    estimate = sum(q * half_up(p) for q, p in lines)
    exact = sum(q * p for q, p in lines)
    return {"a": str(estimate), "b": cents(exact), "c": cents(abs(estimate - exact))}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FOLLOWS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(p.distractors and len(p.distractors) <= 3 for p in q.parts)
        assert all(p.option_notes for p in q.parts)


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"] and exam[0].params == {"tier": "exam"}
    assert _SKILL.sources[-1] is exam[0]


def test_prompt_shapes_add_to_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]
    diff = src.difficulty if src.difficulty in PRESETS.get(src.topic, {}) else None
    hard = {prompt_shape(make(src.topic, seed=s, difficulty=diff, **src.params)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam source adds no prompt shape beyond the hard tier"


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == [], q.text


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.seed, q.stem) for q in questions if oracle(q.stem) != _canonical(q)]
    assert bad == []


def test_distractors_rejected(questions):
    for q in questions[:120]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.seed, p.label, d)


def test_answer_space(questions):
    assert len({q.answer_text for q in questions}) >= 100
    top = max(sum(1 for q in questions if q.answer_text == t) for t in {q.answer_text for q in questions})
    assert top <= 15


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {label: "999999" for label in LABELS})
        assert result.marks == 0 and not result.correct


def test_follow_through_from_wrong_estimate(questions):
    for q in questions[:120]:
        lines = parse(q.stem)
        exact = sum(n * p for n, p in lines)
        wrong = Fraction(q.parts[0].distractors[0])
        c = abs(wrong - exact)
        result = grade(_SKILL, q, {**_canonical(q), "a": q.parts[0].distractors[0], "c": cents(c)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        assert result.marks == 3 and not result.correct


def test_follow_through_from_wrong_exact_total(questions):
    for q in questions[:60]:
        lines = parse(q.stem)
        estimate = sum(n * half_up(p) for n, p in lines)
        wrong = Fraction(q.parts[1].distractors[0])
        result = grade(_SKILL, q, {**_canonical(q), "b": q.parts[1].distractors[0], "c": cents(abs(estimate - wrong))})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 1
