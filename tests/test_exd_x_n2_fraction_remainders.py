"""Tests for the multi-part exam source of `fraction-arithmetic`.

The oracle reads the two fractions and the amount left from the rendered stem and recomputes the fraction left,
the original amount and the amount used in the second step with exact fractions.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_fraction_remainders"
SKILL_ID = "fraction-arithmetic"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _read(q):
    f1, f2 = (Fraction(m) for m in re.findall(r"\d+/\d+", q.stem)[:2])
    m = re.search(r"\$([\d,]+)|([\d,]+) (?:litres|metres)", q.stem)
    x = int((m.group(1) or m.group(2)).replace(",", ""))
    return f1, f2, x


def oracle(q) -> dict[str, str]:
    f1, f2, x = _read(q)
    left = (1 - f1) * (1 - f2)
    whole = x / left
    second = whole * (1 - f1) * f2
    assert whole.denominator == 1 and second.denominator == 1
    return {"a": f"{left.numerator}/{left.denominator}", "b": str(int(whole)), "c": str(int(second))}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired_last_with_exam_stage():
    source = _SKILL.sources[-1]
    assert source.topic == TOPIC and source.difficulty == "exam"
    assert list(source.stages) == ["exam"] and dict(source.params) == {"tier": "exam"}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == [], seed


def test_prompt_shapes_new_against_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(_HARD.topic, seed=s, difficulty="hard", **dict(_HARD.params or {}))) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_oracle_agrees_over_300_seeds(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_key_accepted_and_every_distractor_rejected(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)


def test_tagged_distractors_carry_values(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)


def test_follow_through_from_a_wrong_fraction(questions):
    checked = 0
    for q in questions[:150]:
        f1, f2, x = _read(q)
        for d in q.parts[0].distractors:
            wrong = Fraction(d)
            whole = x / wrong
            second = whole * (1 - f1) * f2
            if whole.denominator != 1 or second.denominator != 1:
                continue
            if len({int(whole), int(second), int(q.parts[1].answer_text), int(q.parts[2].answer_text)}) < 4:
                continue
            result = grade(_SKILL, q, {"a": d, "b": str(int(whole)), "c": str(int(second))})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (q.text, d)
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, d)
            checked += 1
            break
    assert checked >= 30


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
