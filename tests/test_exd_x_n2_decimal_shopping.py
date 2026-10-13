"""Tests for the multi-part exam source of `decimal-arithmetic`.

The oracle reads the masses, prices, note and second price off the rendered prompt and recomputes the three
answers with exact fractions.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_decimal_shopping"
SKILL_ID = "decimal-arithmetic"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _txt(x: Fraction) -> str:
    cents = x * 100
    assert cents.denominator == 1, x
    return f"{int(cents) // 100}.{int(cents) % 100:02d}"


def oracle(q) -> dict[str, str]:
    w = Fraction(re.search(r"([\d.]+) kg of", q.stem).group(1))
    p = Fraction(re.search(r"\$([\d.]+) per kilogram", q.stem).group(1))
    n, e = re.search(r"(?<![\d.$])(\d+) (?!kg)[^$]*?\$([\d.]+) each", q.stem).groups()
    note = int(re.search(r"\$(\d+) note", q.stem).group(1))
    p2 = Fraction(re.search(r"at \$([\d.]+) per kilogram", q.parts[2].prompt).group(1))
    cost = w * p
    change = note - cost - int(n) * Fraction(e)
    return {"a": _txt(cost), "b": _txt(change), "c": _txt(change / p2)}


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
    for q in questions[:120]:
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


def test_follow_through_from_a_wrong_cost(questions):
    checked = 0
    for q in questions[:120]:
        note = int(re.search(r"\$(\d+) note", q.stem).group(1))
        n, e = re.search(r"(?<![\d.$])(\d+) (?!kg)[^$]*?\$([\d.]+) each", q.stem).groups()
        p2 = Fraction(re.search(r"at \$([\d.]+) per kilogram", q.parts[2].prompt).group(1))
        wrong = next(
            (Fraction(d) for d in q.parts[0].distractors if 0 < note - Fraction(d) - int(n) * Fraction(e) < note), None
        )
        if wrong is None:
            continue
        change = note - wrong - int(n) * Fraction(e)
        kg = Fraction(int(change / p2 * 100 + Fraction(1, 2)), 100)
        if (change / p2 * 100) % 1 == Fraction(1, 2):  # an exact tie sits on the tolerance edge
            continue
        if change == wrong or kg == Fraction(q.parts[2].answer_text) or change == Fraction(q.parts[1].answer_text):
            continue
        result = grade(_SKILL, q, {"a": _txt(wrong), "b": _txt(change), "c": _txt(kg)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (q.text, wrong, change)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, change, kg)
        checked += 1
    assert checked >= 30


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
