"""Tests for the multi-part exam source of `senior-use-leading-digit-approximation-to-obtain-estimates-of`.

The oracle reads the numbers off the rendered story, rounds each to its leading digit, evaluates the estimate and
the exact value, and recomputes the percentage error, all with exact integers and fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_leading_digit_error"
SKILL_ID = "senior-use-leading-digit-approximation-to-obtain-estimates-of"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _lead(n: int) -> int:
    place = 10 ** (len(str(n)) - 1)
    return math.floor(Fraction(n, place) + Fraction(1, 2)) * place


def _round_half_up(x: Fraction) -> int:
    return math.floor(x + Fraction(1, 2))


def oracle(q) -> dict[str, str]:
    nums = [int(x) for x in re.findall(r"\d+", q.stem)]
    if len(nums) == 4:
        a, b, c, d = nums
        est, exact = _lead(a) * _lead(b) + _lead(c) * _lead(d), a * b + c * d
    else:
        a, b, c = nums
        assert (_lead(a) * _lead(b)) % _lead(c) == 0 and (a * b) % c == 0
        est, exact = _lead(a) * _lead(b) // _lead(c), a * b // c
    pct = _round_half_up(Fraction(100 * abs(est - exact), exact))
    return {"a": str(est), "b": str(exact), "c": str(pct)}


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
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]


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


def test_both_structures_occur(questions):
    kinds = {len(re.findall(r"\d+", q.stem)) for q in questions}
    assert kinds == {3, 4}


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_key_accepted_and_every_distractor_rejected(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)


def test_tagged_distractors_carry_values(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)


def test_follow_through_from_a_wrong_estimate(questions):
    checked = 0
    for q in questions[:150]:
        exact = int(q.parts[1].answer_text)
        for wrong in q.parts[0].distractors:
            w = int(wrong)
            if not 0 < abs(w - exact) * 100 < exact * 300 or w == exact:
                continue
            pct = _round_half_up(Fraction(100 * abs(w - exact), exact))
            if pct == 0 or str(pct) in (wrong, q.parts[2].answer_text):
                continue
            result = grade(_SKILL, q, {"a": wrong, "b": str(exact), "c": str(pct)})
            assert result.parts["a"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, wrong, pct)
            checked += 1
            break
    assert checked >= 40


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
