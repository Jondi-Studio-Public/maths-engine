"""Tests for the multi-part exam source of `senior-use-currency-exchange-rates-to-determine-the-cost`.

The oracle reads the rendered prompt only: the dollars paid, the buying and selling rates and the foreign amount
spent, then recomputes the three answers with exact fractions.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_currency_round_trip"
SKILL_ID = "senior-use-currency-exchange-rates-to-determine-the-cost"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round_cents(x: Fraction) -> Fraction:
    return Fraction(int(x * 100 + Fraction(1, 2)), 100)


def _txt(x: Fraction) -> str:
    cents = int(_round_cents(x) * 100)
    return f"{cents // 100}.{cents % 100:02d}"


def oracle(q) -> dict[str, str]:
    aud = int(re.search(r"A\$([\d,]+)(?!\d| =)", q.stem).group(1).replace(",", ""))
    rates = [Fraction(m) for m in re.findall(r"A\$1 = [^\d]*(\d+\.\d\d)", q.stem)]
    assert len(rates) == 2
    r_buy, r_sell = rates
    spent = int(re.search(r"spends [^\d]*([\d,]+) on", q.stem).group(1).replace(",", ""))
    foreign = aud * r_buy
    back = _round_cents((foreign - spent) / r_sell)
    return {"a": _txt(foreign), "b": _txt(back), "c": _txt(aud - back)}


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
        assert [p.marks for p in q.parts] == [2, 3, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


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
        assert result.correct and result.marks == result.max_marks == 7
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)


def test_tagged_distractors_carry_values(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)
            assert set(p.distractor_values) <= set(p.distractors)


def test_follow_through_from_a_wrong_part_a(questions):
    checked = 0
    for q in questions[:120]:
        aud = int(re.search(r"A\$([\d,]+)(?!\d| =)", q.stem).group(1).replace(",", ""))
        r_sell = Fraction(re.findall(r"A\$1 = [^\d]*(\d+\.\d\d)", q.stem)[1])
        spent = int(re.search(r"spends [^\d]*([\d,]+) on", q.stem).group(1).replace(",", ""))
        a_wrong = next(
            (Fraction(d) for d in q.parts[0].distractors if spent + 20 < Fraction(d) < 2 * aud * Fraction(7, 10)), None
        )
        if a_wrong is None:
            continue
        b_wrong = _round_cents((a_wrong - spent) / r_sell)
        if b_wrong == a_wrong or b_wrong >= aud:  # no follow-through for a repeat or a negative cost
            continue
        checked += 1
        c_wrong = aud - b_wrong
        result = grade(_SKILL, q, {"a": _txt(a_wrong), "b": _txt(b_wrong), "c": _txt(c_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 3 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
        assert result.marks == 5 and not result.correct
    assert checked >= 20


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
