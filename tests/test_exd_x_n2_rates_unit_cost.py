"""Tests for the multi-part exam source of `senior-use-rates-to-make-comparisons`.

The oracle reads the rendered stem and the quantity in part (c) and recomputes both unit rates and the difference
over the quantity with exact fractions.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_rates_unit_cost"
SKILL_ID = "senior-use-rates-to-make-comparisons"
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


def _rates(q):
    """(rate_a, rate_b, quantity in the rate's unit) read from the rendered text."""
    c_prompt = q.parts[2].prompt
    if "Car A" in q.stem:
        fa = Fraction(re.search(r"Car A uses ([\d.]+) L", q.stem).group(1))
        fb = Fraction(re.search(r"Car B uses ([\d.]+) L", q.stem).group(1))
        price = Fraction(re.search(r"\$([\d.]+)", q.stem).group(1))
        km = int(re.search(r"trip of (\d+) km", c_prompt).group(1))
        return fa * price, fb * price, Fraction(km, 100)
    ga, ca, gb, cb = (Fraction(x) for x in re.findall(r"\d+(?:\.\d+)?", q.stem))
    kg = int(re.search(r"needs (\d+) kg", c_prompt).group(1))
    return ca / ga * 100, cb / gb * 100, Fraction(kg * 10)


def oracle(q) -> dict[str, str]:
    ra, rb, qty = _rates(q)
    ra, rb = _round_cents(ra), _round_cents(rb)
    return {"a": _txt(ra), "b": _txt(rb), "c": _txt(abs(ra - rb) * qty)}


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


def test_both_settings_occur(questions):
    assert {"Car A" in q.stem for q in questions} == {True, False}


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


def test_follow_through_from_a_wrong_rate(questions):
    checked = 0
    for q in questions[:120]:
        ra, rb, qty = _rates(q)
        wrong = Fraction(q.parts[0].distractors[0])
        rb_c = _round_cents(rb)
        c_wrong = _round_cents(abs(wrong - rb_c) * qty)
        if wrong <= 0 or wrong == rb_c or c_wrong in (wrong, rb_c) or c_wrong == Fraction(q.parts[2].answer_text):
            continue
        result = grade(_SKILL, q, {"a": _txt(wrong), "b": q.parts[1].answer_text, "c": _txt(c_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, wrong, c_wrong)
        checked += 1
    assert checked >= 40


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
