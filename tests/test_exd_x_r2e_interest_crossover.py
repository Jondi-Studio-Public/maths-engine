"""Tests for the multi-part exam source of `senior-use-technology-to-compare-numerically-and-graphically-the`
(`exd_x_r2e_interest_crossover`): a compound rate recovered from a first-year balance, a balance, a simple-interest
balance, and the year one account overtakes the other."""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_interest_crossover"
SKILL_ID = "senior-use-technology-to-compare-numerically-and-graphically-the"
HARD = ("pv0b_interest_crossover", {"mode": "different_principals"})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _half_up(x: Fraction) -> int:
    return int(x + Fraction(1, 2)) if x >= 0 else -int(-x + Fraction(1, 2))


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem and the part prompts alone, in exact arithmetic."""
    money = [int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", q.stem)]
    p_simple, yearly, p_comp, first_year = money
    n = int(re.search(r"end of year (\d+)", q.parts[1].prompt).group(1))
    growth = Fraction(first_year, p_comp)
    rate = (growth - 1) * 100
    balance_n = _half_up(p_comp * growth**n)
    simple_n = p_simple + yearly * n
    year, bal = 0, Fraction(p_comp)
    while not bal > simple_n:
        bal *= growth
        year += 1
    return {
        "a": f"{float(rate):.1f}",
        "b": str(balance_n),
        "c": str(simple_n),
        "d": str(year),
    }


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_add_to_the_hard_tier():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    tuples = collections.Counter(tuple(p.answer_text for p in q.parts) for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 8


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_a_wrong_rate(questions):
    """A wrong (a) used correctly earns (b) and (d), whatever it is."""
    checked = 0
    for q in questions[:120]:
        p_comp = int(re.findall(r"\$([\d,]+)", q.stem)[2].replace(",", ""))
        n = int(re.search(r"end of year (\d+)", q.parts[1].prompt).group(1))
        w = int(q.parts[2].answer_text)
        rate = Fraction(q.parts[0].answer_text) + Fraction(1, 2)  # 0.5 percentage points too high
        growth = 1 + rate / 100
        b_wrong = _half_up(p_comp * growth**n)
        year, bal = 0, Fraction(p_comp)
        while not bal > w:
            bal *= growth
            year += 1
        result = grade(_SKILL, q, {"a": str(rate), "b": str(b_wrong), "c": str(w), "d": str(year)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 1
        assert result.parts["d"].marks == 3
        # When the wrong rate happens to give the right year the part is simply correct.
        assert result.parts["d"].follow_through or str(year) == q.parts[3].answer_text
        assert result.marks == 6 and not result.correct
        checked += 1
    assert checked == 120


def test_follow_through_from_a_wrong_wei_balance(questions):
    for q in questions[:60]:
        canon = _canonical(q)
        p_comp = int(re.findall(r"\$([\d,]+)", q.stem)[2].replace(",", ""))
        growth = Fraction(int(re.findall(r"\$([\d,]+)", q.stem)[3].replace(",", "")), p_comp)
        w_wrong = int(canon["c"]) + 3000
        year, bal = 0, Fraction(p_comp)
        while not bal > w_wrong:
            bal *= growth
            year += 1
        result = grade(_SKILL, q, {**canon, "c": str(w_wrong), "d": str(year)})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 3
        assert result.parts["d"].follow_through or str(year) == canon["d"]


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "1.1", "b": "7", "c": "9", "d": "99"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=11).to_dict() == make(TOPIC, seed=11).to_dict()
