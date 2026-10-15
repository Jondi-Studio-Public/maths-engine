"""Tests the exam source of `senior-prepare-a-personal-budget-for-a-given-income`: weekly fixed costs, weekly savings, weeks to a goal."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n1_budget_to_goal"
SKILL_ID = "senior-prepare-a-personal-budget-for-a-given-income"
HARD_TOPIC = "p10c_personal_budget"
LABELS = ("a", "b", "c")
SEEDS = range(120)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _fmt(value, places):
    q = Decimal(1).scaleb(-places)
    return f"{Decimal(value).quantize(q, rounding=ROUND_HALF_UP):.{places}f}"


def _fr(text):
    return Fraction(text.replace(",", ""))


def _round(value, places):
    """Fraction `value` rounded half up to `places` decimal places, as a Fraction."""
    scale = 10**places
    scaled = value * scale
    return Fraction(math.floor(scaled + Fraction(1, 2)), scale)


def _text(value, places):
    return _fmt(Decimal(value.numerator) / Decimal(value.denominator), places) if places else str(round(value))


def _budget(q):
    income, rent, phone, ins, reg, goal = (_fr(x) for x in re.findall(r"\$([\d,]+)", q.stem))
    pct = int(re.search(r"(\d+)% of", q.stem).group(1))
    return income, rent, phone, ins, reg, goal, pct


def oracle(q):
    income, rent, phone, ins, reg, goal, pct = _budget(q)
    fixed = _round(rent + phone * 12 / 52 + ins / 13 + reg / 52, 2)
    savings = _round(income - fixed - income * pct / 100, 2)
    return {"a": _text(fixed, 2), "b": _text(savings, 2), "c": str(math.ceil(goal / savings))}


def ft_cases(q):
    income, rent, phone, ins, reg, goal, pct = _budget(q)
    for d in q.parts[0].distractors:
        wa = _fr(d)
        wb = _round(income - wa - income * pct / 100, 2)
        if wb > 0:
            yield "a", d, {"b": _text(wb, 2), "c": str(math.ceil(goal / wb))}
    for d in q.parts[1].distractors:
        wb = _fr(d)
        if wb > 0:
            yield "b", d, {"c": str(math.ceil(goal / wb))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == list(LABELS)
        assert all(p.answer_kind in ("number", "fraction") for p in q.parts)
        assert all(p.distractors for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts)


def test_distractors_rejected(questions):
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through(questions):
    cases = 0
    for q in questions:
        canon = _canonical(q)
        for wrong_label, wrong_text, later in ft_cases(q):
            if wrong_text in later.values() or any(_fr(v) == 0 for v in later.values() if "/" not in v):
                continue  # the engine withholds follow-through from a repeated or zero value
            reply = {**canon, wrong_label: wrong_text, **later}
            result = grade(_SKILL, q, reply)
            assert result.parts[wrong_label].marks == 0, (q.text, wrong_label, wrong_text)
            for label in later:
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, label, wrong_text, later)
                if later[label] != canon[label]:
                    assert part.follow_through is True
            assert not result.correct
            cases += 1
    assert cases >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:30]:
        result = grade(_SKILL, q, {label: "999999.9" for label in LABELS})
        assert result.marks == 0 and not result.correct
