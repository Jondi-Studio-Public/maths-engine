"""Tests the four-part exam chain on a flat-rate loan whose term is found from the repayment (annual interest, principal repaid per year, years, total interest)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_flat_loan_term_chain"
SKILL_ID = "flat-loan-repayment"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _hard(seed):
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]  # the hardest practice tier
    return make(src.topic, seed=seed, **src.params)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_FLAGS
        assert all(p.distractors and all(d in p.option_notes for d in p.distractors) for p in q.parts)


def test_prompt_shapes_beyond_the_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_oracle_agrees_with_the_rendered_prompt(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == [], bad[:2]


def test_full_marks_for_the_key(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_every_distractor_is_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_all_wrong_earns_nothing(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: f"9999{i}" for i, p in enumerate(q.parts)})
        assert result.marks == 0 and not result.correct


from decimal import ROUND_HALF_UP, Decimal

LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]
FT_FLAGS = [False, True, True, True]


def _q(x, places=2):
    return str(Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _facts(q):
    b = Decimal(re.search(r"\$([\d,]+) (?:to pay|for)", q.stem).group(1).replace(",", ""))
    r = Decimal(re.search(r"(\d+)% per year", q.stem).group(1))
    rep = Decimal(re.search(r"repayments of \$([\d,]+\.\d\d)", q.stem).group(1).replace(",", ""))
    return b, r, rep


def oracle(q):
    b, r, rep = _facts(q)
    interest = b * r / 100
    # solve total repaid = principal + interest for the number of years: 12 R t = B + B r t / 100
    t = b / (12 * rep - interest)
    assert t == int(t)
    return {"a": _q(interest), "b": _q(12 * rep - interest), "c": str(int(t)), "d": _q(interest * t)}


def test_follow_through_from_a_wrong_annual_interest(questions):
    for q in questions[:100]:
        b, r, rep = _facts(q)
        wrong = Decimal(q.parts[0].distractors[0])
        principal = 12 * rep - wrong
        if principal <= 0:
            continue
        years = b / principal
        whole = Decimal(_q(years, 0))
        reply = {"a": str(wrong), "b": _q(principal), "c": str(whole), "d": _q(wrong * whole)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (q.text, reply)
        if reply["c"] != q.parts[2].answer_text and whole != 0:
            assert result.parts["c"].marks == 2, (q.text, reply)


def test_the_term_is_a_whole_number_of_years(questions):
    for q in questions:
        assert 2 <= int(q.parts[2].answer_text) <= 7


def test_every_repayment_is_a_whole_number_of_cents(questions):
    for q in questions:
        b, r, rep = _facts(q)
        assert rep * 100 == int(rep * 100)
