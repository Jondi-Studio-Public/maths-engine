"""Tests the four-part exam chain on a loan recurrence with a change of rate (balance, balance after the change, repayment to clear, interest)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_loan_recurrence_chain"
SKILL_ID = "senior-use-technology-and-a-recurrence-relation-to-model"
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
MARKS = [2, 3, 3, 2]
FT_FLAGS = [False, True, True, True]


def _money(x: Fraction) -> str:
    d = Decimal(x.numerator) / Decimal(x.denominator)
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _facts(q):
    p = int(re.search(r"\$([\d,]+) home loan", q.stem).group(1).replace(",", ""))
    r1 = Fraction(re.search(r"(?:interest of|rate is) ([\d.]+)%", q.stem).group(1)) / 100
    r2 = Fraction(re.search(r"(?:rises|falls) to ([\d.]+)%", q.stem).group(1)) / 100
    rep = int(re.search(r"repayment (?:of|is) \$([\d,]+)", q.stem).group(1).replace(",", ""))
    n1 = int(re.search(r"After (\d+) months", q.stem).group(1))
    m = int(re.search(r"owing (\d+) months", q.parts[1].prompt).group(1))
    k = int(re.search(r"exactly (\d+) further", q.parts[2].prompt).group(1))
    return p, r1, r2, rep, n1, m, k


def _run(b, r, rep, n):
    for _ in range(n):
        b = b * (1 + r) - rep
    return b


def _annuity(b, r, k):
    return b * r / (1 - (1 + r) ** (-k))


def _chain(q, first=None):
    p, r1, r2, rep, n1, m, k = _facts(q)
    b1 = Fraction(_money(_run(Fraction(p), r1, rep, n1))) if first is None else Fraction(first)
    b2 = Fraction(_money(_run(b1, r2, rep, m)))
    r_new = Fraction(_money(_annuity(b2, r2, k)))
    return b1, b2, r_new, k * r_new - b2


def oracle(q):
    b1, b2, r_new, interest = _chain(q)
    return {"a": _money(b1), "b": _money(b2), "c": _money(r_new), "d": _money(interest)}


def test_follow_through_from_a_wrong_first_balance(questions):
    for q in questions[:80]:
        p, r1, r2, rep, n1, m, k = _facts(q)
        wrong = q.parts[0].distractors[0]
        b1, b2, r_new, interest = _chain(q, first=wrong)
        if b2 <= 0:
            continue
        reply = {"a": wrong, "b": _money(b2), "c": _money(r_new), "d": _money(interest)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == MARKS["abcd".index(label)], (q.text, label, reply)
            assert result.parts[label].follow_through is True


def test_the_new_repayment_clears_the_balance(questions):
    for q in questions[:60]:
        p, r1, r2, rep, n1, m, k = _facts(q)
        b1, b2, r_new, interest = _chain(q)
        left = _run(b2, r2, r_new, k)
        assert abs(left) < Fraction(1, 200) * ((1 + r2) ** k - 1) / r2 + Fraction(1, 100)


def test_both_rate_directions_occur(questions):
    assert {"rises" in q.stem for q in questions} == {True, False}
