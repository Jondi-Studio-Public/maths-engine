"""Tests the multi-part exam source of `senior-use-a-recurrence-relation-to-model-an-annuity`.

The oracle reads the amount, rate, compounding and withdrawal from the rendered stem and runs the recurrence in
Decimal arithmetic (the generator uses exact fractions).
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, getcontext

from mathgen.engine import grade
from tests.exd_x_r3c_support import (
    check_all_wrong,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_full_marks_and_distractors,
    check_oracle,
    check_shapes,
    check_text,
    check_wired,
    exam_questions,
    skill_for,
)

TOPIC = "exd_x_r3c_annuity_payout"
SKILL_ID = "senior-use-a-recurrence-relation-to-model-an-annuity"
SKILL = skill_for(SKILL_ID)
_STEM = re.compile(r"\$([\d,]+)[^%]*?([\d.]+)% per annum, compounded (monthly|quarterly)\..*?\$([\d,]+)")


def parse(q):
    p, rate, how, d = _STEM.search(q.stem).groups()
    per_year = 12 if how == "monthly" else 4
    return Decimal(p.replace(",", "")), Decimal(rate) / 100 / per_year, Decimal(d.replace(",", ""))


def cents(x):
    return f"{Decimal(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def oracle(q):
    getcontext().prec = 60
    p, i, d = parse(q)
    balance, history = p, [p]
    while balance * (1 + i) >= d:
        balance = balance * (1 + i) - d
        history.append(balance)
    n = len(history) - 1
    return {"a": cents(history[1]), "b": str(n), "c": cents(balance), "d": cents(balance * (1 + i))}


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=30)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_count():
    """A count one too high, carried into the balance and the final payment."""
    getcontext().prec = 60
    for q in exam_questions(TOPIC, 30):
        p, i, d = parse(q)
        wrong = int(q.parts[1].answer_text) + 1
        growth = (1 + i) ** wrong
        balance = p * growth - d * (growth - 1) / i
        final = balance * (1 + i)
        got = grade(SKILL, q, {"a": q.parts[0].answer_text, "b": str(wrong), "c": cents(balance), "d": cents(final)})
        assert got.parts["b"].marks == 0
        if balance > 0:
            for label in "cd":
                assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                    q.text,
                    label,
                )


def test_follow_through_from_a_wrong_balance():
    getcontext().prec = 60
    for q in exam_questions(TOPIC, 30):
        p, i, d = parse(q)
        wrong = Decimal(q.parts[2].distractors[0])
        got = grade(
            SKILL,
            q,
            {p_.label: p_.answer_text for p_ in q.parts} | {"c": str(wrong), "d": cents(wrong * (1 + i))},
        )
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [1, 2, 2, 1]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
