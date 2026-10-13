"""Tests the multi-part exam source of `senior-with-the-aid-of-a-financial-calculator-or`.

The oracle reads the amount, rate and term from the rendered stem and works the repayment with mpmath's annuity
formula; a second test finds the repayment independently by bisection on the month-by-month schedule.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, getcontext

import mpmath

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

TOPIC = "exd_x_r3c_loan_first_period"
SKILL_ID = "senior-with-the-aid-of-a-financial-calculator-or"
SKILL = skill_for(SKILL_ID)
_STEM = re.compile(r"\$([\d,]+) has an interest rate of ([\d.]+)% .* over (\d+) years")


def parse(q):
    p, rate, years = _STEM.search(q.stem).groups()
    return Decimal(p.replace(",", "")), Decimal(rate), int(years)


def cents(x):
    return Decimal(str(mpmath.nstr(x, 30))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt(d):
    return f"{d:.2f}"


def oracle(q):
    p, rate, years = parse(q)
    # 40 digits only inside the oracle: setting mpmath.mp.dps globally leaks into every later test in the worker.
    with mpmath.workdps(40):
        i = mpmath.mpf(str(rate)) / 1200
        n = 12 * years
        pay = cents(mpmath.mpf(str(p)) * i / (1 - (1 + i) ** (-n)))
        interest1 = cents(mpmath.mpf(str(p)) * i)
        balance1 = p + interest1 - pay
        interest2 = cents(mpmath.mpf(str(balance1)) * i)
    return {"a": fmt(pay), "b": fmt(interest1), "c": fmt(balance1), "d": fmt(interest2)}


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_repayment_clears_the_loan_by_bisection():
    """Find the repayment by bisection on the month-by-month schedule; it must round to the stated (a)."""
    getcontext().prec = 40
    for q in exam_questions(TOPIC, 12):
        p, rate, years = parse(q)
        i = rate / 1200
        lo, hi = Decimal(0), p
        for _ in range(70):
            mid = (lo + hi) / 2
            balance = p
            for _ in range(12 * years):
                balance = balance * (1 + i) - mid
            lo, hi = (mid, hi) if balance > 0 else (lo, mid)
        assert fmt(lo.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) == q.parts[0].answer_text, q.stem


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=30)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_repayment():
    """A wrong repayment carried correctly into the balance and the second month's interest."""
    for q in exam_questions(TOPIC, 30):
        p, rate, years = parse(q)
        i = rate / 1200
        wrong = Decimal(q.parts[0].distractors[0])
        interest1 = Decimal(q.parts[1].answer_text)
        balance1 = p + interest1 - wrong
        interest2 = (balance1 * i).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        got = grade(SKILL, q, {"a": str(wrong), "b": q.parts[1].answer_text, "c": fmt(balance1), "d": fmt(interest2)})
        assert got.parts["a"].marks == 0
        for label in "cd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                q.text,
                label,
            )


def test_follow_through_from_a_wrong_first_interest():
    for q in exam_questions(TOPIC, 30):
        p, rate, years = parse(q)
        i = rate / 1200
        wrong = Decimal(q.parts[1].distractors[0])
        repayment = Decimal(q.parts[0].answer_text)
        balance1 = p + wrong - repayment
        interest2 = (balance1 * i).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        got = grade(SKILL, q, {"a": q.parts[0].answer_text, "b": str(wrong), "c": fmt(balance1), "d": fmt(interest2)})
        assert got.parts["b"].marks == 0
        assert got.parts["c"].marks == got.parts["c"].max_marks and got.parts["c"].follow_through, q.text


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [2, 1, 1, 1]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
