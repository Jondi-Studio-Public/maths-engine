"""Independent oracle for `senior-use-a-recurrence-relation-to-model-a-reducing`.

Reads only `question.text`: it parses the one-line schedule, finds the rate per period and the repayment from
whatever the question gives (the words, the recurrence A(n+1) = m x A(n) - R, an annual rate with its
compounding, or, for the exam tier, the first row of the schedule itself), runs the recurrence from the first
opening balance in whole cents with interest rounded half-up to the cent, and checks every given cell, every
blank, the canonical `answer_text` and the real `table` grader. It does not import the generator module.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table

TOPIC = "senior_reducing_balance_schedule"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("SENIOR_LOAN_ORACLE_SEEDS", "150"))

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")
_HEAD = ["Opening balance ($)", "Interest ($)", "Repayment ($)", "Closing balance ($)"]


def _parse(text):
    m = _TABLE_RE.search(text)
    assert m, text
    head = m.group("head").split(" — ")
    assert head[1:] == _HEAD
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    return head, rows


def _cents(text) -> int:
    return int(Fraction(text.replace(",", "")) * 100)


def _money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def _round_half_up(x: Fraction) -> int:
    return (x + Fraction(1, 2)).__floor__()


def _rate_and_repayment(text, rows):
    """(rate per period as a Fraction, repayment in cents), from the question alone."""
    stem = text.split("Table (")[0]
    m = re.search(r"A\(n\+1\) = (1\.\d+) x A\(n\) - (\d+)[,.] ", stem)
    if m:
        return Fraction(m.group(1)) - 1, int(m.group(2)) * 100
    m = re.search(r"([\d.]+)% per annum, with interest (?:calculated and )?added (monthly|quarterly)", stem)
    if m:
        periods = 12 if m.group(2) == "monthly" else 4
        repay = re.search(r"(?:R = |repaid by )\$([\d,]+)", stem)
        return Fraction(m.group(1)) / 100 / periods, int(repay.group(1).replace(",", "")) * 100
    m = re.search(r"([\d.]+)% (?:of the balance|interest)", stem)
    if m:
        repay = re.search(r"repayment of \$([\d,]+)", stem)
        return Fraction(m.group(1)) / 100, int(repay.group(1).replace(",", "")) * 100
    # exam: neither is stated. The first row shows opening, interest and closing.
    first = rows[0]
    opening, interest, closing = _cents(first[1]), _cents(first[2]), _cents(first[4])
    assert first[3] == "?"
    return Fraction(interest, opening), opening + interest - closing


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_schedule_matches_every_cell_and_the_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        head, rows = _parse(q.text)
        has_total = rows[-1][0] == "Total"
        body = rows[:-1] if has_total else rows
        rate, repayment = _rate_and_repayment(q.text, body)
        balance = _cents(body[0][1])
        parts, interest_total, repaid_total = [], 0, 0
        for cells in body:
            interest = _round_half_up(Fraction(balance) * rate)
            assert Fraction(balance) * rate - (Fraction(balance) * rate).__floor__() != Fraction(1, 2), "tie"
            closing = balance + interest - repayment
            expected = [balance, interest, repayment, closing]
            for col, cell, value in zip(head[1:], cells[1:], expected):
                if cell == "?":
                    parts.append(f"{cells[0]} {col} = {_money(value)}")
                else:
                    assert cell == _money(value), (tier, seed, cells[0], col, cell, _money(value))
            interest_total += interest
            repaid_total += repayment
            balance = closing
        if has_total:
            total = rows[-1]
            for col, cell, value in zip(head[2:4], total[2:4], (interest_total, repaid_total)):
                if cell == "?":
                    parts.append(f"Total {col} = {_money(value)}")
                else:
                    assert cell == _money(value)
        assert parts and len(parts) == len(q.answer["blanks"])
        assert set(parts) == set(q.answer_text.split(", "))
        assert q.answer_text == ", ".join(parts)
        assert grade_table(q, ", ".join(parts)).correct, (tier, seed)


def test_exam_rate_and_repayment_are_recoverable_exactly_from_the_first_row():
    """The recovered rate is a whole number of tenths of a percent, so the learner cannot be left guessing."""
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="exam")
        _head, rows = _parse(q.text)
        rate, repayment = _rate_and_repayment(q.text, rows[:-1])
        assert (rate * 1000).denominator == 1
        assert repayment % 1000 == 0 and repayment > 0


def test_hard_rate_per_period_is_the_annual_rate_divided_by_the_periods():
    seen = set()
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="hard")
        stem = q.text.split("Table (")[0]
        annual = Fraction(re.search(r"([\d.]+)% per annum", stem).group(1)) / 100
        rate, _repayment = _rate_and_repayment(q.text, _parse(q.text)[1][:-1])
        seen.add(annual / rate)
    assert seen == {12, 4}
