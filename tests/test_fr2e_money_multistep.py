"""Tests `f10-financial-multistep-whole-numbers` (topic `fr2e_money_multistep`, exam chain `exd_x_fr2e_money_multistep`).

The oracle reads the prices, counts, notes and weekly savings back out of the question TEXT and recomputes every answer
with plain integer arithmetic, so a generator that changes its wording without changing its sums is caught.
"""

from __future__ import annotations

import math
import re

import pytest

import mathgen.generators.fr2e_money_multistep_build  # noqa: F401  (registers the topics)
from tests.fr2e_support import (
    SEEDS,
    TIERS,
    assert_chain_shape,
    assert_clean_and_deterministic,
    assert_exam_grading,
    assert_follow_through,
    assert_registered,
    assert_single_grading,
    assert_tiers_differ,
    exam,
    single,
)

SKILL = "f10-financial-multistep-whole-numbers"
TOPIC = "fr2e_money_multistep"
EXAM = "exd_x_fr2e_money_multistep"

_LINE = re.compile(r"(\d+) [a-z ]+? at \$(\d+) each")


def _bill(text: str) -> int:
    lines = _LINE.findall(text)
    assert lines, text
    return sum(int(n) * int(p) for n, p in lines)


def _first_amount(pattern: str, text: str) -> int:
    return int(re.search(pattern, text).group(1))


def _oracle_single(tier: str, text: str) -> int:
    if tier == "easy":
        bill = (
            _bill(text)
            if "at $" in text
            else _first_amount(r"(\d+) [a-z ]+?\. Each costs", text) * _first_amount(r"Each costs \$(\d+)", text)
        )
        if "note" in text:
            return _first_amount(r"\$(\d+) note", text) - bill
        return bill
    if tier == "medium":
        bill = _bill(text)
        if "voucher" in text:
            return bill - _first_amount(r"\$(\d+) voucher", text)
        if "share the bill" in text:
            assert bill % 2 == 0
            return bill // 2
        note = _first_amount(r"only \$(\d+) notes", text)
        return math.ceil(bill / note) * note - bill
    if "lay-by" in text:
        cost = _first_amount(r"for \$(\d+) on lay-by", text)
        return (cost - _first_amount(r"\$(\d+) deposit", text)) // _first_amount(r"then (\d+) equal", text)
    cost = _first_amount(r"costs \$(\d+)", text)
    if "birthday present" in text:
        have = _first_amount(r"has \$(\d+)", text)
        rate, weeks = (int(v) for v in re.search(r"saves \$(\d+) each week for (\d+) weeks", text).groups())
        return cost - (have + rate * weeks - _first_amount(r"spends \$(\d+)", text))
    return math.ceil((cost - _first_amount(r"saved \$(\d+)", text)) / _first_amount(r"saves \$(\d+) each week", text))


def test_registered_sources():
    assert_registered(SKILL, TOPIC, EXAM)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_matches_oracle(tier):
    for seed in SEEDS:
        q = single(TOPIC, tier, seed)
        assert int(q.answer_text) == _oracle_single(tier, q.text), (tier, seed, q.text, q.answer_text)


def test_clean_and_deterministic():
    assert_clean_and_deterministic(TOPIC, TIERS)
    assert_clean_and_deterministic(EXAM, ("exam",))


def test_single_grading():
    assert_single_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    assert_tiers_differ(TOPIC)
    # Easy is one operation; medium combines two kinds of item or a note rule; hard is a multi-week plan or a lay-by.
    assert all(len(_LINE.findall(single(TOPIC, "easy", s).text)) <= 1 for s in SEEDS)
    assert all(len(_LINE.findall(single(TOPIC, "medium", s).text)) == 2 for s in SEEDS)
    assert all("week" in single(TOPIC, "hard", s).text or "lay-by" in single(TOPIC, "hard", s).text for s in SEEDS)


def test_exam_matches_oracle():
    for seed in range(80):
        q = exam(EXAM, seed)
        total = _bill(q.text)
        note = _first_amount(r"\$(\d+) note", q.text)
        goal = int(re.findall(r"costs \$(\d+)", q.text)[-1])
        week = _first_amount(r"saves \$(\d+) each week", q.text)
        change = note - total
        weeks = math.ceil((goal - change) / week)
        left = change + week * weeks - goal
        assert [int(p.answer_text) for p in q.parts] == [total, change, weeks, left], (seed, q.text)


def test_exam_chain_shape():
    assert_chain_shape(EXAM, ["number"] * 4, [2, 1, 2, 1], [False, True, True, True])


def test_exam_grading():
    assert_exam_grading(SKILL, EXAM)


def test_follow_through_from_a_wrong_total():
    """A wrong total in part (a) still earns part (b) when the change is worked from that wrong total."""

    def change(q, canon, wrong_total):
        note = _first_amount(r"\$(\d+) note", q.text)
        return str(note - int(wrong_total)) if note - int(wrong_total) >= 0 else None

    assert_follow_through(SKILL, EXAM, "b", "a", change)


def test_follow_through_from_a_wrong_change():
    """A wrong change in part (b) still earns the weeks in part (c) when they are worked from that change."""

    def weeks(q, canon, wrong_change):
        goal = int(re.findall(r"costs \$(\d+)", q.text)[-1])
        week = _first_amount(r"saves \$(\d+) each week", q.text)
        return str(max(0, math.ceil((goal - int(wrong_change)) / week)))

    assert_follow_through(SKILL, EXAM, "c", "b", weeks)
