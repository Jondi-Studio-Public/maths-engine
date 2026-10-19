"""Tests `f10-financial-multistep-percent` (topic `fr2e_percent_multistep`, exam chain `exd_x_fr2e_percent_multistep`).

The oracle reads the marked prices, percentages and pack sizes back out of the question TEXT and recomputes each answer
with exact `Fraction` arithmetic (no floating point), including which shop or pack is cheaper.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import mathgen.generators.fr2e_percent_multistep_build  # noqa: F401  (registers the topics)
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

SKILL = "f10-financial-multistep-percent"
TOPIC = "fr2e_percent_multistep"
EXAM = "exd_x_fr2e_percent_multistep"


def _money(value: Fraction) -> str:
    cents = value * 100
    assert cents.denominator == 1, value
    return f"{cents.numerator // 100}.{cents.numerator % 100:02d}"


def _oracle_single(tier: str, text: str) -> str:
    if tier == "easy":
        marked = int(re.search(r"marked \$(\d+)", text).group(1))
        off = int(re.search(r"(\d+)%", text).group(1))
        return _money(marked * Fraction(100 - off, 100))
    if tier == "medium":
        marked = int(re.search(r"marked \$(\d+)", text).group(1))
        off = int(re.search(r"for (\d+)% off", text).group(1))
        return _money(marked * Fraction(100 - off, 100) * Fraction(110, 100))
    size_a, cost_a = re.search(r"Pack A is (\d+) g and costs \$(\d+\.\d\d)", text).groups()
    size_b, marked_b, off = re.search(r"Pack B is (\d+) g and is marked \$(\d+\.\d\d), with (\d+)% off", text).groups()
    per_a = Fraction(cost_a) * 100 / int(size_a)
    per_b = Fraction(marked_b) * Fraction(100 - int(off), 100) * 100 / int(size_b)
    return _money(abs(per_a - per_b))


def _shops(text: str):
    a_marked, a_off = (
        int(v) for v in re.search(r"Shop A marks each [a-z ]+ at \$(\d+) and takes (\d+)% off", text).groups()
    )
    b_marked = int(re.search(r"Shop B marks each at \$(\d+) before GST", text).group(1))
    count = int(re.search(r"buying (\d+) identical", text).group(1))
    price_a = a_marked * Fraction(100 - a_off, 100)
    price_b = b_marked * Fraction(110, 100)
    return price_a, price_b, count


def test_registered_sources():
    assert_registered(SKILL, TOPIC, EXAM)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_matches_oracle(tier):
    for seed in SEEDS:
        q = single(TOPIC, tier, seed)
        assert q.answer_text == _oracle_single(tier, q.text), (tier, seed, q.text, q.answer_text)


def test_clean_and_deterministic():
    assert_clean_and_deterministic(TOPIC, TIERS)
    assert_clean_and_deterministic(EXAM, ("exam",))


def test_single_grading():
    assert_single_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    assert_tiers_differ(TOPIC)
    assert all("GST" not in single(TOPIC, "easy", s).text for s in SEEDS)
    assert all("GST" in single(TOPIC, "medium", s).text for s in SEEDS)
    assert all("per 100 g" in single(TOPIC, "hard", s).text for s in SEEDS)


def test_exam_matches_oracle():
    for seed in range(80):
        q = exam(EXAM, seed)
        price_a, price_b, count = _shops(q.text)
        cheaper = min(price_a, price_b)
        answers = [p.answer_text for p in q.parts]
        assert answers[0] == _money(price_a), (seed, q.text)
        assert answers[1] == _money(price_b), (seed, q.text)
        shop = "Shop A" if price_a < price_b else "Shop B"
        assert (
            answers[2]
            == f"{shop} is cheaper, by ${_money(abs(price_a - price_b))} for each {re.search(r'for one (.+?),', q.parts[2].prompt).group(1)}"
        )
        assert answers[3] == _money(cheaper * count), (seed, q.text)


def test_exam_chain_shape():
    assert_chain_shape(EXAM, ["number", "number", "mcq", "number"], [2, 2, 2, 2], [False, False, False, True])


def test_exam_grading():
    assert_exam_grading(SKILL, EXAM)


def test_follow_through_from_a_wrong_shop_price():
    """The total in part (d) follows the cheaper of the student's own two shop prices."""

    def total(q, canon, wrong):
        _, price_b, count = _shops(q.text)
        return _money(min(Fraction(wrong), Fraction(canon["b"])) * count)

    assert_follow_through(SKILL, EXAM, "d", "a", total)
