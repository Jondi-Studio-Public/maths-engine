"""Exam-tier checks for `calculate-wages`: gross pay with a loaded rate, tax withheld, average pay per hour."""

import re
from fractions import Fraction

from mathgen import make
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_variety,
    check_wired,
    decimal_text,
    text_of,
)

TOPIC = "exd_x_sc2_wages_penalty"
SKILL = "calculate-wages"


def _facts(q):
    text = text_of(q)
    rate = Fraction(re.search(r"\$(\d+\.\d\d) per hour", text).group(1))
    ordinary = int(re.search(r"first (\d+) hours", text).group(1))
    total = int(re.search(r"worked (\d+) hours", text).group(1))
    loading = Fraction(2) if "double time" in text else Fraction(3, 2)
    percent = int(re.search(r"Tax of (\d+)%", text).group(1))
    return rate, ordinary, total, loading, percent


def oracle(q):
    rate, ordinary, total, loading, percent = _facts(q)
    gross = rate * ordinary + rate * loading * (total - ordinary)
    return {
        "a": decimal_text(gross, 2),
        "b": decimal_text(gross * percent / 100, 2),
        "c": decimal_text(gross / total, 2),
    }


def wrong_chain(q):
    rate, ordinary, total, loading, percent = _facts(q)
    gross = Fraction(decimal_text(rate * ordinary + rate * loading * (total - ordinary) + Fraction(37), 2))
    return {
        "a": decimal_text(gross, 2),
        "b": decimal_text(gross * percent / 100, 2),
        "c": decimal_text(gross / total, 2),
    }


def test_no_text_defects():
    check_no_text_defects(TOPIC)


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("earnings", seed=s, kind="wages"))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_deterministic():
    check_deterministic(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_gross_pay():
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "1.11", "b": "2.22", "c": "3.33"})
