"""Tests the exam tier of `f10-multiplication-facts-3-4-5-10`: items in bags, the bag cost, then two equal shares."""

from __future__ import annotations

from fractions import Fraction

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_market_fact_family"
SKILL_ID = "f10-multiplication-facts-3-4-5-10"


def oracle(nums):
    t, k, c, m, g = nums
    total = Fraction(t * k)
    share = total / m
    each = share / g
    assert share.denominator == each.denominator == 1
    return {"a": str(total), "b": str(k * c), "c": str(share), "d": str(each)}


def follow(nums, a):
    t, k, c, m, g = nums
    wrong = a + m * g
    return wrong, {"c": wrong // m, "d": wrong // m // g}


def test_wired():
    S.check_wired(TOPIC, SKILL_ID)


def test_variety():
    S.check_variety(TOPIC, SKILL_ID)


def test_key_from_prompt():
    S.check_oracle(TOPIC, oracle)


def test_grading():
    S.check_grading(TOPIC, SKILL_ID)


def test_tagged():
    S.check_tagged(TOPIC)


def test_follow_through():
    S.check_follow_through(TOPIC, SKILL_ID, follow)


def test_clean():
    S.check_deterministic_and_clean(TOPIC)
