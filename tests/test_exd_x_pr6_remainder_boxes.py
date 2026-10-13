"""Tests the exam tier of `f10-division-remainder-context`: boxes needed, empty places, cost."""

from __future__ import annotations

from fractions import Fraction

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_remainder_boxes"
SKILL_ID = "f10-division-remainder-context"


def oracle(nums):
    n, d, p = nums
    needed = -((-Fraction(n)) // d)  # ceiling, exact
    return {"a": str(needed), "b": str(needed * d - n), "c": str(needed * p)}


def follow(nums, a):
    n, d, p = nums
    wrong = a + 1
    return wrong, {"b": wrong * d - n, "c": wrong * p}


def test_wired():
    S.check_wired(TOPIC, SKILL_ID)


def test_variety():
    S.check_variety(TOPIC, SKILL_ID)


def test_key_from_prompt():
    S.check_oracle(TOPIC, oracle)


def test_answer_space():
    S.check_answer_space(TOPIC)


def test_grading():
    S.check_grading(TOPIC, SKILL_ID)


def test_tagged():
    S.check_tagged(TOPIC)


def test_follow_through():
    S.check_follow_through(TOPIC, SKILL_ID, follow)


def test_clean():
    S.check_deterministic_and_clean(TOPIC)
