"""Tests the exam tier of `f10-multiply-divide-one-digit-arrays`: an array, some taken away, the rest in equal piles."""

from __future__ import annotations

from fractions import Fraction

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_array_stacks"
SKILL_ID = "f10-multiply-divide-one-digit-arrays"


def oracle(nums):
    r, c, x, s = nums
    total = r * c
    piles = Fraction(total - x, s)
    assert piles.denominator == 1
    return {"a": str(total), "b": str(total - x), "c": str(piles)}


def follow(nums, a):
    r, c, x, s = nums
    wrong = a + s
    return wrong, {"b": wrong - x, "c": (wrong - x) // s}


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
