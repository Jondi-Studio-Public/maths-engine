"""Tests the exam tier of `f10-times-two-facts-inverse`: halve twice, then double and add a fee."""

from __future__ import annotations

from fractions import Fraction

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_pairs_bundles"
SKILL_ID = "f10-times-two-facts-inverse"


def oracle(nums):
    n, fee = nums[0], nums[-1]
    pairs = Fraction(n, 2)
    bundles = pairs / 2
    assert bundles.denominator == 1
    return {"a": str(pairs), "b": str(bundles), "c": str(2 * bundles + fee)}


def follow(nums, a):
    wrong = a + 2
    return wrong, {"b": wrong // 2, "c": 2 * (wrong // 2) + nums[-1]}


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
