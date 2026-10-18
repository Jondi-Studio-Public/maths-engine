"""Tests the exam tier of `f10-mental-mult-div-extension`: tens times tens, divide by tens, then price."""

from __future__ import annotations

from fractions import Fraction

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_scale_crates"
SKILL_ID = "f10-mental-mult-div-extension"


def oracle(nums):
    a0, b0, s0, q = nums
    total = Fraction(a0 * b0)
    crates = total / s0
    assert crates.denominator == 1
    return {"a": str(total), "b": str(crates), "c": str(crates * q)}


def follow(nums, a):
    a0, b0, s0, q = nums
    wrong = a + s0
    return wrong, {"b": wrong // s0, "c": wrong // s0 * q}


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
