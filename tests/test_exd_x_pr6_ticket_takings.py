"""Tests the exam tier of `f10-multidigit-multiplication`: takings from two ticket kinds, then what is left."""

from __future__ import annotations

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_ticket_takings"
SKILL_ID = "f10-multidigit-multiplication"


def oracle(nums):
    na, a_price, nc, c_price, hire = nums
    a, b = na * a_price, nc * c_price
    return {"a": str(a), "b": str(b), "c": str(a + b - hire)}


def follow(nums, a):
    na, a_price, nc, c_price, hire = nums
    wrong = a + 100
    return wrong, {"c": wrong + nc * c_price - hire}


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
