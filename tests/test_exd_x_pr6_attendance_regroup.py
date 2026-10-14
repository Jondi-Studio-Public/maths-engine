"""Tests the exam tier of `f10-multidigit-add-subtract-regroup`: weekend total, a comparison, and the shortfall."""

from __future__ import annotations

import exd_x_pr6_support as S

TOPIC = "exd_x_pr6_attendance_regroup"
SKILL_ID = "f10-multidigit-add-subtract-regroup"


def oracle(nums):
    x, y, z, target = nums
    return {"a": str(x + y + z), "b": str(y - x), "c": str(target - (x + y + z))}


def follow(nums, a):
    wrong = a + 10
    return wrong, {"c": nums[3] - wrong}


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
