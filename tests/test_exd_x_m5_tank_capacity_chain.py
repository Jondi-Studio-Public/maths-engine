"""Exam source of `senior-use-formulas-to-find-the-volume-and-capacity`: cylindrical tank, pump, square tank."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_tank_capacity_chain"
SKILL_ID = "senior-use-formulas-to-find-the-volume-and-capacity"


def oracle_full(q):
    t = q.stem
    nums = re.findall(r"(\d+(?:\.\d+)?) (?:m\b|L|litres)", t)
    d, h = Fraction(nums[0]), Fraction(nums[1])
    rate = int(re.search(r"(\d+) (?:litres per minute|L every minute)", t).group(1))
    s = Fraction(re.search(r"(?:side|sides of) (\d+\.\d) m", t).group(1))
    p = int(re.search(r"to (\d+)% of", q.parts[1].prompt).group(1))
    litres = Fraction(math.pi) * (d / 2) ** 2 * h * 1000
    cap = c.round_half_up(litres, 0)
    return {
        "a": c.show(cap, 0),
        "b": c.show(Fraction(p, 100) * cap / rate, 0),
        "c": c.show(cap / 1000 / (s * s), 1),
    }


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle_full)


def test_answer_space():
    for label in "abc":
        c.check_answer_space(TOPIC, label, minimum=70 if label == "c" else 100)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_capacity():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
