"""Exam source of `senior-obtain-a-scale-factor-and-use-it-to`: map scale, then areas both ways."""

from __future__ import annotations

import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_map_area_scale_chain"
SKILL_ID = "senior-obtain-a-scale-factor-and-use-it-to"


def oracle(q):
    t = q.stem
    m = Fraction(re.search(r"(\d+) cm apart", t).group(1))
    dist = Fraction(re.search(r"(\d+(?:\.\d+)?) km", t).group(1))
    area = Fraction(re.search(r"(\d+\.\d) cm²", t).group(1))
    ha = Fraction(re.search(r"(\d+\.\d) ha", t).group(1))
    n = dist * 100_000 / m
    assert n.denominator == 1
    return {
        "a": str(n),
        "b": c.show(area * n * n / 10**8, 1),
        "c": c.show(ha * 10**8 / (n * n), 1),
    }


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle)


def test_answer_space():
    for label in "bc":
        c.check_answer_space(TOPIC, label)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_scale():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
