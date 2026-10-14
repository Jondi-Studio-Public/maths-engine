"""Exam source of `senior-use-formulas-to-calculate-areas-of-regular-shapes`: rectangle plus semicircle."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_rectangle_semicircle_chain"
SKILL_ID = "senior-use-formulas-to-calculate-areas-of-regular-shapes"


def oracle(q):
    t = q.stem
    w = Fraction(re.search(r"(\d+) m", t).group(1))
    total = Fraction(re.search(r"(\d+\.\d) m²", t).group(1))
    semi = c.round_half_up(Fraction(math.pi * float(w) ** 2 / 8), 1)
    rect = total - semi
    side = c.round_half_up(rect / w, 1)
    return {"a": c.show(semi, 1), "b": c.show(rect, 1), "c": c.show(side, 1)}


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle)


def test_answer_space():
    c.check_answer_space(TOPIC, "b")
    c.check_answer_space(TOPIC, "c", minimum=70)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_semicircle():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
