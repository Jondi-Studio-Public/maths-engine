"""Exam source of `senior-the-sum-of-the-squares-of-the-lengths`: other diagonal, angle DAB, area."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_parallelogram_diagonal_chain"
SKILL_ID = "senior-the-sum-of-the-squares-of-the-lengths"


def oracle(q):
    t = q.stem
    a = int(re.search(r"AB (?:=|is) (\d+)", t).group(1))
    b = int(re.search(r"AD (?:=|is) (\d+)", t).group(1))
    given = int(re.search(r"diagonal AC (?:measures |is )?(\d+)", t).group(1))
    other_sq = 2 * (a * a + b * b) - given**2
    other = math.isqrt(other_sq)
    assert other * other == other_sq
    bd = other
    angle = math.degrees(math.acos((a * a + b * b - bd * bd) / (2 * a * b)))
    shown = c.round_half_up(Fraction(angle), 1)
    area = a * b * math.sin(math.radians(angle))
    return {"a": str(other), "b": c.show(shown, 1), "c": c.show(Fraction(area), 0)}


def test_shapes():
    c.check_shapes(TOPIC, SKILL_ID)


def test_source_wired():
    c.check_source_wired(TOPIC, SKILL_ID)


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle)


def test_answer_space():
    for label in "bc":
        c.check_answer_space(TOPIC, label, minimum=100 if label == "b" else 60)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_diagonal():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
