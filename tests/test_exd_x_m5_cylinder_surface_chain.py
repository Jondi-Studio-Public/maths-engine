"""Exam source of `senior-find-the-surface-area-of-familiar-solids-including`: a closed cylinder worked backwards."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_cylinder_surface_chain"
SKILL_ID = "senior-find-the-surface-area-of-familiar-solids-including"
PI = Fraction(math.pi)


def oracle(q):
    r = int(re.search(r"(?:radius of|Its radius is) (\d+) cm", q.stem).group(1))
    s = Fraction(re.search(r"(\d+\.\d) cm²", q.stem).group(1))
    h = c.round_half_up(s / (2 * PI * r) - r, 1)
    return {
        "a": c.show(h, 1),
        "b": c.show(PI * r * r * h, 0),
        "c": c.show(2 * PI * r * h, 0),
    }


def test_shapes():
    c.check_shapes(TOPIC, SKILL_ID)


def test_source_wired():
    c.check_source_wired(TOPIC, SKILL_ID)


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle)


def test_answer_space():
    for label in "abc":
        c.check_answer_space(TOPIC, label)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_height():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
