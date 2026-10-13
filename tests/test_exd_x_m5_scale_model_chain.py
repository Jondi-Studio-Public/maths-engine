"""Exam source of `senior-obtain-a-scale-factor-and-use-it-to-acmgm025`: a scale model and the real solid."""

from __future__ import annotations

import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_scale_model_chain"
SKILL_ID = "senior-obtain-a-scale-factor-and-use-it-to-acmgm025"


def oracle(q):
    t = q.stem
    h1 = Fraction(re.search(r"(\d+) cm tall", t).group(1))
    h2 = Fraction(re.search(r"(\d+(?:\.\d+)?) m tall", t).group(1))
    area = Fraction(re.search(r"(\d+) cm²", t).group(1))
    vol = Fraction(re.search(r"(\d+) mL", t).group(1))
    k = h2 * 100 / h1
    assert k.denominator == 1
    return {
        "a": str(k),
        "b": c.show(area * k * k / 10_000, 1),
        "c": c.show(vol * k**3 / 1000, 0),
    }


def test_shapes():
    c.check_shapes(TOPIC, SKILL_ID)


def test_source_wired():
    c.check_source_wired(TOPIC, SKILL_ID)


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle)


def test_answer_space():
    for label in "bc":
        c.check_answer_space(TOPIC, label)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_factor():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
