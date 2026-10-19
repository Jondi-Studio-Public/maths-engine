"""Tests for the multi-part exam source `exd_x_r2h_poisson_chain` (skill `ucv1-poisson-count-probability`).

The oracle re-derives every part's answer from the rendered stem and prompts alone, with exact fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks

TOPIC = "exd_x_r2h_poisson_chain"
SKILL_ID = "ucv1-poisson-count-probability"
HARD_TOPIC = "p12e_poisson_count"
LABELS = list("abcd")
MARKS = [2, 1, 2, 2]


def _frac(f: Fraction) -> str:
    return str(f)


def oracle(q):
    text = q.text
    lam = int(re.search(r"average of (\d+)", text).group(1))
    mins = int(re.search(r"(\d+)-minute", text).group(1))
    big_e = int(re.search(r"exactly 1/(\d+)\.", text).group(1))
    k = int(re.search(r"(?:exactly |N = )(\d+)", text.split("(a)")[1]).group(1))
    nxt = int(re.search(r"(\d+) minutes", text.split("(d)")[1]).group(1))
    mu = Fraction(lam * mins, 60)
    e_inv = Fraction(1, big_e)
    pk = mu**k / math.factorial(k) * e_inv
    at_least_one = 1 - e_inv
    c = nxt // mins
    assert c * mins == nxt
    return {
        "a": _frac(pk),
        "b": _frac(at_least_one),
        "c": _frac(pk / at_least_one),
        "d": _frac(pk * e_inv**c),
    }


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, ["c", "d"], HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_follow_through_from_wrong_a():
    def wrong_chain(q):
        pa_wrong = Fraction(q.parts[0].distractors[0])
        pb = Fraction(q.parts[1].answer_text)
        big_e_c = Fraction(q.parts[3].answer_text) / Fraction(q.parts[0].answer_text)  # (1/E)^c
        replies = {
            "a": str(pa_wrong),
            "b": q.parts[1].answer_text,
            "c": str(pa_wrong / pb),
            "d": str(pa_wrong * big_e_c),
        }
        return "a", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_b():
    def wrong_chain(q):
        pb_wrong = Fraction(q.parts[1].distractors[0])
        pa = Fraction(q.parts[0].answer_text)
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(pb_wrong),
            "c": str(pa / pb_wrong),
            "d": q.parts[3].answer_text,
        }
        return "b", replies, 6

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        result = grade(skill, q, {"a": "211/1013", "b": "307/1019", "c": "401/1031", "d": "503/1049"})
        assert result.marks == 0
