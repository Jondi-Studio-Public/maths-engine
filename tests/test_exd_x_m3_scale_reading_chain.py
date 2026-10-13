"""Tests for the exam source `exd_x_m3_scale_reading_chain` (skill `f10-unmarked-partial-measurement`)."""

from __future__ import annotations

import re
from fractions import Fraction

from tests.exd_x_m3_support import (
    check_answer_space,
    check_follow_through,
    check_grading,
    check_oracle,
    check_shapes,
    check_source_wired,
    check_tagged,
    check_text_and_determinism,
    fmt,
)

TOPIC, SKILL = "exd_x_m3_scale_reading_chain", "f10-unmarked-partial-measurement"
_FACTOR = {("mL", "L"): 1000, ("g", "kg"): 1000, ("mm", "m"): 1000, ("cm", "m"): 100}


def _inputs(q):
    lo, unit, hi = re.search(r"(\d+) (\w+)(?: mark)? and (?:the )?(\d+) \2", q.stem).group(1, 2, 3)
    n = int(re.search(r"(\d+) equal", q.stem).group(1))
    d1 = int(re.search(r"(\d+) divisions? (?:above|past)", q.stem).group(1))
    d2 = int(re.search(r"(\d+) divisions? (?:below|short)", q.stem).group(1))
    big = re.search(r"in (L|kg|m)\.", q.parts[3].prompt).group(1)
    dp = int(re.search(r"to (\d) decimal places", q.parts[3].prompt).group(1))
    return int(lo), int(hi), n, d1, d2, _FACTOR[(unit, big)], dp


def oracle(q):
    lo, hi, n, d1, d2, factor, dp = _inputs(q)
    division = Fraction(hi - lo, n)
    p_read, q_read = lo + d1 * division, hi - d2 * division
    return {
        "a": str(division),
        "b": str(p_read),
        "c": str(q_read),
        "d": fmt((p_read + q_read) / factor, dp),
    }


def test_source_wired():
    check_source_wired(SKILL, TOPIC)


def test_shapes_new_over_hard_tier():
    check_shapes(SKILL, TOPIC)


def test_text_clean_and_deterministic():
    check_text_and_determinism(TOPIC)


def test_key_matches_independent_derivation_over_300_seeds():
    check_oracle(TOPIC, oracle, 300)


def test_grader_accepts_key_and_rejects_every_distractor():
    check_grading(SKILL, TOPIC, 100)
    check_tagged(TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_division_size():
    def wrong(q):
        return str(int(oracle(q)["a"]) + 3)

    def downstream(q, a_reply):
        lo, hi, _, d1, d2, factor, dp = _inputs(q)
        division = int(a_reply)
        p_read, q_read = lo + d1 * division, hi - d2 * division
        return {
            "b": str(p_read),
            "c": str(q_read),
            "d": fmt(Fraction(p_read + q_read, factor), dp),
        }

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b", "c", "d"})
