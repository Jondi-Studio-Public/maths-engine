"""Tests for the exam source `exd_x_m3_rectangle_perimeter_unknown` (skill `rectangle-perimeter`)."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from tests.exd_x_m3_support import (
    check_answer_space,
    check_follow_through,
    check_grading,
    check_oracle,
    check_shapes,
    check_source_wired,
    check_tagged,
    check_text_and_determinism,
)

TOPIC, SKILL = "exd_x_m3_rectangle_perimeter_unknown", "rectangle-perimeter"


def _pd(q):
    P, d = (int(v) for v in re.findall(r"(\d+) m", q.stem)[:2])
    return P, d


def oracle(q):
    P, d = _pd(q)
    w = sympy.Symbol("w")
    width = sympy.solve(sympy.Eq(2 * (w + (w + d)), P), w)[0]
    return {"a": str(width), "b": str(width + d), "c": str(width * (width + d))}


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


def test_follow_through_from_wrong_width():
    def wrong(q):
        return str(int(oracle(q)["a"]) + 3)

    def downstream(q, a_reply):
        _, d = _pd(q)
        w = int(a_reply)
        return {"b": str(w + d), "c": str(w * (w + d))}

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b", "c"})
    assert Fraction(1) == 1
