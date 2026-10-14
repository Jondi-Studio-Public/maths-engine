"""Tests for the exam source `exd_x_m3_window_perimeter_chain` (skill `senior-calculate-perimeters-...`)."""

from __future__ import annotations

import re
from decimal import Decimal
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

TOPIC = "exd_x_m3_window_perimeter_chain"
SKILL = "senior-calculate-perimeters-of-familiar-shapes-including-triangles-squares"
PI = Fraction(Decimal("3.14159265358979323846264338327950288419716939937510"))


def _inputs(q):
    return tuple(int(v) for v in re.search(r"(\d+) cm wide and (\d+) cm high", q.stem).groups())


def oracle(q):
    w, h = _inputs(q)
    arc = PI * w / 2
    perimeter = arc + w + 2 * h
    return {"a": fmt(arc, 1), "b": fmt(perimeter, 1), "c": fmt((perimeter - 2 * w) / 2, 1)}


def test_text_clean_and_deterministic():
    check_text_and_determinism(TOPIC)


def test_key_matches_independent_derivation_over_300_seeds():
    check_oracle(TOPIC, oracle, 300)


def test_grader_accepts_key_and_rejects_every_distractor():
    check_grading(SKILL, TOPIC, 100)
    check_tagged(TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_arc():
    def wrong(q):
        return fmt(Fraction(oracle(q)["a"]) + 8, 1)

    def downstream(q, a_reply):
        w, h = _inputs(q)
        b = Fraction(fmt(Fraction(a_reply) + w + 2 * h, 1))
        return {"b": str(b), "c": fmt((b - 2 * w) / 2, 1)}

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b", "c"})
