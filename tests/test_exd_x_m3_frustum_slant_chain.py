"""Tests for the exam source `exd_x_m3_frustum_slant_chain` (skill `frustum-volume`)."""

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
    fmt,
)

TOPIC, SKILL = "exd_x_m3_frustum_slant_chain", "frustum-volume"
PI = Fraction("3.14159")


def _inputs(q):
    R = int(re.search(r"base radius (?:of|is) (\d+) cm", q.stem).group(1))
    r = int(re.search(r"top radius (?:of|is) (\d+) cm", q.stem).group(1))
    s = int(re.search(r"slant height (?:of|is) (\d+) cm", q.stem).group(1))
    return R, r, s


def _cone(radius, height):
    return PI * radius * radius * height / 3


def oracle(q):
    R, r, s = _inputs(q)
    h = sympy.sqrt(s * s - (R - r) ** 2)
    assert h.is_Integer
    h = Fraction(int(h))
    frustum = PI * h * (R * R + R * r + r * r) / 3
    small = _cone(r, h * r / (R - r))  # similar triangles: small height / r = h / (R - r)
    whole = _cone(R, h * R / (R - r))
    return {"a": str(h), "b": fmt(frustum, 1), "c": fmt(small, 1), "d": fmt(whole, 1)}


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


def test_follow_through_from_wrong_height():
    def wrong(q):
        return str(int(oracle(q)["a"]) + 2)

    def downstream(q, a_reply):
        R, r, _ = _inputs(q)
        h = Fraction(int(a_reply))
        frustum = Fraction(fmt(PI * h * (R * R + R * r + r * r) / 3, 1))
        small = Fraction(fmt(_cone(r, h * r / (R - r)), 1))
        return {"b": str(frustum), "c": str(small), "d": fmt(frustum + small, 1)}

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b", "c", "d"})
