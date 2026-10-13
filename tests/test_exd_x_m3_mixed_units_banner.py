"""Tests for the exam source `exd_x_m3_mixed_units_banner` (skill `rectangle-area-perimeter-mixed-units`)."""

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

TOPIC, SKILL = "exd_x_m3_mixed_units_banner", "rectangle-area-perimeter-mixed-units"


def _inputs(q):
    L = Fraction(re.search(r"(\d+\.\d) m", q.stem).group(1))
    W = Fraction(re.search(r"(\d+) cm", q.stem).group(1))
    edge = int(re.search(r"\$(\d+) per metre", q.parts[1].prompt).group(1))
    paint = int(re.search(r"\$(\d+) per square metre", q.parts[3].prompt).group(1))
    return L, W, edge, paint


def oracle(q):
    L, W, edge, paint = _inputs(q)
    metres = W / 100  # centimetres to metres
    perimeter = 2 * (L + metres)
    area_m2 = L * metres
    return {
        "a": fmt(perimeter, 2),
        "b": fmt(perimeter * edge, 2),
        "c": fmt(area_m2 * 10000, 0),
        "d": fmt(area_m2 * paint, 2),
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


def test_follow_through_from_wrong_perimeter_and_area():
    def wrong(q):
        return fmt(Fraction(oracle(q)["a"]) + Fraction(3, 2), 2)

    def downstream(q, a_reply):
        _, _, edge, _ = _inputs(q)
        return {
            "b": fmt(Fraction(a_reply) * edge, 2),
            "c": oracle(q)["c"],
            "d": oracle(q)["d"],
        }

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b"})


def test_follow_through_from_wrong_area_in_square_centimetres():
    from mathgen.engine import grade
    from tests.exd_x_m3_support import exam_questions, skill

    sk = skill(SKILL)
    for q in exam_questions(TOPIC, 60):
        _, _, _, paint = _inputs(q)
        wrong_c = int(oracle(q)["c"]) + 5000
        reply = {
            **{p.label: p.answer_text for p in q.parts},
            "c": str(wrong_c),
            "d": fmt(Fraction(wrong_c, 10000) * paint, 2),
        }
        result = grade(sk, q, reply)
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks and result.parts["d"].follow_through is True
