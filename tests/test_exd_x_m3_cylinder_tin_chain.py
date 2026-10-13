"""Tests for the exam source `exd_x_m3_cylinder_tin_chain` (skill `f10-volume-surface-area-prisms-cylinders`)."""

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

TOPIC, SKILL = "exd_x_m3_cylinder_tin_chain", "f10-volume-surface-area-prisms-cylinders"
PI = Fraction("3.14")


def _inputs(q):
    D = int(re.search(r"diameter (?:of|is) (\d+) cm", q.stem).group(1))
    H = int(re.search(r"height (?:of|is) (\d+) cm", q.stem).group(1))
    return Fraction(D, 2), H  # radius, height


def oracle(q):
    r, H = _inputs(q)
    vol = PI * r * r * H
    curved = 2 * PI * r * H
    total = curved + 2 * PI * r * r
    return {"a": fmt(vol, 1), "b": fmt(vol / 1000, 2), "c": fmt(curved, 1), "d": fmt(total, 1)}


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


def test_follow_through_from_wrong_volume_and_label():
    def wrong(q):
        return fmt(Fraction(oracle(q)["a"]) + 100, 1)

    def downstream(q, a_reply):
        r, _ = _inputs(q)
        want = oracle(q)
        return {
            "b": fmt(Fraction(a_reply) / 1000, 2),
            "c": want["c"],
            "d": want["d"],
        }

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b"})


def test_follow_through_from_wrong_label_area():
    from mathgen.engine import grade
    from tests.exd_x_m3_support import exam_questions, skill

    sk = skill(SKILL)
    for q in exam_questions(TOPIC, 60):
        r, _ = _inputs(q)
        wrong_c = Fraction(oracle(q)["c"]) + 20
        reply = {
            **{p.label: p.answer_text for p in q.parts},
            "c": fmt(wrong_c, 1),
            "d": fmt(wrong_c + 2 * PI * r * r, 1),
        }
        result = grade(sk, q, reply)
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks and result.parts["d"].follow_through is True
