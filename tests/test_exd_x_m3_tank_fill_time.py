"""Tests for the exam source `exd_x_m3_tank_fill_time` (skill `prism-capacity-metric-units`)."""

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
)

TOPIC, SKILL = "exd_x_m3_tank_fill_time", "prism-capacity-metric-units"


def _inputs(q):
    L = Fraction(re.search(r"(\d+\.\d) m", q.stem).group(1))
    W, H = (int(v) for v in re.search(r"(\d+) cm wide and (\d+) cm", q.stem).groups())
    r = int(re.search(r"(\d+) litres", q.stem).group(1))
    t = int(re.search(r"(\d+) minutes", q.stem).group(1))
    return L, W, H, r, t


def oracle(q):
    L, W, H, r, t = _inputs(q)
    litres = L * 100 * W * H / 1000  # cubic centimetres to litres
    left = litres - r * t
    return {"a": str(litres), "b": str(left), "c": str(left / r)}


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


def test_follow_through_from_wrong_capacity():
    def wrong(q):
        return str(int(oracle(q)["a"]) + 3 * _inputs(q)[3])

    def downstream(q, a_reply):
        _, _, _, r, t = _inputs(q)
        left = int(a_reply) - r * t
        return {"b": str(left), "c": str(Fraction(left, r))}

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"b", "c"})
