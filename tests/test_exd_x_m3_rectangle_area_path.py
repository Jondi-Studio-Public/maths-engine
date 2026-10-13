"""Tests for the exam source `exd_x_m3_rectangle_area_path` (skill `rectangle-area`): area of a path round a garden."""

from __future__ import annotations

import re

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

TOPIC, SKILL = "exd_x_m3_rectangle_area_path", "rectangle-area"


def _numbers(q):
    stem = q.stem
    w = int(re.search(r"path (\d+) m wide", stem).group(1))
    rest = stem.replace(re.search(r"path (\d+) m wide", stem).group(0), "")
    L, W = (int(v) for v in re.findall(r"(\d+) m", rest))
    price = int(re.search(r"\$(\d+) per square metre", q.parts[3].prompt).group(1))
    return L, W, w, price


def oracle(q):
    L, W, w, price = _numbers(q)
    inner, outer = L * W, (L + 2 * w) * (W + 2 * w)
    return {"a": str(inner), "b": str(outer), "c": str(outer - inner), "d": str((outer - inner) * price)}


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


def test_follow_through_from_wrong_inner_area():
    def wrong(q):
        return str(int(oracle(q)["a"]) + 7)

    def downstream(q, a_reply):
        _, _, _, price = _numbers(q)
        want = oracle(q)
        c = int(want["b"]) - int(a_reply)
        return {"b": want["b"], "c": str(c), "d": str(c * price)}

    check_follow_through(SKILL, TOPIC, wrong, downstream, {"c", "d"})
