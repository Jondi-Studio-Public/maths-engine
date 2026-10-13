"""Tests for the exam source of senior-interpret-commonly-used-probability-statements-including-possible-probable.

P(both), P(at least one) and the convention's term are re-derived from the two stated percentages.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_probability_language  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-interpret-commonly-used-probability-statements-including-possible-probable"
TOPIC = "exd_x_r2k_probability_language"


def oracle(q) -> dict[str, str]:
    p, r = (int(v) for v in re.findall(r"is (\d+)%", q.stem))
    both = Fraction(p * r, 100)
    either = p + r - both
    x = either / 100
    term = (
        "possible"
        if x <= Fraction(3, 10)
        else "probable"
        if x <= Fraction(11, 20)
        else "likely"
        if x < Fraction(9, 10)
        else "certain"
    )
    return {"a": kit.fixed(both, 2), "b": kit.fixed(either, 2), "c": term}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_400_seeds():
    bad = [s for s in range(400) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_every_term_occurs_and_edges_are_reached():
    terms = [kit.canon(kit.exam_q(SKILL, s))["c"] for s in range(400)]
    assert set(terms) == {"possible", "probable", "likely", "certain"}
    on_edge = [s for s in range(700) if kit.canon(kit.exam_q(SKILL, s))["b"] in ("30.00", "55.00")]
    assert on_edge, "no draw lands on a range edge"


def test_grading_and_tags():
    kit.check_grading(SKILL, min_parts=3)
    kit.check_tags(SKILL)


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_follow_through_from_a_wrong_both():
    def chain(q):
        p, r = (int(v) for v in re.findall(r"is (\d+)%", q.stem))
        c = kit.canon(q)
        a_wrong = Fraction(c["a"]) + Fraction(5, 2)
        return "a", {**c, "a": kit.fixed(a_wrong, 2), "b": kit.fixed(p + r - a_wrong, 2)}

    kit.check_follow_through(SKILL, chain)
