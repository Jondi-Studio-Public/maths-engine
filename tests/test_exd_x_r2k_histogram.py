"""Tests for the exam source of senior-compare-the-characteristics-of-the-shape-of-histograms.

The mean, median, their difference and the shape are re-derived from the table in the rendered stem.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_histogram  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-compare-the-characteristics-of-the-shape-of-histograms"
TOPIC = "exd_x_r2k_histogram"
_ROW = re.compile(r"(\d+) to under (\d+): (\d+)")


def oracle(q) -> dict[str, str]:
    rows = [(int(a), int(b), int(f)) for a, b, f in _ROW.findall(q.stem)]
    n = sum(f for _, _, f in rows)
    mean = sum(Fraction(a + b, 2) * f for a, b, f in rows) / n
    cum = 0
    median = None
    for a, b, f in rows:
        if cum + f >= Fraction(n, 2):
            median = a + (Fraction(n, 2) - cum) / f * (b - a)
            break
        cum += f
    a_val, b_val = kit.half_up(mean, 1), kit.half_up(median, 1)
    diff = a_val - b_val
    width = rows[0][1] - rows[0][0]
    exact_gap = mean - median
    shape = "symmetric" if abs(exact_gap) < Fraction(width, 10) else ("right" if exact_gap > 0 else "left")
    return {"a": kit.fixed(mean, 1), "b": kit.fixed(median, 1), "c": kit.fixed(diff, 1), "d": shape}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_all_three_shapes_occur():
    shapes = {kit.canon(kit.exam_q(SKILL, s))["d"] for s in range(120)}
    assert shapes == {"right", "left", "symmetric"}


def test_grading_and_tags():
    kit.check_grading(SKILL, min_parts=4)
    kit.check_tags(SKILL)


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_follow_through_from_a_wrong_mean():
    def chain(q):
        c = kit.canon(q)
        a_wrong = Fraction(c["a"]) + Fraction(3, 2)
        diff = a_wrong - Fraction(c["b"])
        return "a", {**c, "a": kit.fixed(a_wrong, 1), "c": kit.fixed(diff, 1)}

    kit.check_follow_through(SKILL, chain)


def test_follow_through_from_a_wrong_median():
    def chain(q):
        c = kit.canon(q)
        b_wrong = Fraction(c["b"]) - Fraction(7, 10)
        diff = Fraction(c["a"]) - b_wrong
        return "b", {**c, "b": kit.fixed(b_wrong, 1), "c": kit.fixed(diff, 1)}

    kit.check_follow_through(SKILL, chain)


def test_shape_answer_is_spread():
    """Part (d) names right, left or symmetric in roughly equal shares (it was "right" on three seeds in five)."""
    from tests.exd_junk_probe import assert_answer_spread

    assert_answer_spread("exd_x_r2k_histogram", "d", max_share=0.42, at_least=3)
