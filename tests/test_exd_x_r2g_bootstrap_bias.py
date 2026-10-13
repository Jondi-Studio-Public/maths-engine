"""Tests for the multi-part exam source of ucv1-bootstrap-summary: bootstrap mean, bias, corrected estimate, interval.

Each part's key is re-derived from the rendered stem with exact fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import mathgen.generators.exd_x_r2g_bootstrap_bias  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2g_bootstrap_bias"
SKILL = "ucv1-bootstrap-summary"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 1, 1, 2]


def _two_dp(x: Fraction) -> str:
    return f"{math.floor(x * 100 + Fraction(1, 2)) / 100:.2f}"


def _parse(q):
    th = int(re.search(r"sample mean is (\d+) ", q.stem).group(1))
    values = [int(v) for v in re.findall(r": ([\d, ]{20,})\.", q.stem)[-1].split(", ")]
    return th, values


def oracle(q):
    th, values = _parse(q)
    ordered = sorted(values)
    mean = Fraction(sum(values), 20)
    bias = mean - th
    return {
        "a": _two_dp(mean),
        "b": _two_dp(bias),
        "c": _two_dp(th - bias),
        "d": str(2 * th - ordered[18]),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"b", "c"}, "p14e_bootstrap_percentile")


def test_source_wired():
    CHECKS["wired"]()


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_follow_through_from_wrong_mean():
    """A wrong (a) used correctly in (b) and (c) earns both; (d) is independent and stays right."""

    def wrong_chain(q):
        th, values = _parse(q)
        ordered = sorted(values)
        wrong = Fraction(sum(values), 19)  # divided by n - 1
        wrong_text = _two_dp(wrong)
        wrong = Fraction(wrong_text)
        bias = wrong - th
        replies = {
            "a": wrong_text,
            "b": _two_dp(bias),
            "c": _two_dp(th - Fraction(_two_dp(bias))),
            "d": str(2 * th - ordered[18]),
        }
        return "a", replies, 4

    follow_through_check(TOPIC, SKILL, wrong_chain)


def test_follow_through_from_wrong_bias():
    def wrong_chain(q):
        th, values = _parse(q)
        ordered = sorted(values)
        mean = Fraction(sum(values), 20)
        wrong_bias = Fraction(_two_dp(-(mean - th)))  # sign reversed
        replies = {
            "a": _two_dp(mean),
            "b": _two_dp(wrong_bias),
            "c": _two_dp(th - wrong_bias),
            "d": str(2 * th - ordered[18]),
        }
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL, wrong_chain)


def test_wrong_percentile_rank_loses_only_part_d():
    from mathgen.engine import grade
    from tests.exd_x_r2g_support import skill_of

    skill = skill_of(SKILL)
    q = exam(TOPIC, 3)
    canon = {p.label: p.answer_text for p in q.parts}
    th, values = _parse(q)
    result = grade(skill, q, {**canon, "d": str(2 * th - sorted(values)[0])})
    assert result.marks == 4 and result.parts["d"].marks == 0
