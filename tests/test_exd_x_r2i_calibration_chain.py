"""Tests for the multi-part exam source of ucv1-validation-calibration-check: mean prediction, confidence gap,
binned calibration error and the cancellation between bins. Keys are re-derived from the rendered stem."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mathgen.generators.exd_x_r2i_calibration_chain  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, skill_of, standard_checks

TOPIC = "exd_x_r2i_calibration_chain"
SKILL = "ucv1-validation-calibration-check"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]


def _fmt(x: Fraction) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _points(q):
    return [(Fraction(p), int(o)) for p, o in re.findall(r"\((\d\.\d), ([01])\)", q.stem)]


def _facts(q):
    pts = _points(q)
    n = len(pts)
    mean_p = sum(p for p, _ in pts) / n
    mean_o = Fraction(sum(o for _, o in pts), n)
    ece = Fraction(0)
    for in_bin in (lambda p: p < Fraction(1, 2), lambda p: p >= Fraction(1, 2)):
        grp = [(p, o) for p, o in pts if in_bin(p)]
        ece += Fraction(len(grp), n) * abs(
            sum(p for p, _ in grp) / len(grp) - Fraction(sum(o for _, o in grp), len(grp))
        )
    return n, mean_p, mean_o, mean_p - mean_o, ece


def oracle(q):
    n, mean_p, mean_o, gap, ece = _facts(q)
    assert n == 10
    return {"a": _fmt(mean_p), "b": _fmt(gap), "c": _fmt(ece), "d": _fmt(ece - abs(gap))}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"b", "d"})


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_oracle_agrees_with_every_part():
    CHECKS["oracle_agrees"]()


def test_composed_reply_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_are_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_bins_pull_in_opposite_directions():
    """The chain's last part is only worth asking when the two bins' signed gaps differ in sign."""
    for s in range(100):
        q = exam(TOPIC, s)
        pts = _points(q)
        signed = []
        for in_bin in (lambda p: p < Fraction(1, 2), lambda p: p >= Fraction(1, 2)):
            grp = [(p, o) for p, o in pts if in_bin(p)]
            signed.append(sum(p for p, _ in grp) / len(grp) - Fraction(sum(o for _, o in grp), len(grp)))
        assert signed[0] * signed[1] < 0
        assert Fraction(q.parts[3].answer_text) > 0


def test_wrong_mean_carries_into_the_gap():
    def chain(q):
        n, mean_p, mean_o, gap, ece = _facts(q)
        wrong_a = mean_p + Fraction(1, 10)
        wrong_b = wrong_a - mean_o
        replies = {"a": _fmt(wrong_a), "b": _fmt(wrong_b), "c": _fmt(ece), "d": _fmt(ece - abs(wrong_b))}
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_error_carries_into_the_cancellation():
    def chain(q):
        n, mean_p, mean_o, gap, ece = _facts(q)
        wrong_c = ece + Fraction(1, 10)
        replies = {"a": _fmt(mean_p), "b": _fmt(gap), "c": _fmt(wrong_c), "d": _fmt(wrong_c - abs(gap))}
        return "c", replies, sum(MARKS) - MARKS[2]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_gap_is_not_rescued_by_a_right_later_part():
    skill = skill_of(SKILL)
    from mathgen.engine import grade

    for s in range(40):
        q = exam(TOPIC, s)
        n, mean_p, mean_o, gap, ece = _facts(q)
        replies = {"a": _fmt(mean_p), "b": _fmt(-gap), "c": _fmt(ece), "d": _fmt(ece - abs(gap))}
        result = grade(skill, q, replies)
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == MARKS[3]
