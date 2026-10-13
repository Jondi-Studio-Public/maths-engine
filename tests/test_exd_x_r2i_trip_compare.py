"""Tests for the multi-part exam source of senior-compare-the-time-taken-to-travel-a-specific: a direct trip against
a faster service with a fixed extra time. Keys are re-derived from the rendered stem."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mathgen.generators.exd_x_r2i_trip_compare  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_trip_compare"
SKILL = "senior-compare-the-time-taken-to-travel-a-specific"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 1, 2]


def _fmt1(x: Fraction) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _facts(q):
    distance = int(re.search(r"a (\d+) km trip", q.stem).group(1))
    v1 = int(re.search(r"By (?:car|bike|ferry), the average speed is (\d+) km/h", q.stem).group(1))
    m = re.search(r"By [a-z ]+, the average speed is (\d+) (km/h|m/s), but it takes (\d+) minutes", q.stem)
    v2 = Fraction(int(m.group(1))) * (Fraction(18, 5) if m.group(2) == "m/s" else 1)
    return distance, v1, v2, int(m.group(3))


def oracle(q):
    distance, v1, v2, wait = _facts(q)
    direct = Fraction(60 * distance, v1)
    total = Fraction(60 * distance) / v2 + wait
    return {
        "a": str(direct),
        "b": str(total),
        "c": str(abs(direct - total)),
        "d": _fmt1(Fraction(60 * distance) / total),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"c", "d"})


def test_source_wired():
    CHECKS["wired"]()


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


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_wait_is_not_forgotten_and_speed_units_vary():
    seen_ms = seen_kmh = False
    for s in range(120):
        q = exam(TOPIC, s)
        seen_ms |= "m/s" in q.stem
        seen_kmh |= "m/s" not in q.stem
        distance, v1, v2, wait = _facts(q)
        assert int(q.parts[1].answer_text) > wait
    assert seen_ms and seen_kmh


def test_wrong_total_carries_into_gap_and_speed():
    def chain(q):
        distance, v1, v2, wait = _facts(q)
        direct = Fraction(60 * distance, v1)
        wrong_b = Fraction(60 * distance) / v2 + wait + 7  # a slip in the ride time
        replies = {
            "a": str(direct),
            "b": str(wrong_b),
            "c": str(abs(direct - wrong_b)),
            "d": _fmt1(Fraction(60 * distance) / wrong_b),
        }
        return "b", replies, sum(MARKS) - MARKS[1]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_direct_time_carries_into_the_gap_only():
    def chain(q):
        distance, v1, v2, wait = _facts(q)
        total = Fraction(60 * distance) / v2 + wait
        wrong_a = Fraction(60 * distance, v1) + 9
        replies = {
            "a": str(wrong_a),
            "b": str(total),
            "c": str(abs(wrong_a - total)),
            "d": _fmt1(Fraction(60 * distance) / total),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)
