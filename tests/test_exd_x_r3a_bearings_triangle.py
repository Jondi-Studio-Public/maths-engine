"""Exam chain of `senior-solve-problems-involving-bearings`: interior angle, cosine rule, bearing of the finish."""

from __future__ import annotations

import math
import re

import pytest

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_bearings_triangle"
SKILL = S.load_skill("senior-solve-problems-involving-bearings")


def _half_up(x: float, places: int) -> float:
    return math.floor(x * 10**places + 0.5) / 10**places


def _legs(q):
    m = re.search(
        r"(\d+) km on a bearing of (\d+)° from A to B, then (\d+) km on a bearing of (\d+)° from B to C", q.stem
    )
    d1, b1, d2, b2 = map(int, m.groups())
    return d1, b1, d2, b2


def oracle(q):
    d1, b1, d2, b2 = _legs(q)
    # Coordinates: east and north of C from A; the interior angle is the angle between BA and BC.
    east = d1 * math.sin(math.radians(b1)) + d2 * math.sin(math.radians(b2))
    north = d1 * math.cos(math.radians(b1)) + d2 * math.cos(math.radians(b2))
    angle_b = abs(((b2 - b1 + 180) % 360) - 180)  # the turn, 0..180
    interior = 180 - angle_b
    dist = _half_up(math.hypot(east, north), 1)
    bearing = int(_half_up(math.degrees(math.atan2(east, north)) % 360, 0)) % 360
    return {"a": str(interior), "b": f"{dist:.1f}", "c": str(bearing)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_angle(questions):
    def chain(q):
        d1, b1, d2, b2 = _legs(q)
        a = int(q.parts[0].distractors[0])  # the turn angle used as the interior angle
        if not 0 < a < 180:
            return None
        dist = _half_up(math.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * math.cos(math.radians(a))), 1)
        turn_right = (b2 - b1) % 360 < 180
        cos_a = (d1**2 + dist**2 - d2**2) / (2 * d1 * dist)
        angle = math.degrees(math.acos(cos_a))
        bearing = int(_half_up((b1 + (angle if turn_right else -angle)) % 360, 0)) % 360
        return {"a": str(a), "b": f"{dist:.1f}", "c": str(bearing)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "999", "b": "1.1", "c": "2"})


def test_deterministic():
    S.check_deterministic(TOPIC)
