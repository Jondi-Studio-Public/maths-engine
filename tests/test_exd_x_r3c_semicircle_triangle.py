"""Tests the multi-part exam source of `senior-an-angle-in-a-semicircle-is-a-right`.

The oracle reads the diameter (or radius) and the chord from the rendered stem and places the triangle on
coordinates: the circle is x^2 + y^2 = r^2 and C is found from the given chord.
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

from mathgen.engine import grade
from tests.exd_x_r3c_support import (
    check_all_wrong,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_full_marks_and_distractors,
    check_oracle,
    check_shapes,
    check_text,
    check_wired,
    exam_questions,
    skill_for,
)

TOPIC = "exd_x_r3c_semicircle_triangle"
SKILL_ID = "senior-an-angle-in-a-semicircle-is-a-right"
SKILL = skill_for(SKILL_ID)


def parse(q):
    """(diameter, chord): the first length in the stem is the diameter, or the radius when the stem says so."""
    first, chord = (int(n) for n in re.findall(r"(\d+) (?:cm|m)\b", q.stem))
    return (2 * first if "radius" in q.stem else first), chord


def _dp1(x):
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def oracle(q):
    d, a = parse(q)
    r = d / 2
    # C = (cx, cy) on the circle with |AC| = a, A = (-r, 0)
    cx = -r + a * a / (2 * r)
    cy = math.sqrt(r * r - cx * cx)
    b = math.hypot(cx - r, cy)
    area = 0.5 * d * cy
    return {"a": str(round(b)), "b": str(round(area)), "c": _dp1(round(cy, 9))}


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=40)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_chord():
    """A BC that is two too long, used correctly in the area and the height."""
    done = 0
    for q in exam_questions(TOPIC, 60):
        d, a = parse(q)
        b = int(q.parts[0].answer_text) + 2
        area = a * b // 2 if (a * b) % 2 == 0 else None
        if area is None:
            continue
        height = _dp1(Decimal(2 * area) / Decimal(d))
        got = grade(SKILL, q, {"a": str(b), "b": str(area), "c": height})
        assert got.parts["a"].marks == 0
        for label in "bc":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, q.text
        done += 1
    assert done >= 10


def test_follow_through_from_a_wrong_area():
    for q in exam_questions(TOPIC, 40):
        d, a = parse(q)
        wrong = int(q.parts[1].distractors[0])
        height = _dp1(Decimal(2 * wrong) / Decimal(d))
        got = grade(SKILL, q, {"a": q.parts[0].answer_text, "b": str(wrong), "c": height})
        assert got.parts["b"].marks == 0
        assert got.parts["c"].marks == got.parts["c"].max_marks and got.parts["c"].follow_through, q.text


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abc")
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
