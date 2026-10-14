"""Tests the multi-part exam source of `senior-define-and-use-unit-vectors-and-the-perpendicular`.

The oracle reads A, B, the length and the turn from the rendered stem and works the answers out with complex
numbers (multiplying by i turns a vector anticlockwise through 90 degrees).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

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

TOPIC = "exd_x_r3c_unit_vector_perp"
SKILL_ID = "senior-define-and-use-unit-vectors-and-the-perpendicular"
SKILL = skill_for(SKILL_ID)
_POINTS = re.compile(r"\(?(-?\d+), (-?\d+)\)")
_LENGTH = re.compile(r"(?:magnitude|further|length) (\d+)")
_TURN = re.compile(r"90° (anticlockwise|clockwise)")


def parse(q):
    (xa, ya), (xb, yb) = _POINTS.findall(q.stem)
    length = _LENGTH.search(q.stem).group(1)
    return int(xa), int(ya), int(xb), int(yb), int(length), _TURN.search(q.stem).group(1) == "anticlockwise"


def oracle(q):
    xa, ya, xb, yb, length, anti = parse(q)
    ab = complex(xb - xa, yb - ya)
    mag = math.isqrt(round(abs(ab) ** 2))
    assert mag * mag == round(abs(ab) ** 2)
    turned = ab * (1j if anti else -1j)  # a turn through 90 degrees is multiplication by i (or -i)
    wi = Fraction(round(turned.real)) * length / mag
    return {
        "a": str(mag),
        "b": str(Fraction(round(ab.imag), mag)),
        "c": str(wi),
        "d": str(xb + wi),
    }


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=30)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_magnitude():
    def chain(q, wrong):
        xa, ya, xb, yb, length, anti = parse(q)
        uy = Fraction(yb - ya) / int(wrong)
        ux = Fraction(xb - xa) / int(wrong)
        wi = length * (-uy if anti else uy)
        return {"b": str(uy), "c": str(wi), "d": str(xb + wi)}

    check_follow_through(SKILL, TOPIC, chain, n=30)


def test_follow_through_from_a_wrong_unit_component():
    for q in exam_questions(TOPIC, 30):
        xa, ya, xb, yb, length, anti = parse(q)
        wrong = q.parts[1].distractors[0]
        wi = length * (-Fraction(wrong) if anti else Fraction(wrong))
        canon = {p.label: p.answer_text for p in q.parts}
        got = grade(SKILL, q, {**canon, "b": wrong, "c": str(wi), "d": str(xb + wi)})
        assert got.parts["b"].marks == 0
        for label in "cd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "fraction"]
        assert all(p.follow_through for p in q.parts[1:])


def test_deterministic():
    check_deterministic(TOPIC)
