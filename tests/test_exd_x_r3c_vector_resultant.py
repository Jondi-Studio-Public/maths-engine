"""Tests the multi-part exam source of `senior-express-a-vector-in-component-form-using-the`.

The oracle reads a, b, k and the length from the rendered stem and works r = a - kb out component by component.
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

TOPIC = "exd_x_r3c_vector_resultant"
SKILL_ID = "senior-express-a-vector-in-component-form-using-the"
SKILL = skill_for(SKILL_ID)
_VEC = r"((?:-?\d*[ij])(?: [+-] \d*[ij])?)"
_STEM = re.compile(rf"a = {_VEC} and b = {_VEC}.*a - (\d)b")
_LENGTH = re.compile(r"magnitude (\d+) and points")


def vec(text):
    out = {"i": 0, "j": 0}
    for sign, mag, axis in re.findall(r"([+-]?)\s*(\d*)([ij])", text.replace(" ", "")):
        out[axis] = (-1 if sign == "-" else 1) * int(mag or 1)
    return out["i"], out["j"]


def parse(q):
    a, b, k = _STEM.search(q.stem).groups()
    return vec(a), vec(b), int(k), int(_LENGTH.search(q.parts[3].prompt).group(1))


def oracle(q):
    (a1, a2), (b1, b2), k, length = parse(q)
    r1, r2 = a1 - k * b1, a2 - k * b2
    mag = math.hypot(r1, r2)
    assert mag == round(mag)
    return {"a": str(r1), "b": str(r2), "c": f"{round(mag)}.0", "d": f"{length * r2 // round(mag)}.0"}


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


def _dp1(x):
    return Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def test_follow_through_from_a_wrong_coefficient():
    """A wrong j-coefficient (b) carried through the magnitude and the scaled vector earns (c) and (d)."""
    for q in exam_questions(TOPIC, 40):
        (a1, a2), (b1, b2), k, length = parse(q)
        r1 = a1 - k * b1
        wrong = int(q.parts[1].distractors[0])
        mag = _dp1(math.sqrt(r1 * r1 + wrong * wrong))
        if mag in (r1, wrong):
            continue  # the magnitude merely repeats an earlier reply: the marker withholds follow-through
        reply = {"a": str(r1), "b": str(wrong), "c": str(mag), "d": str(_dp1(Decimal(length * wrong) / mag))}
        got = grade(SKILL, q, reply)
        assert got.parts["b"].marks == 0
        for label in "cd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, q.text


def test_follow_through_from_a_wrong_magnitude():
    for q in exam_questions(TOPIC, 40):
        (a1, a2), (b1, b2), k, length = parse(q)
        r2 = a2 - k * b2
        wrong = q.parts[2].distractors[0]
        reply = {p.label: p.answer_text for p in q.parts}
        reply.update({"c": wrong, "d": str(_dp1(Decimal(length * r2) / Decimal(wrong)))})
        got = grade(SKILL, q, reply)
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
