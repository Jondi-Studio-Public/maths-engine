"""Tests the multi-part exam source of `complex-imaginary-part`: Re(z^2), Im(z^2), Im(z^3).

The oracle reads z from the rendered stem and expands it with SymPy.
"""

from __future__ import annotations

import re

import sympy

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

TOPIC = "exd_x_r3c_complex_cube"
SKILL_ID = "complex-imaginary-part"
SKILL = skill_for(SKILL_ID)
_Z = re.compile(r"z = (-?\d+) ([+-]) (\d*)i\.")


def parse(q):
    a, sign, mag = _Z.search(q.stem).groups()
    return int(a), (-1 if sign == "-" else 1) * int(mag or 1)


def oracle(q):
    a, b = parse(q)
    z = a + b * sympy.I
    sq, cube = sympy.expand(z**2), sympy.expand(z**3)
    return {"a": str(sympy.re(sq)), "b": str(sympy.im(sq)), "c": str(sympy.im(cube))}


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=60)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_real_part():
    def chain(q, wrong):
        a, b = parse(q)
        im2 = 2 * a * b
        return {"b": str(im2), "c": str(int(wrong) * b + im2 * a)}

    check_follow_through(SKILL, TOPIC, chain, n=60, ft={"c"})


def test_follow_through_from_a_wrong_imaginary_part():
    for q in exam_questions(TOPIC, 40):
        from mathgen.engine import grade

        a, b = parse(q)
        wrong = q.parts[1].distractors[0]
        reply = {"a": str(a * a - b * b), "b": wrong, "c": str((a * a - b * b) * b + int(wrong) * a)}
        got = grade(SKILL, q, reply)
        assert got.parts["b"].marks == 0
        assert got.parts["c"].marks == got.parts["c"].max_marks and got.parts["c"].follow_through


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 1, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]


def test_deterministic():
    check_deterministic(TOPIC)
