"""Tests the multi-part exam tier of `complex-conjugate`: a quotient, its conjugate and their product."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_complex_conjugate"
SKILL_ID = "complex-conjugate"
_SKILL = S.skill(SKILL_ID)
_CARTESIAN = re.compile(r"(-?\d+) ([+-]) (\d*)i")


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _complex(text: str) -> tuple[int, int]:
    a, sign, b = _CARTESIAN.fullmatch(text).groups()
    return int(a), (-1 if sign == "-" else 1) * int(b or 1)


def _z(q):
    num, den = re.search(r"z = \((.+)\)/\((.+)\)\.$", q.stem).groups()
    (p, qq), (r, s) = _complex(num), _complex(den)
    return complex(p, qq) / complex(r, s), (r, s)


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abcd", [1, 2, 2, 2])


def test_keys_match_independent_complex_arithmetic_over_300_seeds(questions):
    for q in questions:
        z, (r, s) = _z(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert int(a) == r * r + s * s, q.stem
        assert abs(float(Fraction(b)) - z.real) < 1e-9, q.stem
        assert abs(float(Fraction(c)) - z.conjugate().imag) < 1e-9, q.stem
        assert abs(float(Fraction(d)) - (z * z.conjugate()).real) < 1e-9, q.stem


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_denominator_modulus(questions):
    checked = 0
    for q in questions[:100]:
        z, (r, s) = _z(q)
        bad_a = q.parts[0].distractors[0]
        if int(bad_a) <= 0:
            continue
        scale = Fraction(int(q.parts[0].answer_text), int(bad_a))
        b = Fraction(q.parts[1].answer_text) * scale
        c = Fraction(q.parts[2].answer_text) * scale
        d = b * b + c * c
        if Fraction(bad_a) in (b, c, d, -c):  # a reply that only repeats an earlier one earns no follow-through
            continue
        reply = {"a": bad_a, "b": _fraction_text(b), "c": _fraction_text(c), "d": _fraction_text(d)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.stem, label, reply)
            assert result.parts[label].follow_through is True
        checked += 1
    assert checked >= 40


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999", "b": "7/3", "c": "5", "d": "11/2"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
