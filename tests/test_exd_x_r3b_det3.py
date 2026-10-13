"""Tests the multi-part exam tier of `matrix-determinant-3x3`: a determinant with an unknown entry."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_det3"
SKILL_ID = "matrix-determinant-3x3"
_SKILL = S.skill(SKILL_ID)
k = sympy.Symbol("k")


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _matrix(q):
    body = re.search(r"\[([^\]]+)\]", q.stem).group(1)
    return sympy.Matrix([[k if v == "k" else int(v) for v in row.split()] for row in body.split(";")])


def _parse(text: str):
    return sympy.sympify(re.sub(r"(\d)k", r"\1*k", text))


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abcd", [2, 1, 2, 2])


def test_keys_match_sympy_determinant_over_300_seeds(questions):
    for q in questions:
        det = _matrix(q).det()
        assert sympy.expand(_parse(q.parts[0].answer_text) - det) == 0, q.stem
        k0 = int(re.search(r"k = (-?\d+)", q.parts[1].prompt).group(1))
        assert int(q.parts[1].answer_text) == det.subs(k, k0), q.stem
        c = int(re.search(r"det\((\d)A", q.parts[2].prompt).group(1))
        assert Fraction(q.parts[2].answer_text) == Fraction(str(sympy.Rational(c**3) / det.subs(k, k0))), q.stem
        root = sympy.solve(det, k)[0]
        assert Fraction(q.parts[3].answer_text) == Fraction(int(root.p), int(root.q)), q.stem


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_expression(questions):
    checked = 0
    for q in questions[:100]:
        wrong = q.parts[0].distractors[0]
        expr = _parse(wrong)
        k0 = int(re.search(r"k = (-?\d+)", q.parts[1].prompt).group(1))
        value = expr.subs(k, k0)
        c = int(re.search(r"det\((\d)A", q.parts[2].prompt).group(1))
        if value == 0 or str(value) == q.parts[1].answer_text:
            continue
        c_val = sympy.Rational(c**3) / value
        coeff = sympy.Poly(expr, k)
        root = sympy.solve(expr, k)
        if not root:
            continue
        reply = {
            "a": wrong,
            "b": str(value),
            "c": f"{c_val.p}/{c_val.q}" if c_val.q != 1 else str(c_val.p),
            "d": f"{root[0].p}/{root[0].q}" if root[0].q != 1 else str(root[0].p),
        }
        assert coeff.degree() <= 1
        if len(set(reply.values())) < 4:  # a reply that only repeats another earns no follow-through
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.stem, label, reply)
            if reply[label] != q.parts["abcd".index(label)].answer_text:
                assert result.parts[label].follow_through is True
        checked += 1
    assert checked >= 40


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "5k + 1", "b": "777", "c": "3/7", "d": "91"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
