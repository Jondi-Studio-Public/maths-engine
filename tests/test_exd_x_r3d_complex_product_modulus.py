"""Exam chain of `complex-modulus`: Re(zw), Im(zw), then |zw|."""

from __future__ import annotations

import math
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_complex_product_modulus"
SKILL = S.load_skill("complex-modulus")
_TR = standard_transformations + (implicit_multiplication_application,)


def _complex(text: str):
    return parse_expr(text.replace("i", "I"), transformations=_TR, local_dict={"I": sympy.I})


def _numbers(q):
    z = re.search(r"z = (.+?)(?: and w|,)", q.stem).group(1)
    w = re.search(r"w = (.+?)(?:\.|,| and|$)", q.stem).group(1)
    return _complex(z), _complex(w)


def oracle(q):
    z, w = _numbers(q)
    product = sympy.expand(z * w)
    return {
        "a": str(sympy.re(product)),
        "b": str(sympy.im(product)),
        "c": str(sympy.sqrt(sympy.re(product) ** 2 + sympy.im(product) ** 2)),
    }


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_modulus_is_product_of_moduli(questions):
    for q in questions[:100]:
        z, w = _numbers(q)
        assert abs(z) ** 2 == abs(w) ** 2
        assert int(q.parts[2].answer_text) == abs(z) * abs(w)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions, minimum=3)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_parts(questions):
    def chain(q):
        a = int(q.parts[0].distractors[0])
        b = int(q.parts[1].distractors[0])
        c = round(math.sqrt(a * a + b * b))
        if c in (a, b):
            return None
        return {"a": str(a), "b": str(b), "c": str(c)}, {"c"}

    S.check_follow_through(SKILL, questions, chain)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999", "b": "8888", "c": "7777"})


def test_deterministic():
    S.check_deterministic(TOPIC)
