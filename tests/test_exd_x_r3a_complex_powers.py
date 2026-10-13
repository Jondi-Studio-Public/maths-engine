"""Exam chain of `senior-define-the-imaginary-number-i-as-a-root`: z^2, then z^4 or z^3, then z^5 or z^4."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_complex_powers"
SKILL = S.load_skill("senior-define-the-imaginary-number-i-as-a-root")


def _read(text: str) -> sympy.Expr:
    return sympy.sympify(re.sub(r"(\d)i", r"\1*I", text).replace("i", "I"))


def _show(z) -> str:
    re_, im = sympy.re(z), sympy.im(z)
    if im == 0:
        return str(re_)
    mag = "i" if abs(im) == 1 else f"{abs(im)}i"
    sign = "-" if im < 0 else ""
    return f"{sign}{mag}" if re_ == 0 else f"{re_} {'-' if im < 0 else '+'} {mag}"


def _exponents(q):
    return [int(m) for m in re.findall(r"write \w\^(\d)", q.text)]


def oracle(q):
    z = _read(re.search(r"Let \w = (.*), where", q.stem).group(1))
    n2, n3 = _exponents(q)
    return {
        "a": _show(sympy.expand(z**2)),
        "b": _show(sympy.expand(z**n2)),
        "c": _show(sympy.expand(z**n3)),
    }


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "expression" for p in q.parts)
    assert {tuple(_exponents(q)) for q in questions} == {(4, 5), (3, 4)}
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions, n=40)
    S.check_distractors_rejected(SKILL, questions, n=30)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_square(questions):
    def chain(q):
        z = _read(re.search(r"Let \w = (.*), where", q.stem).group(1))
        a_wrong = _read(q.parts[0].distractors[0])
        n2, n3 = _exponents(q)
        second = sympy.expand(a_wrong**2) if n2 == 4 else sympy.expand(a_wrong * z)
        third = sympy.expand(second * z)
        return {"a": _show(a_wrong), "b": _show(second), "c": _show(third)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=40)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "7", "b": "3i", "c": "1 - 9i"})


def test_deterministic():
    S.check_deterministic(TOPIC)
