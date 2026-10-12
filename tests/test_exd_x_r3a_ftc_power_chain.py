"""Exam chain of `ucv1-ftc-polynomial`: the coefficient, a second integral, then an upper limit."""

from __future__ import annotations

import math
import re

import pytest
import sympy

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_ftc_power_chain"
SKILL = S.load_skill("ucv1-ftc-polynomial")
X, A, K = sympy.symbols("x a k", positive=True)


def _read(q):
    m = re.search(r"p\(x\) = a (x(?:\^(\d))?), where .* from x = (\d+) to x = (\d+) is (\d+)\.", q.stem)
    n = int(m.group(2)) if m.group(2) else 1
    low, up, value = int(m.group(3)), int(m.group(4)), int(m.group(5))
    low2, up2 = map(int, re.search(r"from x = (\d+) to x = (\d+)\.", q.parts[1].prompt).groups())
    mult = {"twice": 2, "three times": 3}[re.search(r"is (twice|three times) the answer", q.parts[2].prompt).group(1)]
    return n, low, up, value, low2, up2, mult


def _half_up(x: float, places: int) -> str:
    return f"{math.floor(x * 10**places + 0.5) / 10**places:.{places}f}"


def oracle(q):
    n, low, up, value, low2, up2, mult = _read(q)
    (a,) = sympy.solve(sympy.integrate(A * X**n, (X, low, up)) - value, A)
    second = sympy.integrate(a * X**n, (X, low2, up2))
    roots = sympy.solve(sympy.integrate(a * X**n, (X, low, K)) - mult * second, K)
    (k,) = [r for r in roots if r.is_real and r > low]
    return {"a": str(a), "b": str(second), "c": _half_up(float(k.evalf(30)), 2)}


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
    assert {_read(q)[0] for q in questions} == {1, 2, 3}
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions[:120], oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_coefficient(questions):
    def chain(q):
        n, low, up, value, low2, up2, mult = _read(q)
        wrong = int(q.parts[0].distractors[1])  # the dropped-factor slip, a positive whole number
        if wrong <= 0:
            return None
        e = n + 1
        second = sympy.Rational(wrong * (up2**e - low2**e), e)
        radicand = sympy.Rational(e * mult) * second / wrong + low**e
        k = float(radicand ** sympy.Rational(1, e))
        if second.q != 1 or len({wrong, second}) < 2:
            return None
        return {"a": str(wrong), "b": str(second), "c": _half_up(k, 2)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=200, min_seen=10)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "999", "b": "888", "c": "77.77"})


def test_deterministic():
    S.check_deterministic(TOPIC)
