"""Exam chain of `senior-review-sine-cosine-and-tangent-as-ratios-of`: sin A, hypotenuse, leg, altitude."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_right_triangle_chain"
SKILL = S.load_skill("senior-review-sine-cosine-and-tangent-as-ratios-of")


def _show(v) -> str:
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def oracle(q):
    m = re.search(r"(tan|cos)\(A\) = (\d+)/(\d+) and BC = (\d+) cm", q.stem)
    kind, p, r, length = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    ratio = sympy.Rational(p, r)
    sin_a = sympy.Symbol("s", positive=True)
    # Solve for sin A from the given ratio and sin^2 + cos^2 = 1 (A acute).
    if kind == "cos":
        s = sympy.sqrt(1 - ratio**2)
    else:
        s = sympy.solve(sympy.Eq(sin_a**2, ratio**2 * (1 - sin_a**2)), sin_a)[0]
    ac = length / s
    ab = sympy.sqrt(ac**2 - length**2)
    bd = ab * length / ac
    return {"a": _show(sympy.nsimplify(s)), "b": _show(ac), "c": _show(ab), "d": _show(bd)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_sine(questions):
    def chain(q):
        m = re.search(r"BC = (\d+) cm", q.stem)
        length = Fraction(m.group(1))
        a_wrong = Fraction(q.parts[0].distractors[0])
        if not 0 < a_wrong < 1:
            return None
        ac = length / a_wrong
        if ac <= length:
            return None
        ab2 = ac * ac - length * length
        # keep an exact square root only: a learner would otherwise round
        import math

        num, den = ab2.numerator, ab2.denominator
        if math.isqrt(num) ** 2 != num or math.isqrt(den) ** 2 != den:
            return None
        ab = Fraction(math.isqrt(num), math.isqrt(den))
        bd = ab * length / ac
        t = lambda f: str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"  # noqa: E731
        return {"a": t(a_wrong), "b": t(ac), "c": t(ab), "d": t(bd)}, {"b", "c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=300, min_seen=3)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9/7", "b": "1/3", "c": "2/9", "d": "5"})


def test_deterministic():
    S.check_deterministic(TOPIC)
