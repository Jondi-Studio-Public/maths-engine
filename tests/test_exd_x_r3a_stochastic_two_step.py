"""Exam chain of `ucv1-stochastic-euler-step`: Y1, Y2 with re-evaluated coefficients, then the effect of the noise."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_stochastic_two_step"
SKILL = S.load_skill("ucv1-stochastic-euler-step")
Y = sympy.Symbol("Y")


def _show(f) -> str:
    f = sympy.Rational(f)
    return str(f.p) if f.q == 1 else f"{f.p}/{f.q}"


def _read(q):
    m = re.search(
        r"dY = \((.*?)\) dt \+ σ\(Y\) dW, where σ\(Y\) = (.*?) and Y\(0\) = (-?\d+)\. .*? h = (\S+) are taken.*?"
        r"ΔW1 = (\S+) for the first step and ΔW2 = (\S+) for the second",
        q.stem,
    )
    drift_text, sigma_text, y0, h, dw1, dw2 = m.groups()
    fix = lambda s: sympy.sympify(re.sub(r"(\d)Y", r"\1*Y", s))  # noqa: E731
    return (
        fix(drift_text),
        fix(sigma_text),
        sympy.Rational(y0),
        sympy.Rational(h),
        sympy.Rational(dw1),
        sympy.Rational(dw2),
    )


def _steps(drift, sigma, y0, h, dws):
    value = y0
    out = []
    for dw in dws:
        value = value + drift.subs(Y, value) * h + sigma.subs(Y, value) * dw
        out.append(value)
    return out


def oracle(q):
    drift, sigma, y0, h, dw1, dw2 = _read(q)
    y1, y2 = _steps(drift, sigma, y0, h, [dw1, dw2])
    _, d2 = _steps(drift, sigma, y0, h, [0, 0])
    return {"a": _show(y1), "b": _show(y2), "c": _show(y2 - d2)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
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


def test_follow_through_from_wrong_first_step(questions):
    def chain(q):
        drift, sigma, y0, h, dw1, dw2 = _read(q)
        wrong = sympy.Rational(q.parts[0].distractors[0])
        y2 = wrong + drift.subs(Y, wrong) * h + sigma.subs(Y, wrong) * dw2
        _, d2 = _steps(drift, sigma, y0, h, [0, 0])
        diff = y2 - d2
        if len({wrong, y2, diff}) < 3 or 0 in (wrong, y2, diff):
            return None
        return {"a": _show(wrong), "b": _show(y2), "c": _show(diff)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=200, min_seen=10)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999/7", "b": "8888/3", "c": "7777/9"})


def test_deterministic():
    S.check_deterministic(TOPIC)
