"""Exam source of `gradient-descent-quadratic`: gradient, then two steps on a loss given in expanded form."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from tests.exd_x_u5_support import (
    as_fraction,
    canonical_reply,
    check_key_and_distractors,
    check_no_text_defects,
    check_shapes,
    exam_question,
    exam_sources,
    parse_expression,
    rational,
    skill_of,
)

from mathgen.engine import grade

SKILL = "gradient-descent-quadratic"
t = sympy.Symbol("t")


def read(q):
    """(loss, theta0, eta) parsed from the rendered stem alone."""
    stem = q.text.split(" (a)")[0]
    loss_text = re.search(r"L\(θ\) = ([^.,]+)", stem).group(1)
    loss = parse_expression(loss_text, t=t)
    theta0 = int(re.search(r"(?:at|from) θ = (-?\d+)", stem).group(1))
    num, den = re.search(r"η = (\d+)/(\d+)", stem).groups()
    return loss, theta0, Fraction(int(num), int(den))


def oracle(q):
    loss, theta0, eta = read(q)
    grad = sympy.diff(loss, t)

    g0 = rational(grad.subs(t, theta0))
    th1 = theta0 - eta * g0
    g1 = rational(grad.subs(t, sympy.Rational(th1.numerator, th1.denominator)))
    th2 = th1 - eta * g1
    return {"a": g0, "b": th1, "c": th2}


def test_source_wired():
    assert [s.topic for s in exam_sources(SKILL)] == ["exd_x_u5_gradient_descent"]
    assert exam_sources(SKILL)[0].stages == ["exam"]


def test_shapes_and_text():
    check_shapes(SKILL)
    check_no_text_defects(SKILL)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam_question(SKILL, seed)
        expected = oracle(q)
        got = {p.label: as_fraction(p.answer_text) for p in q.parts}
        assert got == expected, (seed, q.text)


def test_grader_accepts_key_rejects_distractors():
    check_key_and_distractors(SKILL)


def test_follow_through_from_wrong_gradient():
    skill = skill_of(SKILL)
    checked = 0
    for seed in range(30):
        q = exam_question(SKILL, seed)
        loss, theta0, eta = read(q)
        grad = sympy.diff(loss, t)
        wrong_g0 = as_fraction(q.parts[0].distractors[0])
        th1 = theta0 - eta * wrong_g0
        th2 = th1 - eta * Fraction(str(grad.subs(t, sympy.Rational(th1.numerator, th1.denominator))))
        if th2 == th1 or th1 == 0:  # a repeated or zero earlier reply carries no follow-through
            continue
        checked += 1
        reply = {"a": q.parts[0].distractors[0], "b": str(th1), "c": str(th2)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (seed, reply)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, reply)
        assert result.marks == 4 and not result.correct
    assert checked >= 20


def test_follow_through_from_wrong_first_step():
    skill = skill_of(SKILL)
    for seed in range(30):
        q = exam_question(SKILL, seed)
        loss, theta0, eta = read(q)
        grad = sympy.diff(loss, t)
        wrong_th1 = as_fraction(q.parts[1].distractors[0])
        if wrong_th1 == 0:
            continue
        th2 = wrong_th1 - eta * Fraction(str(grad.subs(t, sympy.Rational(wrong_th1.numerator, wrong_th1.denominator))))
        reply = {**canonical_reply(q), "b": q.parts[1].distractors[0], "c": str(th2)}
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, reply)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL)
    q = exam_question(SKILL, 3)
    assert grade(skill, q, {"a": "999", "b": "998", "c": "997"}).marks == 0
