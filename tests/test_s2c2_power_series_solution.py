"""Tests `uc2-power-series-solution`: the topic `s2c2_power_series_solution` (easy, medium, hard) and the exam chain
`exd_x_s2c2_power_series_solution`.

The oracle reads the equation, the expansion point and the initial values back out of the question text and finds the
Taylor coefficients by repeated differentiation of y'' = -(Q y' + R y) / P at the expansion point. That route shares
nothing with the generator's coefficient recurrence.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2c2_power_series_solution_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects
from tests.s2c2_oracle import half_up

SKILL_ID = "uc2-power-series-solution"
TOPIC = "s2c2_power_series_solution"
EXAM = "exd_x_s2c2_power_series_solution"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
x = sympy.Symbol("x")
Y = sympy.Function("y")
_T = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _read(text):
    text = text.replace("²", "**2").replace("^", "**")
    text = re.sub(r"(?<![A-Za-z])x\(", "x*(", text)
    return parse_expr(
        text,
        transformations=_T,
        local_dict={"x": x, "D2": sympy.Symbol("D2"), "D1": sympy.Symbol("D1"), "D0": sympy.Symbol("D0")},
    )


def _problem(text):
    """(P, Q, R, x0, y(x0), y'(x0)) from the stem."""
    match = re.search(r"problem (.+?) = 0, y\((-?\d+)\) = (-?\d+), y'\((-?\d+)\) = (-?\d+)", text)
    assert match, text
    ode, x0, a0, x0b, a1 = match.groups()
    assert x0 == x0b
    ode = ode.replace("y''", "D2").replace("y'", "D1")
    ode = re.sub(r"(?<![A-Za-z])y(?![A-Za-z(])", "D0", ode)
    expr = sympy.expand(_read(ode))
    d2, d1, d0 = sympy.symbols("D2 D1 D0")
    return expr.coeff(d2), expr.coeff(d1), expr.coeff(d0), int(x0), int(a0), int(a1)


def _taylor(p, q, r, x0, a0, a1, n):
    """The coefficient of (x - x0)^n by repeated differentiation of y'' = -(q y' + r y)/p."""
    y0, y1 = sympy.symbols("y0 y1")
    # represent d^k y/dx^k as c0(x) y + c1(x) y' with coefficient functions of x
    c = [(sympy.Integer(1), sympy.Integer(0)), (sympy.Integer(0), sympy.Integer(1))]
    for _ in range(2, n + 1):
        u, v = c[-1]
        # d/dx (u y + v y') = u' y + u y' + v' y' + v y'', and y'' = -(q y' + r y)/p
        y2 = (-r / p, -q / p)
        new_u = sympy.diff(u, x) + v * y2[0]
        new_v = u + sympy.diff(v, x) + v * y2[1]
        c.append((sympy.simplify(new_u), sympy.simplify(new_v)))
    u, v = c[n]
    value = (u * a0 + v * a1).subs(x, x0)
    return sympy.nsimplify(value / sympy.factorial(n))


def _asked(text):
    return int(re.search(r"coefficient a_(\d)", text).group(1))


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM, "exam", ["exam"])
    assert [src.difficulty for src in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_taylor_coefficient(tier):
    for seed in range(40):
        q = _q(tier, seed)
        p, qq, r, x0, a0, a1 = _problem(q.text)
        n = _asked(q.text)
        assert sympy.Rational(q.answer_text) == _taylor(p, qq, r, x0, a0, a1, n), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_wrong_answers(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert set(q.option_notes) == set(q.distractors)
        assert all(note in CATALOGUE for note in q.option_notes.values())
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)


def test_the_tiers_differ_in_shape():
    for seed in range(40):
        easy = _problem(_q("easy", seed).text)
        medium = _problem(_q("medium", seed).text)
        hard = _problem(_q("hard", seed).text)
        assert easy[0] == 1 and easy[3] == 0
        assert sympy.degree(medium[0], x) == 2 and medium[3] == 0
        assert hard[0] == 1 and hard[3] != 0
        assert "(x " in _q("hard", seed).text  # the series is in powers of (x - x0)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    texts = [_q(tier, seed).text for seed in range(80)]
    assert len(set(texts)) >= 50
    answers = [_q(tier, seed).answer_text for seed in range(80)]
    assert len(set(answers)) >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _exam_problem(q):
    return _problem(q.stem.replace("Consider the initial value problem", "problem").replace("«", ""))


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "fraction", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_keys_match_independent_derivation():
    for seed in range(30):
        q = _exam(seed)
        p, qq, r, x0, a0, a1 = _problem(re.sub(r"^Consider the initial value problem ", "problem ", q.stem))
        a, b, c, d = q.parts
        for part, n in ((a, 2), (b, 3), (c, 4)):
            assert sympy.Rational(part.answer_text) == _taylor(p, qq, r, x0, a0, a1, n), (seed, part.label)
        at = sympy.Rational(d.prompt.split("y(")[1].split(")")[0])
        total = sum(_taylor(p, qq, r, x0, a0, a1, n) * at**n for n in range(5))
        assert half_up(total) == d.answer_text, (seed, d.answer_text, total)


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(25):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_into_a_3():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = _canonical(q)
        p, qq, r, x0, a0, a1 = _problem(re.sub(r"^Consider the initial value problem ", "problem ", q.stem))
        q0, q1 = qq.subs(x, 0), sympy.diff(qq, x)
        r0, r1 = r.subs(x, 0), sympy.diff(r, x)
        for wrong in q.parts[0].distractors:
            a2 = sympy.Rational(wrong)
            if a2 == 0:
                continue
            reply_b = -(2 * q0 * a2 + (q1 + r0) * a1 + r1 * a0) / 6
            if reply_b == 0 or str(reply_b) in (canon["b"], wrong):
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(reply_b)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 20


def test_follow_through_into_the_sum():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = _canonical(q)
        p, qq, r, x0, a0, a1 = _problem(re.sub(r"^Consider the initial value problem ", "problem ", q.stem))
        at = sympy.Rational(q.parts[3].prompt.split("y(")[1].split(")")[0])
        for wrong in q.parts[2].distractors:
            a4 = sympy.Rational(wrong)
            total = a0 + a1 * at + sympy.Rational(canon["a"]) * at**2 + sympy.Rational(canon["b"]) * at**3 + a4 * at**4
            if abs(total) < 0.01 or half_up(total) == canon["d"] or a4 == 0 or total == a4:
                continue
            result = grade(_SKILL, q, {**canon, "c": wrong, "d": half_up(total)})
            assert result.parts["c"].marks == 0
            assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 20


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(100)]
    assert len(set(answers)) >= 60
    assert _exam(7).to_json() == _exam(7).to_json()
