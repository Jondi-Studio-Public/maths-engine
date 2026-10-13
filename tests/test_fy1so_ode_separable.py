"""`ucv1-ode-separable-product`: separable ODEs dy/dx = g(x) h(y) with an initial condition.

The oracle reads the equation and the initial condition out of the question's own text and checks every answer against
them: a solution must satisfy the differential equation and the initial condition, and each wrong option must fail one of
the two. The exam's four values are re-derived from the solution curve itself (where it blows up or reaches zero, and
where it takes a stated value), never from the generator's formulas.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from tests.fy1so_support import SEEDS, check_chain, check_follow_through, check_single, check_tiers_differ, parse_text

TOPIC = "fy1so_ode_separable"
EXAM = "fy1so_exd_ode_separable"
x = sympy.Symbol("x")
y = sympy.Symbol("y")
_POINTS = [sympy.Rational(1, 20), sympy.Rational(1, 10), sympy.Rational(3, 20), sympy.Rational(1, 5)]


def _rate_and_condition(text: str):
    """(f(x, y), x0, y0) for ``dy/dx = f(x, y)`` or ``x dy/dx = g(y)`` and ``y(x0) = y0`` in `text`."""
    found = re.search(r"problem (x )?dy/dx = (.+?),(?: x > 0,)? y\((\d)\) = (-?\d+)", text)
    lead, rhs, x0, y0 = found.groups()
    rate = parse_text(rhs.replace("e^(-y)", "E**(-y)"), {"y": y})
    if lead:
        rate = rate / x
    return rate, sympy.Integer(x0), sympy.Integer(y0)


def _satisfies(answer, rate, x0, y0, tolerance=1e-12) -> bool:
    for point in _POINTS if x0 == 0 else [sympy.Rational(3, 2), sympy.Rational(5, 2)]:
        slope = sympy.diff(answer, x).subs(x, point)
        want = rate.subs({x: point, y: answer.subs(x, point)})
        if abs(sympy.N(slope - want, 30)) > tolerance:
            return False
    return abs(sympy.N(answer.subs(x, x0) - y0, 30)) <= tolerance


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_the_answer_solves_the_initial_value_problem(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        rate, x0, y0 = _rate_and_condition(q.text)
        answer = parse_text(q.answer_text)
        assert _satisfies(answer, rate, x0, y0), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_every_wrong_option_fails_the_problem(difficulty):
    for seed in range(30):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        rate, x0, y0 = _rate_and_condition(q.text)
        for d in q.distractors:
            assert not _satisfies(parse_text(d), rate, x0, y0, tolerance=1e-9), (seed, q.text, d)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "expression", seeds=range(24), floor=12)  # the expression grader is slow


def test_tiers_use_different_equations():
    pools = check_tiers_differ(TOPIC)
    assert all("y > 0" in t or "y(1)" in t for t in pools["easy"])
    assert any("y²" in t for t in pools["medium"]) and any("e^(-y)" in t for t in pools["medium"])
    assert any("5 - y" in t or "- y)" in t for t in pools["hard"]) and any("(1 + y²)" in t for t in pools["hard"])


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction", "fraction", "number", "number"], [1, 2, 2, 2])
    check_follow_through(EXAM)


def _fraction(text: str) -> sympy.Rational:
    return sympy.Rational(Fraction(text))


def test_exam_oracle():
    blow_up = stops = 0
    for seed in SEEDS:
        q = make(EXAM, seed=seed, difficulty="exam")
        a_part, b_part, c_part, d_part = q.parts
        found = re.search(r"dy/dx = (-?)\(?(\d*)x\)?(?:y²|/y), y\(0\) = (\S+?)[.,]", q.stem)
        sign, coefficient, y0_text = found.groups()
        k = sympy.Integer(coefficient or 1)
        y0 = _fraction(y0_text)
        x1 = _fraction(re.search(r"find y\((\S+?)\)", b_part.prompt).group(1))
        constant = _fraction(a_part.answer_text)
        if sign:  # dy/dx = -k x / y: y^2 = 2C - k x^2
            stops += 1
            curve = sympy.sqrt(2 * constant - k * x**2)
            rate = -k * x / y
            target = _fraction(re.search(r"y = (\d+)", d_part.prompt).group(1))
            end = sympy.solve(sympy.Eq(curve, 0), x)
        else:  # dy/dx = k x y^2: -1/y = (k/2) x^2 + C
            blow_up += 1
            curve = -1 / (k * x**2 / 2 + constant)
            rate = k * x * y**2
            target = _fraction(re.search(r"reaches (\S+?),", d_part.prompt).group(1))
            end = sympy.solve(sympy.Eq(k * x**2 / 2 + constant, 0), x)
        # the curve solves the equation and passes through (0, y0), so the constant is the right one
        assert abs(sympy.N(curve.subs(x, 0) - y0, 30)) < 1e-20, (seed, q.stem)
        for point in _POINTS:
            slope = sympy.diff(curve, x).subs(x, point)
            assert abs(sympy.N(slope - rate.subs({x: point, y: curve.subs(x, point)}), 30)) < 1e-20, (seed, q.stem)
        assert sympy.nsimplify(curve.subs(x, x1)) == _fraction(b_part.answer_text), (seed, q.stem, b_part.answer_text)
        positive_end = max(float(e) for e in end)
        assert abs(positive_end - float(c_part.answer_text)) <= 6e-4, (seed, c_part.answer_text, positive_end)
        reach = [float(r) for r in sympy.solve(sympy.Eq(curve, target), x) if r.is_real and r > 0]
        assert reach and abs(min(reach) - float(d_part.answer_text)) <= 6e-4, (seed, d_part.answer_text, reach)
        assert x1 < positive_end
    assert blow_up >= 15 and stops >= 15


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-ode-separable-product")
    assert skill.status == "active" and skill.answer == "expression"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert skill.sources[-1].stages == ["exam"]
