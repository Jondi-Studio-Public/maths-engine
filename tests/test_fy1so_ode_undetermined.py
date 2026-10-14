"""`ucv1-undetermined-coefficients`: particular solutions of y'' + b y' + c y = g(x), and the initial value problem.

The oracle reads the equation out of the question's own text and puts every answer back into it with exact SymPy
derivatives: a particular solution must satisfy y'' + b y' + c y = g exactly, and each wrong option must fail. The exam's
constants are re-derived by solving the initial conditions as a linear system, not by the generator's formulas.
"""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make
from tests.fy1so_support import SEEDS, check_chain, check_follow_through, check_single, check_tiers_differ, parse_text

TOPIC = "fy1so_ode_undetermined"
EXAM = "fy1so_exd_ode_undetermined"
x = sympy.Symbol("x")
D2, D1, Y = sympy.symbols("D2 D1 Y")
_POINTS = [sympy.Rational(3, 10), sympy.Rational(7, 10), sympy.Rational(11, 10)]


def _equation(text: str):
    """(b, c, g) from the ``y'' + b y' + c y = g`` that follows ``particular solution of`` or ``initial value problem``."""
    found = re.search(
        r"(?:particular solution of|initial value problem) (y''.*?) = (.+?)(?: of the form|\.$|\. |, y\(0\))", text
    )
    lhs, rhs = found.groups()
    lhs = lhs.replace("y''", " D2 ").replace("y'", " D1 ")
    lhs = lhs.replace("y", " Y ")  # the derivative symbols hold no y
    expr = sympy.expand(parse_text(lhs, {"D2": D2, "D1": D1, "Y": Y}))
    assert expr.coeff(D2) == 1
    return expr.coeff(D1), expr.coeff(Y), parse_text(rhs)


def _rate(coefficient: str) -> int:
    """The k of ``e^(kx)`` as printed: ``x`` is 1, ``-x`` is -1."""
    return {"": 1, "-": -1}.get(coefficient, None) or int(coefficient)


def _residual(answer, b, c, g, point):
    value = sympy.diff(answer, x, 2) + b * sympy.diff(answer, x) + c * answer - g
    return abs(sympy.N(value.subs(x, point), 30))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_the_answer_is_a_particular_solution(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        b, c, g = _equation(q.text)
        answer = parse_text(q.answer_text)
        for point in _POINTS:
            assert _residual(answer, b, c, g, point) < 1e-20, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_every_wrong_option_fails_the_equation(difficulty):
    for seed in range(30):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        b, c, g = _equation(q.text)
        for d in q.distractors:
            wrong = parse_text(d)
            assert any(_residual(wrong, b, c, g, point) > 1e-9 for point in _POINTS), (seed, q.text, d)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "expression", seeds=range(24), floor=12)  # the expression grader is slow


def test_tiers_use_different_forcing_terms():
    pools = check_tiers_differ(TOPIC)
    assert all("of the form" in t for t in pools["easy"])
    assert all("of the form" not in t for t in pools["medium"] | pools["hard"])
    assert any("x²" in t for t in pools["medium"]) and any("cos" in t and "sin" in t for t in pools["medium"])
    # the hard forcing term always matches a homogeneous solution: a stated root, a repeated root or y'' + w²y
    assert all(("root" in t) or re.search(r"y'' \+ \d+y = ", t) for t in pools["hard"])


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction", "fraction", "fraction", "number"], [1, 2, 1, 2])
    check_follow_through(EXAM)


def test_exam_oracle():
    for seed in SEEDS:
        q = make(EXAM, seed=seed, difficulty="exam")
        b, c, g = _equation(q.stem)
        y0, v0 = (int(v) for v in re.search(r"y\(0\) = (-?\d+), y'\(0\) = (-?\d+)", q.stem).groups())
        r1, r2 = (_rate(v) for v in re.search(r"y = Be\^\((-?\d*)x\) \+ Ce\^\((-?\d*)x\)", q.stem).groups())
        # the roots really are the characteristic roots of the stated equation
        assert (
            sympy.expand(
                (sympy.Symbol("r") - r1) * (sympy.Symbol("r") - r2)
                - (sympy.Symbol("r") ** 2 + b * sympy.Symbol("r") + c)
            )
            == 0
        )
        k = _rate(re.search(r"y = Ae\^\(?(-?\d*)x\)?", q.stem).group(1))
        a_sym, b_sym, c_sym = sympy.symbols("A B C")
        full = a_sym * sympy.exp(k * x) + b_sym * sympy.exp(r1 * x) + c_sym * sympy.exp(r2 * x)
        # A from the equation, B and C from the initial conditions
        a_value = sympy.solve(
            sympy.simplify((sympy.diff(full, x, 2) + b * sympy.diff(full, x) + c * full - g) / sympy.exp(k * x)).subs(
                {b_sym: 0, c_sym: 0}
            ),
            a_sym,
        )[0]
        solution = sympy.solve([full.subs(x, 0) - y0, sympy.diff(full, x).subs(x, 0) - v0], [b_sym, c_sym], dict=True)[
            0
        ]
        solution = {key: value.subs(a_sym, a_value) for key, value in solution.items()}
        want = [a_value, solution[b_sym], solution[c_sym]]
        for part, value in zip(q.parts, want, strict=False):
            assert sympy.Rational(part.answer_text) == value, (seed, part.label, part.answer_text, value)
        x1 = sympy.nsimplify(re.search(r"Find y\((\S+?)\)", q.parts[3].prompt).group(1))
        total = full.subs({a_sym: want[0], b_sym: want[1], c_sym: want[2]}).subs(x, x1)
        assert abs(sympy.N(total, 30) - sympy.Rational(q.parts[3].answer_text)) <= sympy.Rational(1, 2000)


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-undetermined-coefficients")
    assert skill.status == "active" and skill.answer == "expression"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert skill.sources[-1].stages == ["exam"]
    assert skill.prerequisites == ["ucv1-second-order-characteristic-roots"]
