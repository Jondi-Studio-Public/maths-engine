"""`ucv1-first-order-linear-integrating-factor`: y' + P(x) y = Q(x) by an integrating factor.

The oracle reads the equation out of the question's own text and checks the answer against the equation itself: the
integrating factor must make P the log-derivative of mu, and every solution must satisfy the differential equation and
the initial condition. The exam is re-solved with SymPy's own ODE solver and the limits of its terms.
"""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make
from tests.fy1so_support import SEEDS, check_chain, check_follow_through, check_single, check_tiers_differ, parse_text

TOPIC = "fy1so_ode_integrating_factor"
EXAM = "fy1so_exd_ode_integrating_factor"
x = sympy.Symbol("x", positive=True)
D = sympy.Symbol("D")
Y = sympy.Symbol("Y")
_POINTS = [sympy.Rational(7, 10), sympy.Rational(13, 10), sympy.Rational(19, 10), sympy.Rational(31, 10)]


def _parse_x(text: str):
    """`text` with x read as the positive symbol used here."""
    expr = parse_text(text, {"D": D, "Y": Y})
    return expr.xreplace({sympy.Symbol("x"): x})


def _equation(text: str):
    """(alpha, beta, rhs): alpha y' + beta y = rhs, from the first ``... = ...`` of the ODE in `text`."""
    lhs, rhs = re.search(r"(?:equation|problem) (.+?) = (.+?)(?:\.|,)(?: |$)", text).groups()
    rhs = rhs.split(" for ")[0]
    lhs = lhs.replace("dy/dx", " D ").replace("y", " Y ")
    lhs = re.sub(r"\)\s*Y", ")*Y", lhs)
    expr = sympy.expand(_parse_x(lhs))
    alpha = expr.coeff(D, 1)
    beta = expr.coeff(Y, 1)
    return alpha, beta, _parse_x(rhs)


def _ic(text: str):
    found = re.search(r"y\((\d+)\) = (-?\d+)", text)
    return sympy.Integer(found.group(1)), sympy.Integer(found.group(2))


def test_easy_integrating_factor_oracle():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        _, beta, _ = _equation(q.text)
        mu = _parse_x(q.answer_text)
        # mu' / mu = P, so mu is e^(integral of P) up to a constant factor
        for point in _POINTS:
            lhs = sympy.diff(mu, x).subs(x, point) / mu.subs(x, point)
            assert abs(sympy.N(lhs - beta.subs(x, point), 30)) < 1e-20, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_solution_satisfies_equation_and_condition(difficulty):
    seen = set()
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        alpha, beta, rhs = _equation(q.text)
        y = _parse_x(q.answer_text)
        for point in _POINTS:
            residual = alpha * sympy.diff(y, x) + beta * y - rhs
            assert abs(sympy.N(residual.subs(x, point), 30)) < 1e-20, (seed, q.text, q.answer_text)
        x0, y0 = _ic(q.text)
        assert sympy.simplify(y.subs(x, x0) - y0) == 0, (seed, q.text, q.answer_text)
        seen.add(
            "lead" if alpha != 1 else "resonant" if "e^(-" in q.text.split("=")[1] and difficulty == "hard" else "plain"
        )
    if difficulty == "hard":
        assert {"lead", "resonant"} <= seen


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "expression", seeds=range(24), floor=12)  # the expression grader is slow


def test_every_wrong_option_fails_the_equation():
    """A distractor is wrong as a mathematical statement, not just different from the key."""
    for difficulty in ("medium", "hard"):
        for seed in range(30):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            alpha, beta, rhs = _equation(q.text)
            x0, y0 = _ic(q.text)
            for d in q.distractors:
                y = _parse_x(d)
                residual = abs(sympy.N((alpha * sympy.diff(y, x) + beta * y - rhs).subs(x, sympy.Rational(13, 10)), 20))
                at_start = abs(sympy.N(y.subs(x, x0) - y0, 20))
                assert residual > 1e-9 or at_start > 1e-9, (seed, q.text, d)


def test_tiers_differ_in_shape():
    pools = check_tiers_differ(TOPIC)
    assert all("integrating factor" in t for t in pools["easy"])
    assert all(t.startswith("Solve the initial value problem") for t in pools["medium"] | pools["hard"])


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction", "fraction", "fraction", "number"], [1, 2, 1, 2])
    check_follow_through(EXAM)


def test_exam_oracle():
    for seed in range(8):  # SymPy's ODE solver is slow; the chain's other checks run on all 60 seeds
        q = make(EXAM, seed=seed, difficulty="exam")
        match = re.search(
            r"equation dy/dx \+ (\d+)y = (.+?), y\(0\) = (-?\d+)\.", q.stem.replace("problem ", "equation ")
        )
        a = int(match.group(1))
        forcing = _parse_x(match.group(2))
        y0 = int(match.group(3))
        f = sympy.Function("f")
        solved = sympy.dsolve(sympy.Eq(f(x).diff(x) + a * f(x), forcing), f(x), ics={f(0): y0}).rhs
        big_a = sympy.limit(solved.diff(x), x, sympy.oo)
        big_b = sympy.limit(solved - big_a * x, x, sympy.oo)
        const = y0 - big_b
        x1 = sympy.nsimplify(re.search(r"Find y\((\S+?)\)", q.parts[3].prompt).group(1))
        want = [big_a, big_b, const]
        for part, value in zip(q.parts, want, strict=False):
            ratio = sympy.Rational(part.answer_text) if "/" in part.answer_text else sympy.Integer(part.answer_text)
            assert ratio == value, (seed, part.label, part.answer_text, value)
        assert abs(sympy.N(solved.subs(x, x1), 30) - sympy.Rational(q.parts[3].answer_text)) <= sympy.Rational(1, 2000)
        assert q.parts[2].follow_through.endswith("- part_b")


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-first-order-linear-integrating-factor")
    assert skill.status == "active" and skill.answer == "expression"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert skill.sources[-1].stages == ["exam"]
