"""Exam chain for `solve-ode-exponential`: k from two readings, the value, the rate and the time to a target."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import sympy

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "solve-ode-exponential"
TOPIC = "exd_x_r2j_exponential_ode_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _dp(x, places: int) -> str:
    return str(Decimal(str(sympy.N(x, 30))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def read(q):
    stem = q.stem
    if m := re.search(r"Initially y = (\d+), and after (\d+) \w+ y = (\d+)", stem):
        y0, t1, y1 = map(int, m.groups())
    elif m := re.search(r"At t = 0, y = (\d+)\. At t = (\d+), y = (\d+)", stem):
        y0, t1, y1 = map(int, m.groups())
    else:
        m = re.search(r"initial value is y = (\d+), and y = (\d+) when t = (\d+)", stem)
        y0, y1, t1 = map(int, m.groups())
    t2 = int(re.search(r"when t = (\d+), correct to 1", q.parts[1].prompt).group(1))
    target = int(re.search(r"reaches (\d+)", q.parts[3].prompt).group(1))
    return y0, y1, t1, t2, target


_T, _K = sympy.symbols("t k", real=True)
_GENERAL = sympy.dsolve(sympy.Eq(sympy.Function("y")(_T).diff(_T), _K * sympy.Function("y")(_T)))


def oracle(q) -> dict[str, str]:
    """Solve dy/dt = ky with sympy's dsolve, fit the constants to the stated readings, read everything off."""
    y0, y1, t1, t2, target = read(q)
    sol = _GENERAL.rhs
    c1 = next(s for s in sol.free_symbols if s.name == "C1")
    sol = sol.subs(c1, sympy.solve(sympy.Eq(sol.subs(_T, 0), y0), c1)[0])
    k_val = sympy.solve(sympy.Eq(sol.subs(_T, t1), y1), _K)[0]
    sol = sol.subs(_K, k_val)
    y2 = sol.subs(_T, t2)
    rate = sympy.diff(sol, _T).subs(_T, t2)
    time = sympy.solve(sympy.Eq(sol, target), _T)[0]
    return {"k": sympy.sstr(k_val), "b": _dp(y2, 1), "c": _dp(rate, 1), "d": _dp(time, 2)}


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "number", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees(questions):
    for q in questions[:60]:
        o = oracle(q)
        c = kit.canon(q)
        assert (o["b"], o["c"], o["d"]) == (c["b"], c["c"], c["d"]), q.text
        k_given = sympy.sympify(c["a"].replace("ln", "log"))
        assert abs(sympy.N(k_given - sympy.sympify(o["k"]), 30)) < 1e-25, q.text


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_grading():
    kit.check_grading(SKILL)


def test_equivalent_forms_of_k_accepted(questions):
    for q in questions[:20]:
        y0, y1, t1, _t2, _target = read(q)
        for reply in (
            f"ln({y1}/{y0})/{t1}",
            f"(1/{t1})*ln({sympy.Rational(y1, y0)})",
            f"ln({y1}) / {t1} - ln({y0}) / {t1}",
        ):
            assert grade(kit.skill(SKILL), q, {**kit.canon(q), "a": reply}).parts["a"].marks == q.parts[0].marks, reply


def test_follow_through_from_wrong_k(questions):
    checked = 0
    for q in questions[:100]:
        y0, y1, t1, t2, target = read(q)
        k_wrong = sympy.log(sympy.Rational(y1, y0))  # forgot to divide by the time
        if k_wrong == 0:
            continue
        y2 = y0 * sympy.exp(k_wrong * t2)
        reply = {
            "a": f"ln({sympy.Rational(y1, y0)})",
            "b": _dp(y2, 1),
            "c": _dp(k_wrong * sympy.Rational(_dp(y2, 1)), 1),
            "d": _dp(sympy.log(sympy.Rational(target, y0)) / k_wrong, 2),
        }
        if float(reply["b"]) > 1e7:
            continue
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["d"].marks == q.parts[3].marks, (q.text, reply)
        checked += 1
    assert checked >= 50
