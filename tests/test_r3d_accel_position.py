"""Tests `acceleration-function-of-position`: oracles solve v dv/dx = f(x) from the question text alone."""

from __future__ import annotations

import math
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_follow_through_credit,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    assert_states,
    frac,
    questions,
    skill,
    sources,
)

from mathgen import make
from mathgen.engine import grade

SK = skill("acceleration-function-of-position")
SEEDS = range(60)
x = sympy.Symbol("x")
_IMPLICIT = standard_transformations + (implicit_multiplication_application,)


def f(text: str) -> float:
    return float(frac(text))


def linear(text: str):
    """'(30 - 6x)' -> (30, 6) for a = p - q x; '(-2 + 2x)' -> (-2, -2); '(-x)' -> (0, 1)."""
    expr = parse_expr(
        re.search(r"acceleration is \((.+?)\) m/s²", text).group(1), local_dict={"x": x}, transformations=_IMPLICIT
    )
    return float(expr.subs(x, 0)), float(-sympy.diff(expr, x))


def v_squared(a_of_x, v0: float, x0: float):
    """v² = v0² + 2 ∫ a dx from x0, as a SymPy expression in x."""
    return v0**2 + 2 * sympy.integrate(a_of_x, (x, x0, x))


def real_roots(expr):
    return sorted(float(r) for r in sympy.Poly(sympy.together(expr), x).nroots() if abs(sympy.im(r)) < 1e-9)


def oracle_easy(q):
    s = q.text
    p, k = linear(s)
    x0, v0 = (f(g) for g in re.search(r"passes x = (-?[\d.]+) with speed ([\d.]+) m/s", s).groups())
    x1 = f(re.search(r"speed when x = (-?[\d.]+)", s).group(1))
    return math.sqrt(float(v_squared(p - k * x, v0, x0).subs(x, x1)))


def test_easy_oracle_and_grading():
    assert_single_answer_rules(SK, "easy", SEEDS)
    for q in questions(SK, "easy", range(100)):
        assert_number_distractors_match_places(q)
        assert_states(q, oracle_easy(q), q.text)


def oracle_medium(q):
    s = q.text
    x1 = f(re.search(r"speed when x = (-?[\d.]+)", s).group(1))
    if m := re.search(r"magnitude (\d+)/x²", s):
        big_k = f(m.group(1))
        if "released from rest at" in s:
            x0, v0 = f(re.search(r"released from rest at x = ([\d.]+)", s).group(1)), 0.0
        else:
            x0, v0 = (f(g) for g in re.search(r"passes x = ([\d.]+) with speed ([\d.]+) m/s", s).groups())
        return math.sqrt(float(v_squared(-big_k / x**2, v0, x0).subs(x, x1))), "inverse-square"
    m = re.search(r"acceleration is (-?\d*)x² m/s²\. The particle passes x = (-?[\d.]+) with speed ([\d.]+) m/s", s)
    coefficient = {"": 1, "-": -1}.get(m.group(1)) or f(m.group(1))
    return math.sqrt(float(v_squared(coefficient * x**2, f(m.group(3)), f(m.group(2))).subs(x, x1))), "quadratic"


def test_medium_oracle_and_grading():
    assert_single_answer_rules(SK, "medium", SEEDS)
    kinds = set()
    for q in questions(SK, "medium", range(150)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_medium(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"inverse-square", "quadratic"}


def oracle_hard(q):
    s = q.text
    if m := re.search(
        r"magnitude (\d+)/x² m/s² when it is x metres from O\. It passes x = ([\d.]+) with speed ([\d.]+) m/s, moving away from O",
        s,
    ):
        big_k, x0, v0 = (f(g) for g in m.groups())
        return 1 / (1 / x0 - v0**2 / (2 * big_k)), "turning-point"
    p, k = linear(s)
    x0, v0 = (f(g) for g in re.search(r"passes x = (-?[\d.]+) with speed ([\d.]+) m/s", s).groups())
    v2 = v_squared(p - k * x, v0, x0)
    if "greatest speed" in s:
        return math.sqrt(float(v2.subs(x, p / k))), "greatest-speed"
    roots = real_roots(v2)
    if "negative direction" in s:
        return max(r for r in roots if r < x0 - 1e-9), "first-rest"
    return min(r for r in roots if r > x0 + 1e-9), "first-rest"


def test_hard_oracle_and_grading():
    assert_single_answer_rules(SK, "hard", SEEDS)
    kinds = set()
    for q in questions(SK, "hard", range(150)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_hard(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"turning-point", "greatest-speed", "first-rest"}


def test_first_rest_depends_on_direction():
    """Moving in the negative direction the particle meets the nearer root below its start; the question says so."""
    seen = 0
    for q in questions(SK, "hard", range(150)):
        if "first comes instantaneously to rest" in q.text:
            assert "negative direction" in q.text or "positive direction" in q.text, q.text
            seen += 1
    assert seen >= 5


def exam_oracle(q):
    p, k = linear(q.text)
    v0 = f(re.search(r"(?:at|with(?: initial)? speed) ([\d.]+) m/s", q.text).group(1))
    v2 = v0**2 + 2 * p * x - k * x**2
    x_b = f(re.search(r"when x = ([\d.]+)", q.parts[1].prompt).group(1))
    far = max(real_roots(v2))
    return v2, math.sqrt(float(v2.subs(x, x_b))), far, math.sqrt(float(v2.subs(x, p / k)))


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[2, 2, 2, 2])
    topic = sources(SK)["exam"]
    for seed in range(100):
        q = make(topic, seed=seed, difficulty="exam")
        v2, speed, far, top = exam_oracle(q)
        # (a) the stated expression equals v² for every x
        stated = sympy.sympify(q.parts[0].answer.replace("^", "**"), locals={"x": x})
        assert sympy.simplify(stated - v2) == 0, (q.text, q.parts[0].answer)
        assert grade(
            SK,
            q,
            {
                "a": q.parts[0].answer_text,
                "b": q.parts[1].answer_text,
                "c": q.parts[2].answer_text,
                "d": q.parts[3].answer_text,
            },
        ).correct
        for part, expected in zip(q.parts[1:], (speed, far, top), strict=True):
            assert_states(part, expected, q.text)


def test_exam_expression_accepts_an_equivalent_form():
    q = make(sources(SK)["exam"], seed=3, difficulty="exam")
    p, k = linear(q.text)
    v0 = f(re.search(r"(?:at|with(?: initial)? speed) ([\d.]+) m/s", q.text).group(1))
    v2 = sympy.expand(v0**2 + 2 * p * x - k * x**2)
    canon = {pt.label: pt.answer_text for pt in q.parts}
    reordered = str(sympy.Add(*sorted(v2.as_ordered_terms(), key=str))).replace("**", "^")
    assert grade(SK, q, {**canon, "a": reordered}).parts["a"].marks == q.parts[0].marks
    wrong = str(sympy.expand(v0**2 + p * x - k * x**2 / 2)).replace("**", "^")
    assert grade(SK, q, {**canon, "a": wrong}).parts["a"].marks == 0


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_exam_distractors_keep_the_precision():
    for seed in range(40):
        for pt in make(sources(SK)["exam"], seed=seed, difficulty="exam").parts:
            assert_number_distractors_match_places(pt)


@pytest.mark.parametrize(
    "topic,difficulty,distinct",
    [
        ("r3d_accel_position", "easy", 20),
        ("r3d_accel_position", "medium", 20),
        ("r3d_accel_position", "hard", 20),
        ("r3d_accel_position_exam", "exam", 10),
    ],
)
def test_deterministic_and_varied(topic, difficulty, distinct):
    assert_deterministic_and_varied(topic, difficulty, distinct=distinct)


def test_tiers_differ_in_shape():
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    medium = " ".join(q.text for q in questions(SK, "medium", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "x²" not in easy and "/x²" not in easy
    assert "/x²" in medium and "greatest" not in medium
    assert "greatest" in hard and "rest" in hard
