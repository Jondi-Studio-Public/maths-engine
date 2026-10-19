"""Tests `uc2-recurrence-by-diagonalisation` (topic `s2b2_recurrence_diagonalisation`) and its exam chain.

The oracle reads the recurrence (or the stochastic matrix A and x₀) back out of the question TEXT. A closed form is
checked by iterating the recurrence with exact rationals for n = 0..9; a long-run count is checked against the
eigenvalue-1 eigenvector of A scaled to the total of x₀.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2b2_recurrence_diagonalisation_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests.s2b2_support import (
    TIERS,
    check_exam_shape,
    check_follow_through_marks,
    check_follow_through_strings,
    check_single_tier,
    check_wiring,
    distinct_questions,
    load_skill,
    matrices,
    parse,
    value,
)

TOPIC = "s2b2_recurrence_diagonalisation"
EXAM = "exd_x_s2b2_recurrence_diagonalisation"
SKILL = load_skill("uc2-recurrence-by-diagonalisation")
N = sympy.Symbol("n")


def _vector(text, symbol="x₀"):
    return [sympy.Integer(int(v)) for v in re.search(symbol + r" = \(([^)]*)\)", text).group(1).split(",")]


def limit_vector(a, x0):
    """The eigenvalue-1 eigenvector of the column-stochastic A, scaled to the total of x0."""
    v = (a - sympy.eye(a.rows)).nullspace()[0]
    return v * sum(x0) / sum(v)


def parse_system(text):
    """(A, x0, names) from `xₙ₊₁ = 5xₙ + 2yₙ, yₙ₊₁ = ..., with x₀ = 3, y₀ = 1`."""
    s = text.replace("ₙ₊₁", "@1").replace("ₙ", "@0")
    names = sorted(set(re.findall(r"\b([xyz])@1 =", s)))
    syms = {n.upper(): sympy.Symbol(n.upper()) for n in names}
    rows = []
    for n in names:
        m = re.search(rf"{n}@1 = (.+?)(?:, (?=[xyz]@1 =)|,? with)", s)
        rhs = m.group(1)
        for other in names:
            rhs = rhs.replace(f"{other}@0", other.upper())
        expr = parse(rhs, **syms)
        rows.append([expr.coeff(syms[o.upper()]) for o in names])
    x0 = [sympy.Integer(int(re.search(rf"{n}₀ = (-?\d+)", text).group(1))) for n in names]
    return sympy.Matrix(rows), x0, names


def iterate(a, x0, steps):
    out, x = [], sympy.Matrix(x0)
    for _ in range(steps):
        out.append(x)
        x = a * x
    return out


def check_closed_form(answer, a, x0, index):
    expr = parse(str(answer), n=N)
    for n, x in enumerate(iterate(a, x0, 10)):
        assert sympy.simplify(expr.subs(N, n) - x[index]) == 0, (answer, n)


def test_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression", ["ucv1-diagonalisation", "ucv1-eigenvectors"])


def test_easy_key_is_the_long_run_count_in_the_named_state():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        a = matrices(q.text)[0]
        assert a.shape == (2, 2) and all(sum(a[:, j]) == 1 for j in range(2))
        x0 = _vector(q.text)
        first, second = re.search(r"customers of \((.+?), (.+?)\)", q.text).groups()
        named = re.search(r"number of (.+?) customers", q.text).group(1)
        index = [first, second].index(named)
        assert value(q.answer) == limit_vector(a, x0)[index], (seed, q.text)
        # And the iteration really heads there (A^200 x0 exactly).
        far = (a**200) * sympy.Matrix(x0)
        assert abs(far[index] - value(q.answer)) < sympy.Rational(1, 10**6)


def test_medium_key_is_the_closed_form():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        a, x0, names = parse_system(q.text)
        index = names.index(re.search(r"closed form for ([xyz])ₙ", q.text).group(1))
        check_closed_form(q.answer, a, x0, index)
        assert a.shape == (2, 2)


def test_hard_key_is_a_closed_form_or_a_long_run_count():
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "closed form" in q.text:
            seen.add("closed")
            a, x0, names = parse_system(q.text)
            assert a.shape == (3, 3)
            index = names.index(re.search(r"closed form for ([xyz])ₙ", q.text).group(1))
            check_closed_form(q.answer, a, x0, index)
            assert len(set(a.eigenvals())) == 3
        else:
            seen.add("limit")
            a = matrices(q.text)[0]
            x0 = _vector(q.text)
            assert a.shape == (3, 3) and all(sum(a[:, j]) == 1 for j in range(3))
            index = "PQR".index(re.search(r"station ([PQR])", q.text).group(1))
            assert value(q.answer) == limit_vector(a, x0)[index], (seed, q.text)
            far = (a**128) * sympy.Matrix(x0)
            assert abs(far[index] - value(q.answer)) < sympy.Rational(1, 10**3)
    assert seen == {"closed", "limit"}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier, seeds=40)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier, seeds=40) >= 30


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("number", "fraction", "expression", "number"), (2, 2, 3, 1), 1, seeds=40)
    check_follow_through_strings(EXAM, seeds=40)
    check_follow_through_marks(SKILL, EXAM, seeds=40)


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrices(q.stem)[0]
        x0 = _vector(q.stem)
        assert a.shape == (2, 2) and all(sum(a[:, j]) == 1 for j in range(2))
        assert value(q.parts[0].answer) == a.trace() - 1
        limit = limit_vector(a, x0)
        share = limit[0] / sum(x0)
        assert value(q.parts[1].answer) == share
        check_closed_form(q.parts[2].answer, a, x0, 0)
        assert value(q.parts[3].answer) == limit[0] == sum(x0) * share
        assert q.parts[3].follow_through == f"{sum(x0)}*part_b"
