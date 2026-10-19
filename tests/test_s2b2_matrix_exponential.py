"""Tests `uc2-matrix-exponential` (topic `s2b2_matrix_exponential`) and its exam chain.

The oracle reads A (and the initial vector) back out of the question TEXT and builds e^(At) from SymPy's Jordan form,
block by block: e^(Jt) has e^(λt) Σ (tN)ᵏ/k! on each block. It shares nothing with the generator's exact-fraction
routines.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2b2_matrix_exponential_build  # noqa: F401  (registers the topics)
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
)

TOPIC = "s2b2_matrix_exponential"
EXAM = "exd_x_s2b2_matrix_exponential"
SKILL = load_skill("uc2-matrix-exponential")
T = sympy.Symbol("t")


def exp_at(a):
    """e^(At) from the Jordan form of A."""
    p, j = a.jordan_form()
    n = j.rows
    e = sympy.zeros(n, n)
    i = 0
    while i < n:
        size = 1
        while i + size < n and j[i + size - 1, i + size] == 1:
            size += 1
        lam = j[i, i]
        for r in range(size):
            for c in range(r, size):
                k = c - r
                e[i + r, i + c] = sympy.exp(lam * T) * T**k / sympy.factorial(k)
        i += size
    return (p * e * p.inv()).applyfunc(sympy.simplify)


def key_of(q):
    return sympy.sympify(q.answer.replace("^", "**"), locals={"t": T})


def same(a, b):
    return sympy.simplify(sympy.expand(a - b)) == 0


def test_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression", ["ucv1-diagonalisation", "ucv1-eigenvectors"])


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_entry_or_component_of_the_exponential(tier):
    jordan, diagonal, ivp = 0, 0, 0
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=tier)
        a = matrices(q.text)[0]
        e = exp_at(a)
        entry = re.search(r"\((\d), (\d)\) entry", q.text)
        if entry:
            i, j = int(entry.group(1)) - 1, int(entry.group(2)) - 1
            expected = e[i, j]
        else:
            ivp += 1
            x0 = [int(v) for v in re.search(r"x\(0\) = \(([^)]*)\)ᵀ", q.text).group(1).split(",")]
            row = int(re.search(r"x(\d)\(t\)", q.text).group(1)) - 1
            expected = sum(e[row, k] * x0[k] for k in range(len(x0)))
        assert same(key_of(q), expected), (seed, q.text, q.answer, expected)
        if a.is_diagonalizable():
            diagonal += 1
        else:
            jordan += 1
    if tier == "easy":
        assert diagonal == 20
    if tier == "medium":
        assert jordan == 20
    if tier == "hard":
        assert jordan >= 5 and ivp >= 5


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier, seeds=15)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier, seeds=40) >= 30


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("number",) * 4, (1, 2, 2, 3), 1, seeds=40)
    check_follow_through_strings(EXAM, seeds=40)
    check_follow_through_marks(SKILL, EXAM, seeds=40)


def t_coefficient(expr, lam):
    """The coefficient of t in expr / e^(lam t), which has the form p + q t."""
    return sympy.expand(sympy.simplify(expr * sympy.exp(-lam * T))).coeff(T, 1)


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrices(q.stem)[0]
        e = exp_at(a)
        lam = list(a.eigenvals())
        assert len(lam) == 1 and int(q.parts[0].answer) == lam[0]
        row = int(re.search(r"\((\d), 1\)", q.parts[1].prompt).group(1)) - 1
        assert int(q.parts[1].answer) == t_coefficient(e[row, 0], lam[0])
        assert int(q.parts[2].answer) == t_coefficient(e[row, 1], lam[0])
        x0 = [int(v) for v in re.search(r"x\(0\) = \(([^)]*)\)ᵀ", q.parts[3].prompt).group(1).split(",")]
        xs = sympy.Matrix([sum(e[r, k] * x0[k] for k in range(2)) for r in range(2)])
        assert int(q.parts[3].answer) == t_coefficient(xs[row], lam[0])
        # x' = Ax for the stated solution.
        assert (xs.diff(T) - a * xs).applyfunc(sympy.simplify) == sympy.zeros(2, 1)
