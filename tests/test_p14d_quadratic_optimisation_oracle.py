"""Independent oracle for the `p14d_quadratic_optimisation` questions: it re-derives each answer from the
rendered question text.

Parses ONLY `question.text`, `question.answer_text`, `question.distractors`
and `question.mcq_options` (via `mathgen.grading.mcq_options`) -- never anything
from `mathgen.generators.p14d_quadratic_optimisation_build` -- and re-derives
each answer from scratch: the critical point via a fresh `sympy.solve` of the
gradient equations parsed out of the rendered f(x, y), and the classification
via a fresh `sympy.Matrix.eigenvals()` call on a Hessian built from
coefficients parsed out of the rendered f(x, y).
"""

import re
from dataclasses import replace
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen
from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills

CRIT_ID = "ucv1-unconstrained-critical-point"
CONVEX_ID = "ucv1-convexity-hessian-sign"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_X, _Y = sympy.symbols("x y")
_T = standard_transformations + (convert_xor, implicit_multiplication_application)


def _parse_poly(expr_text):
    return parse_expr(expr_text, local_dict={"x": _X, "y": _Y}, transformations=_T)


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


CRIT_SOURCES = _skill_sources(CRIT_ID)
CONVEX_SOURCES = _skill_sources(CONVEX_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
CONVEX_SOURCES["exam"] = replace(CONVEX_SOURCES["exam"], topic="p14d_convexity_hessian_sign", params={})

_F_RE = re.compile(r"f\(x, y\) = (.+?)\. ")
_PAIR_RE = re.compile(r"x = (-?\d+(?:/\d+)?), y = (-?\d+(?:/\d+)?)")


def _extract_f(text):
    """f(x, y)'s right-hand side, whether written as a single expression or
    as two bracketed pieces summed (the exam-tier disguise) -- either way,
    just hand the whole RHS text to the parser, which reads `(A) + (B)`
    exactly like any other sum."""
    m = _F_RE.search(text)
    assert m, text
    return _parse_poly(m.group(1))


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_critical_point(difficulty, seed):
    src = CRIT_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    f_expr = _extract_f(q.text)
    solutions = sympy.solve([sympy.diff(f_expr, _X), sympy.diff(f_expr, _Y)], [_X, _Y], dict=True)
    assert len(solutions) == 1, (q.text, solutions)
    derived_x = Fraction(str(solutions[0][_X]))
    derived_y = Fraction(str(solutions[0][_Y]))

    m = _PAIR_RE.fullmatch(q.answer_text)
    assert m, q.answer_text
    stated_x, stated_y = Fraction(m.group(1)), Fraction(m.group(2))

    assert derived_x == stated_x and derived_y == stated_y, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\n"
        f"derived: ({derived_x}, {derived_y})\nstated: ({stated_x}, {stated_y})"
    )


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_convexity_classification(difficulty, seed):
    src = CONVEX_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    f_expr = _extract_f(q.text)
    fxx = sympy.diff(f_expr, _X, _X)
    fxy = sympy.diff(f_expr, _X, _Y)
    fyy = sympy.diff(f_expr, _Y, _Y)
    hessian = sympy.Matrix([[fxx, fxy], [fxy, fyy]])
    signs = []
    for eigenvalue, multiplicity in hessian.eigenvals().items():
        value = complex(sympy.N(eigenvalue))
        assert abs(value.imag) < 1e-9, (q.text, eigenvalue)
        signs.extend([1 if value.real > 0 else -1] * multiplicity)
    assert len(signs) == 2, (q.text, signs)
    if all(s > 0 for s in signs):
        derived = "convex"
    elif all(s < 0 for s in signs):
        derived = "concave"
    else:
        derived = "indefinite"

    assert derived == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {q.answer_text}"
    )
    options = mcq_options(q)
    assert set(options) == {"convex", "concave", "indefinite"}


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds x 2 skills).")
