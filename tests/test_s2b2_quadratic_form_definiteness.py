"""Tests `uc2-quadratic-form-definiteness` (topic `s2b2_quadratic_form_definiteness`) and its exam chain.

The oracle reads the quadratic form (and, for the hard tier, the substitution) back out of the question TEXT, parses
it with SymPy, and re-derives the matrix from the Hessian, the eigenvalues, and PᵀAP.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2b2_quadratic_form_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests.s2b2_support import (
    TIERS,
    answer_matrix,
    check_exam_shape,
    check_follow_through_marks,
    check_follow_through_strings,
    check_single_tier,
    check_wiring,
    distinct_questions,
    load_skill,
    parse,
)

TOPIC = "s2b2_quadratic_form_definiteness"
EXAM = "exd_x_s2b2_quadratic_form_definiteness"
SKILL = load_skill("uc2-quadratic-form-definiteness")
_POLY_END = (
    r"(?= can be written| is rewritten| is a sum of squares| has a symmetric|\s+as xᵀAx| on its diagonal|\.|:|\s+gives)"
)


def _form(text):
    """(variables, polynomial) of the form `q(vars) = poly` stated in `text`."""
    m = re.search(r"q\(([a-z, ]+)\) = (.+?)" + _POLY_END, text)
    names = [v.strip() for v in m.group(1).split(",")]
    poly = parse(m.group(2).replace("²", "**2"), **{n: sympy.Symbol(n) for n in names})
    return [sympy.Symbol(n) for n in names], poly


def _matrix_of(names, poly):
    return sympy.hessian(poly, names) / 2


def test_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "matrix", ["ucv1-eigenvectors", "ucv1-diagonalisation"])


def test_easy_key_is_the_symmetric_matrix():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        names, poly = _form(q.text)
        key = answer_matrix(q.answer)
        assert key == _matrix_of(names, poly) and key == key.T, (seed, q.text)
        vec = sympy.Matrix(names)
        assert sympy.expand((vec.T * key * vec)[0] - poly) == 0


def test_medium_key_is_the_diagonal_matrix_of_increasing_eigenvalues():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        names, poly = _form(q.text)
        a = _matrix_of(names, poly)
        eigenvalues = sorted(v for v, m in a.eigenvals().items() for _ in range(m))
        key = answer_matrix(q.answer)
        assert key == sympy.diag(*eigenvalues), (seed, q.text)


def test_hard_key_is_the_form_after_the_change_of_variable_or_a_sum_of_squares():
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        names, poly = _form(q.text)
        key = answer_matrix(q.answer)
        assert key == key.T
        if "ubstitut" in q.text or "rewritten" in q.text:
            seen.add("substitution")
            new_names = re.search(r"in (?:the variables )?((?:[uvw], )*[uvw])\b", q.text).group(1).split(", ")
            ys = [sympy.Symbol(n) for n in new_names]
            subst = {}
            for old, rhs in re.findall(r"\b([xyz]) = ([^,]*?)(?:, |\.| into| gives|$)", q.text):
                subst[sympy.Symbol(old)] = parse(rhs, **{n: sympy.Symbol(n) for n in new_names})
            assert len(subst) == len(names)
            expanded = sympy.expand(poly.subs(subst, simultaneous=True))
            assert key == sympy.hessian(expanded, ys) / 2, (seed, q.text)
        else:
            seen.add("squares")
            assert key == _matrix_of(names, poly), (seed, q.text)
            assert key.is_positive_semidefinite
    assert seen == {"substitution", "squares"}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier, seeds=40)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier, seeds=40) >= 30


def _class_of(eigenvalues):
    pos = sum(1 for v in eigenvalues if v > 0)
    neg = sum(1 for v in eigenvalues if v < 0)
    n = len(eigenvalues)
    if pos == n:
        return "positive definite"
    if neg == n:
        return "negative definite"
    if neg == 0:
        return "positive semi-definite"
    if pos == 0:
        return "negative semi-definite"
    return "indefinite"


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("matrix", "number", "number", "mcq"), (2, 2, 1, 2), 1, seeds=40)
    check_follow_through_strings(EXAM, seeds=40)
    check_follow_through_marks(SKILL, EXAM, seeds=40)


def test_exam_keys_match_independent_derivation():
    classes = set()
    for seed in range(80):
        q = make(EXAM, seed=seed, difficulty="exam")
        names, poly = _form(q.stem)
        a = _matrix_of(names, poly)
        eig = sorted(v for v, m in a.eigenvals().items() for _ in range(m))
        assert answer_matrix(q.parts[0].answer) == a
        assert int(q.parts[1].answer) == eig[1]
        assert int(q.parts[2].answer) == eig[0] == a.trace() - eig[1]
        assert q.parts[3].answer_text == _class_of(eig), (seed, eig, q.parts[3].answer_text)
        assert len(q.parts[3].distractors) >= 3
        classes.add(q.parts[3].answer_text)
    assert len(classes) >= 4
