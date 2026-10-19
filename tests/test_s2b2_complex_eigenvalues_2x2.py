"""Tests `uc2-complex-eigenvalues-2x2` (topic `s2b2_complex_eigenvalues_2x2`) and its exam chain.

The oracle reads A back out of the question TEXT and takes the eigenvalue with positive imaginary part from
`Matrix.eigenvals()`; modulus, argument, and powers of A are then found with SymPy's `Abs`, `arg` and `A**n`.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2b2_complex_eigenvalues_build  # noqa: F401  (registers the topics)
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
    numerically_equal,
    parse,
)

TOPIC = "s2b2_complex_eigenvalues_2x2"
EXAM = "exd_x_s2b2_complex_eigenvalues_2x2"
SKILL = load_skill("uc2-complex-eigenvalues-2x2")
_POWERS = {"²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁻¹": -1}


def upper_eigenvalue(a):
    eig = [e for e in a.eigenvals() if sympy.im(e) > 0]
    assert len(eig) == 1
    return eig[0]


def value(answer):
    return parse(str(answer))


def test_wiring():
    check_wiring(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["ucv1-eigenvectors", "senior-determine-complex-conjugate-solutions-of-real-quadratic-equations"],
    )


def test_easy_key_is_the_imaginary_part():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        a = matrices(q.text)[0]
        assert value(q.answer) == sympy.im(upper_eigenvalue(a)), (seed, q.text)


def test_medium_key_is_the_modulus_of_the_eigenvalues_of_a_power():
    powers = set()
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="medium")
        a = matrices(q.text)[0]
        k = 1
        m = re.search(r"of (A[²³]|A⁻¹)", q.text)
        if m:
            k = _POWERS[m.group(1)[1:]]
        powers.add(k)
        z = upper_eigenvalue(a)
        expected = sympy.Abs(z) ** k
        assert numerically_equal(value(q.answer), expected), (seed, q.text, q.answer)
        assert sympy.simplify(value(q.answer) - sympy.simplify(expected)) == 0
        if k != 1:
            # The eigenvalues of A**k are the powers of the eigenvalues of A (SymPy, independently).
            powered = (a**k).eigenvals()
            assert any(sympy.simplify(sympy.Abs(e) - expected) == 0 for e in powered), (seed, q.text)
    assert powers == {1, 2, 3, -1}


def test_hard_key_is_the_argument_or_an_entry_of_a_power():
    seen = set()
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="hard")
        a = matrices(q.text)[0]
        if "entry" in q.text:
            seen.add("entry")
            i, j = (int(v) - 1 for v in re.search(r"\((\d), (\d)\) entry", q.text).groups())
            n = _POWERS[re.search(r"A([²³⁴⁵⁶⁷⁸])", q.text).group(1)]
            assert value(q.answer) == (a**n)[i, j], (seed, q.text)
        else:
            seen.add("argument")
            theta = sympy.arg(upper_eigenvalue(a))
            assert numerically_equal(value(q.answer), theta), (seed, q.text, q.answer)
    assert seen == {"entry", "argument"}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier, seeds=25)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier, seeds=40) >= 25


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("number", "number", "expression", "expression"), (1, 2, 1, 2), 2, seeds=20)
    check_follow_through_strings(EXAM, seeds=40)
    check_follow_through_marks(SKILL, EXAM, seeds=40)


def test_exam_keys_match_independent_derivation():
    quadrants = set()
    for seed in range(80):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrices(q.stem)[0]
        z = upper_eigenvalue(a)
        assert int(q.parts[0].answer) == sympy.re(z)
        assert int(q.parts[1].answer) == sympy.im(z)
        assert numerically_equal(value(q.parts[2].answer), sympy.Abs(z))
        assert numerically_equal(value(q.parts[3].answer), sympy.arg(z))
        quadrants.add(sympy.re(z) > 0)
    assert quadrants == {True, False}
