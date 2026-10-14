"""Tests for the exam-tier source `exd_x_a5_factor_monomial_gcf` (skill `factor-common-numerical-factor`).

Re-derives the highest common factor from the rendered expression with sympy and integer gcd.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a5_factor_monomial_gcf"
HARD = ("factor_gcf", {})
X, Y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
SEEDS_ORACLE = range(300)
SEEDS_GRADER = range(150)


def _parse(text: str):
    """Parse student-style algebra text ('3x^2y - 4') with x and y as plain symbols."""
    return parse_expr(text, local_dict={"x": X, "y": Y}, transformations=_TRANSFORMS)


def _exam(seed: int):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed: int):
    topic, params = HARD
    return make(topic, seed=seed, difficulty="hard", **params)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert len(exam_shapes - hard_shapes) >= 3, (exam_shapes, hard_shapes)


def test_independent_oracle_over_300_seeds():
    disagreements = []
    for seed in SEEDS_ORACLE:
        q = _exam(seed)
        if not _oracle_agrees(q):
            disagreements.append((seed, q.text, q.answer_text))
    assert disagreements == [], disagreements[:5]


def _expanded_text(prompt: str) -> str:
    match = re.match(r"Factorise (.+) by taking out the highest common factor\.$", prompt)
    assert match, prompt
    return match.group(1)


def _question_expression(q) -> str:
    return _expanded_text(q.text)


def _oracle_agrees(q) -> bool:
    expr = sympy.expand(_parse(_expanded_text(q.text)))
    cut = q.answer_text.index("(")
    factor = _parse(q.answer_text[:cut])
    bracket = _parse(q.answer_text[cut:])
    if sympy.expand(factor * bracket - expr) != 0:
        return False
    # Highest common factor: nothing further can be taken out of the bracket, and the factor is the
    # integer gcd of the coefficients times the smallest power of each variable.
    if sympy.factor_terms(bracket) != bracket:
        return False
    poly = sympy.Poly(expr, X, Y)
    gcd_coef = 0
    for _, coef in poly.terms():
        gcd_coef = sympy.igcd(gcd_coef, abs(int(coef)))
    min_x = min(m[0] for m, _ in poly.terms())
    min_y = min(m[1] for m, _ in poly.terms())
    return factor == gcd_coef * X**min_x * Y**min_y


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["expression"]
    bad = []
    for seed in SEEDS_GRADER:
        q = _exam(seed)
        assert 2 <= len(q.distractors) <= 3, (seed, q.distractors)
        if not grade(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_rejects_restating_the_question_and_near_misses():
    grade = GRADERS["expression"]
    for seed in range(60):
        q = _exam(seed)
        assert not grade(q, _question_expression(q)).correct, (seed, q.text)
        assert not grade(q, q.answer_text + " + 1").correct, (seed, q.answer_text)
        assert not grade(q, "").correct
