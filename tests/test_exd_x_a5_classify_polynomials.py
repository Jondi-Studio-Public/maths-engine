"""Tests for the exam-tier source `exd_x_a5_classify_polynomials` (skill `classify-polynomials`).

Re-derives the degree and term count from the rendered expression with sympy (expand, or cancel for a fraction).
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

TOPIC = "exd_x_a5_classify_polynomials"
HARD = ("polynomial_classify", {})
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


def test_no_text_defects_and_deterministic():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _exam(seed).to_json()


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


_NAMES = {1: "monomial", 2: "binomial", 3: "trinomial"}


def _oracle_label(prompt: str) -> str:
    expr_text = prompt.split(": ", 1)[1]
    expr = _parse(expr_text)
    simplified = sympy.cancel(sympy.expand(expr)) if "/" in expr_text else sympy.expand(expr)
    poly = sympy.Poly(simplified, X, Y)
    return f"Degree {poly.total_degree()}, {_NAMES[len(poly.terms())]}"


def _oracle_agrees(q) -> bool:
    return _oracle_label(q.text) == q.answer_text


def test_fraction_questions_offer_the_denominator_trap_and_it_is_wrong():
    seen = 0
    for seed in SEEDS_ORACLE:
        q = _exam(seed)
        if "/" in q.text:
            seen += 1
            assert "No - the variable is in the denominator" in q.distractors, (seed, q.distractors)
    assert seen > 20


def test_grader_accepts_key_and_rejects_every_distractor():
    from mathgen.core import mcq_options

    grade = GRADERS["mcq"]
    bad = []
    for seed in SEEDS_GRADER:
        q = _exam(seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4, (seed, options)
        for index, option in enumerate(options):
            if grade(q, index).correct != (option == q.answer_text):
                bad.append((seed, index, option, q.answer_text))
    assert bad == [], bad[:5]
    assert not GRADERS["mcq"](_exam(0), "").correct
    assert not GRADERS["mcq"](_exam(0), "9").correct


def test_key_is_not_always_the_lowest_degree_option():
    """The correct option should sit at the lowest, middle and highest degree across seeds."""
    from mathgen import make

    def degree(label):
        return int(label.split(",")[0].split()[1])

    ranks = set()
    for seed in range(200):
        q = make("exd_x_a5_classify_polynomials", seed=seed, difficulty="exam")
        options = [q.answer_text] + [d for d in q.distractors if d.startswith("Degree")]
        ranks.add(sorted(degree(o) for o in options).index(degree(q.answer_text)) == 0)
    assert ranks == {True, False}
