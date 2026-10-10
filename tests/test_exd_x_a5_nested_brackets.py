"""Tests for the exam-tier source `exd_x_a5_nested_brackets` (skill `simplify-bracketed-expressions`).

Re-derives the key by expanding the rendered expression (square brackets read as round) with sympy.
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

TOPIC = "exd_x_a5_nested_brackets"
HARD = ("simplify_expression", {"kind": "brackets"})
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


def _expression_text(prompt: str) -> str:
    match = re.match(r"Simplify (.+)\.$", prompt)
    assert match, prompt
    return match.group(1)


def _question_expression(q) -> str:
    return _expression_text(q.text)


def _oracle_agrees(q) -> bool:
    text = _expression_text(q.text).replace("[", "(").replace("]", ")")
    expected = sympy.expand(_parse(text))
    return sympy.expand(expected - _parse(q.answer_text)) == 0


def test_every_question_has_a_nested_bracket():
    for seed in SEEDS_ORACLE:
        text = _expression_text(_exam(seed).text)
        assert "[" in text and text.count("(") == 2, text


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
