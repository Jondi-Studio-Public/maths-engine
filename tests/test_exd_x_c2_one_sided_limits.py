"""Tests for the exam-tier source of ``one-sided-limits`` (exd_x_c2_one_sided_limits).

The key is re-derived from the rendered prompt: for an asymptote the quotient is evaluated exactly at a plus and
minus 10^-6 on the stated side(s); for the parameter question both pieces are parsed and k is solved from equal
left and right values at the break point.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
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

TOPIC = "exd_x_c2_one_sided_limits"
GRADER = "expression"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("limits", seed=seed, difficulty="hard", **{"kinds": ("one_sided_quad",)})


def _key(q):
    """The answer as an exact Fraction (integers and {p, q} dicts)."""
    if isinstance(q.answer, dict):
        return Fraction(q.answer["p"], q.answer["q"])
    return Fraction(q.answer)


def _parse(text):
    """Parse learner-style maths text with sympy: powers of functions, root sign, implicit products."""
    text = text.replace("√", " sqrt").replace("−", "-")
    text = re.sub(r"(sin|cos)\^(\d+)\(([^()]*)\)", r"(\1(\3))**\2", text)
    return parse_expr(text, transformations=_TRANSFORMS, local_dict={"x": _X})


def test_no_text_defects_over_60_seeds():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes, "the exam tier adds no prompt shape beyond the hard tier"


def test_deterministic():
    for seed in (0, 1, 2, 3, 100, 12345):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_key_accepted_and_every_distractor_rejected_over_120_seeds():
    grade = GRADERS[GRADER]
    for seed in range(120):
        q = _exam(seed)
        assert q.distractors, seed
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors
        assert grade(q, q.answer_text).correct, (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, q.text, q.answer_text, d)


def test_wrong_answer_rejected():
    grade = GRADERS[GRADER]
    for seed in range(40):
        q = _exam(seed)
        assert not grade(q, "12345").correct
        assert not grade(q, "banana").correct


_ASYMPTOTE = (
    re.compile(r"^Find lim x→(-?\d+)(\^[+-])? of (.+)\. Give your answer as a number, ∞ or -∞\.$"),
    re.compile(r"^Evaluate the limit as x → (-?\d+)(\^[+-])? of (.+)\. Give your answer as a number, ∞ or -∞\.$"),
)
_PARAM = (
    re.compile(r"^f\(x\) = (.+) for x < (-?\d+), f\(x\) = (.+) for x ≥ -?\d+\. Find the value of k"),
    re.compile(r"^A function is defined by f\(x\) = (.+) when x < (-?\d+) and f\(x\) = (.+) when x ≥ -?\d+\. For what"),
)


def _oracle(text):
    for pattern in _ASYMPTOTE:
        m = pattern.match(text)
        if m:
            a = int(m.group(1))
            side = m.group(2)
            expr = _parse(m.group(3))
            sides = {"^+": [1], "^-": [-1], None: [1, -1]}[side]
            signs = set()
            for s in sides:
                x_value = sympy.Rational(a) + s * sympy.Rational(1, 10**6)
                value = expr.subs(_X, x_value)
                assert abs(value) > 1000, text
                signs.add(value > 0)
            assert len(signs) == 1, ("sides disagree", text)
            return "∞" if signs.pop() else "-∞"
    for pattern in _PARAM:
        m = pattern.match(text)
        if m:
            k = sympy.Symbol("k")
            a = int(m.group(2))
            left = parse_expr(
                m.group(1).replace("kx", "k*x"), transformations=_TRANSFORMS, local_dict={"x": _X, "k": k}
            )
            right = parse_expr(m.group(3), transformations=_TRANSFORMS, local_dict={"x": _X, "k": k})
            solution = sympy.solve(left.subs(_X, a) - right.subs(_X, a), k)
            assert len(solution) == 1
            return str(solution[0])
    raise AssertionError(text)


def test_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        if _oracle(q.text) != q.answer_text:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad


def test_both_families_and_all_sides_occur():
    texts = [_exam(s).text for s in range(120)]
    assert any("Find the value of k" in t or "For what value of k" in t for t in texts)
    assert any("^+" in t for t in texts) and any("^-" in t for t in texts)
    assert any("^" not in t.split("of")[0] and "limit" in t and "k" not in t for t in texts)
