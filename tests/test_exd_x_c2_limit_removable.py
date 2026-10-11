"""Tests for the exam-tier source of ``limit-removable`` (exd_x_c2_limit_removable).

The key is re-derived from the rendered prompt: the expression is parsed and evaluated with 60-digit arithmetic
at the stated point plus and minus 10^-12; the nearest fraction with a small denominator must equal the key.
Where the prompt hides a constant k, k is first fixed by requiring the numerator to vanish at the point.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mpmath
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


@pytest.fixture(autouse=True)
def _mp_precision():
    """60 digits for this module's tests only: mpmath's precision is global, and a module-level `mp.dps = 60`
    stayed set for every later test in the worker (it made sympy's `nsolve` demand a near-zero residual)."""
    with mpmath.mp.workdps(60):
        yield


TOPIC = "exd_x_c2_limit_removable"
GRADER = "number"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("limits", seed=seed, difficulty="hard", **{"kinds": ("quad_over_quad",)})


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


_PATTERNS = (
    re.compile(r"^Find the limit as x → (-?\d+) of (.+)\.$"),
    re.compile(r"^Evaluate lim x→(-?\d+) of (.+)\.$"),
    re.compile(r"^Determine the limit of (.+) as x → (-?\d+)\.$"),
    re.compile(
        r"^The limit as x → (-?\d+) of (.+) exists and is finite, where k is a constant\. Find the value of the limit\.$"
    ),
    re.compile(r"^For some constant k, lim x→(-?\d+) of (.+) exists and is finite\. Find its value\.$"),
)


def _split(text):
    for i, pattern in enumerate(_PATTERNS):
        m = pattern.match(text)
        if m:
            return (m.group(1), m.group(2)) if i != 2 else (m.group(2), m.group(1))
    raise AssertionError(text)


def _oracle(text):
    point_text, expr_text = _split(text)
    point = int(point_text)
    k = sympy.Symbol("k")
    expr = parse_expr(
        expr_text.replace("√", " sqrt").replace("sin", " sin").replace("cos", " cos"),
        transformations=_TRANSFORMS,
        local_dict={"x": _X, "k": k},
    )
    if expr.has(k):
        numerator, _ = sympy.fraction(sympy.together(expr))
        k_value = sympy.solve(numerator.subs(_X, point), k)[0]
        expr = expr.subs(k, k_value)
    h = sympy.Float("1e-12", 60)
    values = [expr.subs(_X, point + s * h).evalf(60) for s in (1, -1)]
    assert abs(values[0] - values[1]) < sympy.Float("1e-8"), (text, values)
    return Fraction(float(values[0])).limit_denominator(1000)


def test_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        if _oracle(q.text) != _key(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad


def test_every_family_occurs():
    texts = [_exam(s).text for s in range(80)]
    assert any("√" in t for t in texts)
    assert any("sin(" in t and "cos" not in t for t in texts)
    assert any("cos(" in t for t in texts)
    assert any("constant k" in t for t in texts)
