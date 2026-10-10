"""Shared helpers for the `test_exd_x_c1_*` files: independent parsing of a rendered prompt.

Nothing here imports a generator module. The prompt text and the answer text a learner would see are parsed
back into sympy expressions, so every oracle in the test files re-derives its key from what was actually shown.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_FUNCTION_RE = re.compile(r"= (.+?)(?:\.$|, find|, giving| has a stationary)")


def parse_math(text: str):
    """Parse learner-style maths (``3sin^2(4x)``, ``(2x + 1)^3(x^2 - 5)``, ``√x``) into a sympy expression."""
    s = text.replace("√(", "sqrt(").replace("√x", "sqrt(x)")
    s = re.sub(r"(sin|cos)\^(\d)\(([^()]*)\)", r"(\1(\3))**\2", s)
    s = s.replace("^", "**")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": x})


def function_in_prompt(prompt: str):
    """The function the prompt defines, parsed from the rendered text alone."""
    match = _FUNCTION_RE.search(prompt)
    assert match, prompt
    return parse_math(match.group(1))


_PROBES = [sympy.Rational(7, 3), sympy.Rational(11, 5), sympy.Rational(-5, 7), sympy.Rational(13, 4), 3, 5, -2]


def same_function(left, right) -> bool:
    """True when two expressions agree numerically at several probe points (at least three must be finite)."""
    agreed = 0
    for point in _PROBES:
        a = complex(sympy.N(left.subs(x, point)))
        b = complex(sympy.N(right.subs(x, point)))
        if any(v != v or abs(v) == float("inf") for v in (a, b)):
            continue
        if abs(a - b) > 1e-9 * (1 + abs(a)):
            return False
        agreed += 1
    assert agreed >= 3, (left, right)
    return True


def shapes(topic: str, n: int = 60, **kwargs) -> set[str]:
    return {prompt_shape(make(topic, seed=s, **kwargs)) for s in range(n)}


def standard_checks(topic: str, hard_kinds: tuple, oracle_derivative_order: int = 1):
    """The three families of checks every file runs, returned as plain callables so each file can name its tests."""

    def variety():
        exam = shapes(topic, difficulty="exam")
        hard = shapes("differentiation", difficulty="hard", kinds=hard_kinds)
        assert len(exam) >= 3, exam
        assert exam - hard, (exam, hard)

    def key_from_prompt(n: int = 300):
        bad = []
        for seed in range(n):
            q = make(topic, seed=seed, difficulty="exam")
            f = function_in_prompt(q.text)
            expected = sympy.diff(f, x, oracle_derivative_order)
            if not same_function(expected, parse_math(q.answer_text)):
                bad.append((seed, q.text, q.answer_text))
        assert bad == [], bad[:3]

    def grader(kind: str = "expression", n: int = 120):
        bad = []
        for seed in range(n):
            q = make(topic, seed=seed, difficulty="exam")
            if not GRADERS[kind](q, q.answer_text).correct:
                bad.append(("key rejected", seed, q.answer_text))
            assert len(q.distractors) >= 2, (seed, q.distractors)
            for d in q.distractors:
                if GRADERS[kind](q, d).correct:
                    bad.append(("distractor accepted", seed, d, q.answer_text))
        assert bad == [], bad[:3]

    def clean_and_deterministic(n: int = 60):
        for seed in range(n):
            q = make(topic, seed=seed, difficulty="exam")
            assert find_text_defects(q) == [], (seed, q.text)
            assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()

    return variety, key_from_prompt, grader, clean_and_deterministic
