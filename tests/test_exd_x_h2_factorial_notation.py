"""Exam tier of the factorial-notation skill: factor out, combine two ratios, or solve a factorial equation.

The oracle turns the rendered prompt into a sympy expression of ``factorial`` calls and compares values at
integer points, and solves the equation by searching the integers. It shares no code with the generator.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h2_factorial_notation"
_n = sympy.Symbol("n")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(100):
        q = _exam(seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard = {prompt_shape(make("p11c_factorial_notation", seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes - hard) >= 3, (shapes, hard)


def _factorials_to_calls(text: str) -> str:
    text = re.sub(r"\(([^()]*)\)!", r"factorial(\1)", text)
    return re.sub(r"\bn!", "factorial(n)", text)


def _value(expr_text: str, n_value: int):
    return sympy.sympify(_factorials_to_calls(expr_text), locals={"n": _n, "factorial": sympy.factorial}).subs(
        _n, n_value
    )


def _oracle_check(q) -> None:
    text = q.text
    if text.startswith("Given that"):
        lhs, rhs = re.search(r"Given that (.+?) = (\d+) and", text).groups()
        hits = [m for m in range(0, 60) if _value(lhs, m) == int(rhs)]
        assert hits == [int(q.answer_text)], (text, hits, q.answer_text)
        return
    expr_text = re.search(r"Simplify (.+?) for integers", text).group(1)
    # The simplified polynomial has degree at most 3, so eight values pin it down exactly.
    poly = sympy.Poly(sympy.interpolate([(v, _value(expr_text, v)) for v in range(2, 10)], _n), _n)
    term = re.search(r"What is (the constant term|the coefficient of n(?:\^(\d+))?)\?", text)
    degree = 0 if term.group(1) == "the constant term" else int(term.group(2) or 1)
    assert poly.coeff_monomial(_n**degree) == int(q.answer_text), (text, q.answer_text, poly)


def test_key_matches_independent_oracle_over_300_seeds():
    for seed in range(300):
        _oracle_check(_exam(seed))


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["expression"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert q.distractors, seed
        if not grade(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]


def test_restating_the_prompt_expression_is_graded_wrong():
    grade = GRADERS["expression"]
    checked = 0
    for seed in range(200):
        q = _exam(seed)
        if not q.text.startswith("Simplify"):
            continue
        expr_text = re.search(r"Simplify (.+?) for integers", q.text).group(1)
        typed = str(sympy.sympify(_factorials_to_calls(expr_text), locals={"n": _n, "factorial": sympy.factorial}))
        assert not grade(q, typed).correct, (seed, q.text, typed)
        assert not grade(q, expr_text).correct, (seed, q.text)
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d)
        checked += 1
    assert checked >= 100, checked


def test_deterministic_and_all_three_kinds_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    kinds = set()
    for seed in range(60):
        t = _exam(seed).text
        kinds.add("equation" if t.startswith("Given that") else ("factor out" if "!)/" in t else "two ratios"))
    assert len(kinds) == 3, kinds
