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
    answer = parse_expr(q.answer_text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"n": _n})
    for n_value in range(2, 9):
        assert _value(expr_text, n_value) == answer.subs(_n, n_value), (text, q.answer_text, n_value)


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


def test_grader_accepts_an_expanded_equivalent():
    q = next(q for q in (_exam(s) for s in range(60)) if q.text.startswith("Simplify"))
    key = parse_expr(q.answer_text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"n": _n})
    expanded = str(sympy.expand(key))
    assert GRADERS["expression"](q, expanded).correct


def test_deterministic_and_all_three_kinds_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    kinds = set()
    for seed in range(60):
        t = _exam(seed).text
        kinds.add("equation" if t.startswith("Given that") else ("factor out" if "!)/" in t else "two ratios"))
    assert len(kinds) == 3, kinds
