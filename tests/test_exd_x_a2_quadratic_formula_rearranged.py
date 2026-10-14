"""Exam source for solving a disguised quadratic by the formula."""

from __future__ import annotations

import re

import numpy
import sympy
from sympy.parsing.sympy_parser import parse_expr

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_a2_quadratic_formula_rearranged"
X = sympy.Symbol("x")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _num(text):
    text = text.replace("^", "**")
    text = re.sub(r"(\d)(sqrt|x|\()", r"\1*\2", text)
    text = re.sub(r"\)\(", ")*(", text)
    return parse_expr(text, local_dict={"x": X})


def _equation(text):
    body = re.match(r"Solve (.*?)(?:, where x is not 0)?\. Give the larger root exactly\.$", text).group(1)
    lhs, rhs = body.split(" = ")
    return _num(lhs) - _num(rhs)


def _check_larger_root(text, key):
    """The key must satisfy the displayed equation exactly and be the larger of its two real roots."""
    expr = _equation(text)
    if "/x" in text:
        expr = sympy.numer(sympy.together(expr))
    poly = sympy.Poly(sympy.expand(expr), X)
    assert poly.degree() == 2
    if sympy.simplify(sympy.expand(poly.as_expr().subs(X, key))) != 0:
        return False
    coefs = [float(c) for c in poly.all_coeffs()]
    return abs(float(key) - float(max(numpy.roots(coefs)))) < 1e-9


def test_key_matches_independent_solution_over_300_seeds():
    bad = []
    for s in range(300):
        q = _q(s)
        if not _check_larger_root(q.text, _num(q.answer_text.split("= ", 1)[1])):
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_all_three_disguises_occur():
    kinds = set()
    for s in range(60):
        t = _q(s).text
        kinds.add("recip" if "/x" in t else "prod" if t.startswith("Solve (") else "both")
    assert kinds == {"recip", "prod", "both"}


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["expression"]
    bad = []
    for s in range(150):
        q = _q(s)
        if not grade(q, q.answer_text).correct:
            bad.append(("key", s))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor", s, d))
        assert len(q.distractors) >= 2
    assert bad == []
