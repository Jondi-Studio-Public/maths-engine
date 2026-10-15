"""Exam source for expanding sums and differences of binomial products into a trinomial."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import parse_expr

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_a2_expand_binomial_sums"


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("expand_binomial_product_monic", seed=seed, difficulty="hard")


def _key_from_prompt(text):
    body = re.search(r"(\(.*\)(?:\^2)?)", text).group(1)
    body = body.replace("^", "**")
    var = next(c for c in "xynt" if c in body)
    # insert explicit multiplication between adjacent brackets
    body = re.sub(r"(\d)([a-z])", r"\1*\2", body.replace(")(", ")*("))
    expr = parse_expr(body, local_dict={var: sympy.Symbol(var)})
    return sympy.expand(expr), sympy.Symbol(var)


def test_key_matches_independent_expansion_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        expected, sym = _key_from_prompt(q.text)
        given = parse_expr(
            re.sub(r"(\d)([a-z])", r"\1*\2", q.answer_text.replace("^", "**")), local_dict={str(sym): sym}
        )
        poly = sympy.Poly(expected, sym)
        if sympy.expand(given - expected) != 0 or poly.degree() != 2 or any(c == 0 for c in poly.all_coeffs()):
            bad.append((s, q.text, q.answer_text))
    assert bad == []


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
        # the unexpanded prompt echoed back is correct by value but must not be accepted
        echo = re.search(r"(\(.*\)(?:\^2)?)", q.text).group(1)
        if grade(q, echo).correct:
            bad.append(("echo", s))
    assert bad == []


def test_distractors_present_and_deterministic():
    assert all(len(_q(s).distractors) >= 1 for s in range(100))
    assert _q(7).to_json() == _q(7).to_json()
