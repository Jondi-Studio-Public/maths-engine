"""Exam source for expanding cubics from repeated, cubed, quadratic and difference-of-squares factors."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import parse_expr

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_a2_expand_cubic_shapes"


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(text, sym):
    text = text.replace("^", "**").replace(")(", ")*(")
    text = re.sub(r"\)\*\*(\d)\(", r")**\1*(", text)
    text = re.sub(r"(\d)([a-z])", r"\1*\2", text)
    return parse_expr(text, local_dict={str(sym): sym})


def _variable(text):
    return next(c for c in "xynt" if re.search(rf"\b{c}\b|\d{c}|{c}\^", text.split("(", 1)[1]))


def _key(text):
    m = re.search(r"(\(.*\)(?:\^\d)?)", text)
    var = _variable(text)
    sym = sympy.Symbol(var)
    return sympy.expand(_parse(m.group(1), sym)), sym


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {
        prompt_shape(make("expand_quadratic_and_cubic_polynomials_from_factors", seed=s, difficulty="hard"))
        for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


def test_key_matches_independent_expansion_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        want, sym = _key(q.text)
        got = _parse(q.answer_text, sym)
        if sympy.expand(want - got) != 0 or sympy.degree(want, sym) != 3:
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_all_four_shapes_occur():
    kinds = set()
    for s in range(120):
        t = _q(s).text
        if "^3" in t:
            kinds.add("cube")
        elif ")^2(" in t:
            kinds.add("square")
        elif re.search(r"\(\w\^2", t):
            kinds.add("quad")
        else:
            kinds.add("dos")
    assert len(kinds) == 4


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["expression"]
    bad = []
    for s in range(200):
        q = _q(s)
        if not grade(q, q.answer_text).correct:
            bad.append(("key", s))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor", s, d))
        echo = re.search(r"(\(.*\)(?:\^\d)?)", q.text).group(1)
        if grade(q, echo).correct:
            bad.append(("echo", s))
        assert len(q.distractors) >= 2
    assert bad == []


def test_deterministic():
    assert _q(9).to_json() == _q(9).to_json()
