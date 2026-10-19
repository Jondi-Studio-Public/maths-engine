"""Exam source for building real quadratics from complex-conjugate solutions."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import parse_expr

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_a2_complex_quadratic"
X = sympy.Symbol("x")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _poly(text):
    return parse_expr(re.sub(r"(\d)([a-z])", r"\1*\2", text.replace("^", "**")), local_dict={"x": X})


def _independent_key(text):
    """Re-derive the polynomial from the rendered prompt using complex arithmetic."""
    m = re.match(r"(.*) is one solution of the equation ax\^2 \+ bx \+ c = 0, where a = (\d+)", text)
    if m:
        z = sympy.sympify(re.sub(r"(\d)I", r"\1*I", m.group(1).replace("i", "I")))
        a = int(m.group(2))
        return sympy.expand(a * (X - z) * (X - sympy.conjugate(z)))
    m = re.match(r"The solutions of (.*) = 0 are u and v\. .* whose solutions are (.*)\.$", text)
    eq, new = m.groups()
    u, v = sympy.solve(sympy.Eq(_poly(eq), 0), X)
    uu, vv = sympy.symbols("u v")
    spec = sympy.sympify(re.sub(r"(\d)([uv])", r"\1*\2", new.replace("^", "**")).split(" and ")[0], locals={"u": uu})
    spec2 = sympy.sympify(re.sub(r"(\d)([uv])", r"\1*\2", new.replace("^", "**")).split(" and ")[1], locals={"v": vv})
    return sympy.expand((X - spec.subs(uu, u)) * (X - spec2.subs(vv, v)))


def test_key_matches_complex_arithmetic_over_300_seeds():
    bad = []
    shapes = set()
    for s in range(300):
        q = _q(s)
        want = _independent_key(q.text)
        got = _poly(q.answer_text)
        shapes.add("one solution" in q.text or re.search(r"whose solutions are (\S+)", q.text).group(1)[-1])
        if sympy.expand(want - got) != 0:
            bad.append((s, q.text, q.answer_text))
    assert bad == []
    assert len(shapes) >= 3


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


def test_key_is_a_genuine_trinomial_and_deterministic():
    for s in range(200):
        assert all(sympy.Poly(_poly(_q(s).answer_text), X).all_coeffs())
    assert _q(5).to_json() == _q(5).to_json()
