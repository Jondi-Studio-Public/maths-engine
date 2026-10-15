"""Exam source for finding a quadratic from graph features."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import parse_expr

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_a2_quadratic_from_conditions"
X = sympy.Symbol("x")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _poly(answer_text):
    body = answer_text.split("=", 1)[1].replace("^", "**")
    return parse_expr(re.sub(r"(\d)x", r"\1*x", body), local_dict={"x": X})


def _satisfies_prompt(text, poly):
    """Check the stated features directly against the polynomial, never solving for it."""
    deriv = sympy.diff(poly, X)
    m = re.search(r"touches the x-axis at x = (-?\d+) and passes through the point \((-?\d+), (-?\d+)\)", text)
    if m:
        h, px, py = map(int, m.groups())
        return poly.subs(X, h) == 0 and deriv.subs(X, h) == 0 and poly.subs(X, px) == py
    m = re.search(
        r"axis of symmetry x = (-?\d+) and passes through the points \((-?\d+), (-?\d+)\) and \((-?\d+), (-?\d+)\)",
        text,
    )
    if m:
        h, x1, y1, x2, y2 = map(int, m.groups())
        return deriv.subs(X, h) == 0 and poly.subs(X, x1) == y1 and poly.subs(X, x2) == y2
    m = re.search(r"x-intercepts at x = (-?\d+) and x = (-?\d+) and a (minimum|maximum) value of (-?\d+)", text)
    assert m, text
    r1, r2, kind, val = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
    mid = sympy.Rational(r1 + r2, 2)
    lead = sympy.Poly(poly, X).all_coeffs()[0]
    return (
        poly.subs(X, r1) == 0
        and poly.subs(X, r2) == 0
        and poly.subs(X, mid) == val
        and (lead > 0) == (kind == "minimum")
    )


def test_key_satisfies_every_stated_condition_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        poly = _poly(q.answer_text)
        if sympy.Poly(poly, X).degree() != 2 or not _satisfies_prompt(q.text, poly):
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_key_is_unique_because_distractors_violate_conditions():
    bad = []
    for s in range(150):
        q = _q(s)
        for d in q.distractors:
            if _satisfies_prompt(q.text, _poly(d)):
                bad.append((s, d))
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
        assert len(q.distractors) >= 2
    assert bad == []
