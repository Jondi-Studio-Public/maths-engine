"""Independent oracle for limit-removable/one-sided-limits hard tiers. Re-derives from rendered question text
alone -- does not import mathgen.generators.calculus."""

from __future__ import annotations

import re

from sympy import Rational, limit, symbols, sympify

from mathgen import make

x = symbols("x")

_QOQ_RE = re.compile(r"x → (-?\d+) of \((.+)\) / \((.+)\)")
# Capture each piece as free text between its known anchors and let sympy parse whatever join_terms
# actually produced, rather than re-deriving the exact wording shape here.
_OSQ_RE = re.compile(
    r"f\(x\) = (.+?) for x < (-?\d+), f\(x\) = (.+?) for x ≥ (-?\d+)\. "
    r"Find lim x→(-?\d+)\^([+-]) f\(x\)"
)


def test_oracle_quad_over_quad():
    checked = 0
    for seed in range(200):
        q = make("limits", seed=seed, kinds=("quad_over_quad",))
        m = _QOQ_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        a_val, num_str, den_str = int(m.group(1)), m.group(2), m.group(3)
        num = sympify(num_str)
        den = sympify(den_str)
        expected = limit(num / den, x, a_val)
        actual = sympify(q.answer_text)
        assert expected == actual, f"seed={seed}: {q.text!r} -> {q.answer_text!r} expected {expected}"
        checked += 1
    assert checked == 200


def test_oracle_one_sided_quad():
    checked = 0
    for seed in range(200):
        q = make("limits", seed=seed, kinds=("one_sided_quad",))
        m = _OSQ_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        quad_text, a_pt1, lin_text, a_pt2, arrow_pt, side = m.groups()
        assert a_pt1 == a_pt2 == arrow_pt, f"seed={seed}: boundary point disagrees: {q.text!r}"
        a_pt = int(a_pt1)

        # join_terms renders implicit multiplication ("5x", never "5*x"),
        # which sympify cannot parse on its own -- insert the "*" back
        # between a digit and a variable before parsing.
        def _to_sympy(text):
            text = re.sub(r"(\d)(x)", r"\1*\2", text)
            return sympify(text.replace("^", "**"))

        quad_expr = _to_sympy(quad_text)
        lin_expr = _to_sympy(lin_text)
        # side "+" -> right limit -> the x >= a_pt piece (lin_expr) applies;
        # side "-" -> left limit -> the x < a_pt piece (quad_expr) applies.
        # Both pieces are continuous, so the boundary value is just the
        # exact substitution, no nudge needed for an independent check here.
        expected_expr = lin_expr if side == "+" else quad_expr
        expected = expected_expr.subs(x, a_pt)
        actual = sympify(q.answer_text)
        assert expected == actual, f"seed={seed}: {q.text!r} -> {q.answer_text!r} expected {expected}"
        checked += 1
    assert checked == 200
