"""``separable-de-particular-solution``: answers grade right, every distractor wrong, tiers differ, exam parts chain."""

from __future__ import annotations

import re

import sympy

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_separable_de"
EXAM = "exd_x_r3h_separable_de"
add_common_tests(globals(), PRACTICE, EXAM)


def _parse(text):
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    text = re.sub(r"(?<![A-Za-z])e\^", "E^", text)
    return parse_expr(
        text,
        local_dict={"x": sympy.Symbol("x"), "y": sympy.Symbol("y"), "E": sympy.E},
        transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
    )


def test_answer_passes_through_the_given_point_and_solves_the_equation():
    """The explicit solution is checked against the stem by a route that does not use the generator."""
    for tier in ("easy", "medium", "hard"):
        for q in questions(PRACTICE, tier, range(1, 13)):
            m = re.search(r"dy/dx = (.+?)(?: given| and passes)", q.text)
            assert m, q.text
            pt = re.search(r"y = (\S+) when x = (\S+?)\.", q.text) or re.search(r"point \((\S+), (\S+)\)", q.text)
            assert pt, q.text
            if "when" in q.text:
                y0, x0 = pt.group(1), pt.group(2)
            else:
                x0, y0 = pt.group(1).rstrip(","), pt.group(2).rstrip(")")
            x, y = sympy.symbols("x y")
            sol = sympy.sympify(q.answer)
            assert abs(float(sol.subs(x, sympy.Rational(x0)) - sympy.Rational(y0))) < 1e-9, q.text
            rhs = _parse(m.group(1))
            for p in (0.2, 0.4):
                lhs_v = float(sympy.diff(sol, x).subs(x, float(x0) + p))
                rhs_v = float(rhs.subs({y: sol.subs(x, float(x0) + p)}).subs(x, float(x0) + p))
                assert abs(lhs_v - rhs_v) < 1e-7 * max(1, abs(lhs_v)), q.text
