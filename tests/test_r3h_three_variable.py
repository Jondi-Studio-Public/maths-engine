"""``three-variable-elimination-triple``: answers grade right, every distractor wrong, and solve the printed system."""

from __future__ import annotations

import re

import sympy

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_three_variable"
EXAM = "exd_x_r3h_three_variable"
add_common_tests(globals(), PRACTICE, EXAM)

x, y, z = sympy.symbols("x y z")


def _equations(text):
    """The printed equations as SymPy relations, read back from the question text alone."""
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    segment = text.split(":", 1)[1].split("(a)")[0]
    found = re.findall(r"\(\d\) ([^;,]+?)(?=[;,.]|$)", segment)
    out = []
    for item in found:
        left, right = item.split("=")
        tr = standard_transformations + (implicit_multiplication_application,)
        out.append(sympy.Eq(parse_expr(left, transformations=tr), parse_expr(right, transformations=tr)))
    return out


def test_printed_equations_have_exactly_the_stated_solution():
    for tier in ("easy", "medium", "hard"):
        for q in questions(PRACTICE, tier, range(1, 13)):
            eqs = _equations(q.text)
            assert len(eqs) == 3, q.text
            solved = sympy.solve(eqs, [x, y, z], dict=True)
            assert len(solved) == 1, q.text
            assert [solved[0][v] for v in (x, y, z)] == list(q.answer), q.text


def test_easy_systems_have_two_equations_in_two_unknowns():
    for q in questions(PRACTICE, "easy", range(1, 13)):
        eqs = _equations(q.text)
        sizes = sorted(len((e.lhs - e.rhs).free_symbols) for e in eqs)
        assert sizes[:2] == [2, 2], q.text


def test_hard_systems_carry_unknowns_on_both_sides():
    for q in questions(PRACTICE, "hard", range(1, 13)):
        eqs = _equations(q.text)
        assert all(e.rhs.free_symbols for e in eqs), q.text


def test_exam_first_part_cancels_z_and_matches_the_sum():
    for q in questions(EXAM, "exam", range(1, 13)):
        eqs = _equations(q.text)
        total = (eqs[0].lhs + eqs[1].lhs) - (eqs[0].rhs + eqs[1].rhs)
        a = [int(v.strip()) for v in q.parts[0].answer_text.split(",")]
        assert total.coeff(z) == 0
        assert sympy.expand(total - (a[0] * x + a[1] * y - a[2])) == 0, q.text
        solved = sympy.solve(eqs, [x, y, z], dict=True)[0]
        assert [int(p.answer_text) for p in q.parts[1:]] == [solved[x], solved[y], solved[z]]
