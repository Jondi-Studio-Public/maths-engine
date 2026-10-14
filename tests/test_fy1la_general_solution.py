"""`ucv1-general-solution-parametric`: solutions of a consistent system with free variables.

The oracle reads the system out of the rendered text and solves it with SymPy; it never imports the generator.
"""

from __future__ import annotations

import re

import sympy

from fy1la_support import (
    check_exam,
    check_follow_through,
    check_practice,
    check_skill,
    check_variety,
    column,
    parse,
    system_in,
    text_matrix,
)
from mathgen import make
from mathgen.grading import grade

SKILL = "ucv1-general-solution-parametric"
PRACTICE = "fy1la_general_solution"
EXAM = "exd_x_fy1la_general_solution"


def _system(text):
    """The system in a practice stem, without the clause that fixes a free variable."""
    return system_in(re.split(r", given that| has infinitely| is consistent|\. ", text)[0])


def _solve_fixed(a, b, symbols, fixed):
    """The solution of a x = b with some unknowns fixed, which must be unique."""
    eqs = [sum(a[r, c] * symbols[c] for c in range(a.cols)) - b[r] for r in range(a.rows)]
    eqs += [symbols[i] - v for i, v in fixed.items()]
    solved = sympy.solve(eqs, symbols, dict=True)
    assert len(solved) == 1, solved
    return sympy.Matrix([solved[0][s] for s in symbols])


def _easy(q):
    a, b, symbols = _system(q.text)
    assert a.shape == (2, 3) and a.rank() == 2
    z = int(re.search(r"(?:given that|in which) z = (-?\d+)", q.text).group(1))
    assert column(q.answer) == _solve_fixed(a, b, symbols, {2: z})


def _medium(q):
    a, b, symbols = _system(q.text)
    assert a.shape == (3, 3) and a.rank() == 2 and a.row_join(b).rank() == 2
    direction = column(q.answer)
    null = a.nullspace()
    assert len(null) == 1 and null[0].cross(direction).norm() == 0
    assert q.answer_form == "scalar_multiple"
    assert grade("matrix", q, "(" + ", ".join(str(-2 * v) for v in direction) + ")").correct


def _hard(q):
    a, b, symbols = _system(q.text)
    assert a.shape == (3, 4) and a.rank() == 2 and a.row_join(b).rank() == 2
    s0, t0 = (int(v) for v in re.search(r"x3 = (-?\d+) and x4 = (-?\d+)", q.text).groups())
    assert column(q.answer) == _solve_fixed(a, b, symbols, {2: s0, 3: t0})


def test_easy_matches_the_oracle():
    check_practice(PRACTICE, "easy", _easy)


def test_medium_matches_the_oracle_and_refuses_scalar_multiples_of_wrong_vectors():
    check_practice(PRACTICE, "medium", _medium)
    for seed in range(40):
        q = make(PRACTICE, seed=seed, difficulty="medium")
        for d in q.distractors:
            wrong = text_matrix(d)
            assert not grade("matrix", q, "(" + ", ".join(str(3 * v) for v in wrong) + ")").correct


def test_hard_matches_the_oracle():
    check_practice(PRACTICE, "hard", _hard)


def _exam_oracle(q):
    a, b, symbols = system_in(q.stem)
    assert a.shape == (3, 4)
    rank_part, params_part, particular_part, expr_part = q.parts
    assert rank_part.answer == a.rank() == 2 and a.row_join(b).rank() == 2
    assert params_part.answer == 4 - a.rank() and params_part.follow_through == "4 - part_a"
    free = [int(n) - 1 for n in re.findall(r"Let x(\d) = s and x(\d) = t", expr_part.prompt)[0]]
    particular = _solve_fixed(a, b, symbols, {free[0]: 0, free[1]: 0})
    assert column(particular_part.answer) == particular
    s, t = sympy.symbols("s t")
    general = _solve_fixed(a, b, symbols, {free[0]: s, free[1]: t})
    name = re.search(r"Write (x\d) in terms", expr_part.prompt).group(1)
    wanted = general[int(name[1]) - 1]
    assert sympy.expand(parse(expr_part.answer_text, s=s, t=t) - wanted) == 0, (expr_part.answer_text, wanted)


def test_exam_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [1, 1, 2, 3])


def test_follow_through_from_a_wrong_rank():
    check_follow_through(EXAM, SKILL, "b", lambda q: ("a", "3", "1"))


def test_variety_over_200_seeds():
    for tier in ("easy", "medium", "hard"):
        check_variety(PRACTICE, tier, minimum=100)
    check_variety(EXAM, "exam", minimum=100)


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "matrix")
