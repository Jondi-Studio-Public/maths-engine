"""`ucv1-linear-independence-r3`: when three vectors in R^3 are dependent, and the dependence relation.

The oracle reads the vectors out of the rendered text and decides with SymPy determinants; it never imports the
generator.
"""

from __future__ import annotations

import re

import sympy

from fy1la_support import (
    check_exam,
    check_practice,
    check_skill,
    check_variety,
    parse,
    vectors_in,
)
from mathgen import make

SKILL = "ucv1-linear-independence-r3"
PRACTICE = "fy1la_independence"
EXAM = "exd_x_fy1la_independence"
K = sympy.Symbol("k")


def _matrix(text):
    vs = vectors_in(text, ["u", "v", "w"])
    return sympy.Matrix.hstack(*[sympy.Matrix(vs[n]) for n in ("u", "v", "w")])


def _easy(q):
    m = _matrix(q.text)
    assert sympy.solve(m.det(), K) == [q.answer]
    assert m.subs(K, q.answer).rank() == 2
    assert m.subs(K, q.answer + 1).rank() == 3


def _relation(m):
    c1, c2 = sympy.symbols("c1 c2")
    solved = sympy.solve(list(c1 * m[:, 0] + c2 * m[:, 1] + m[:, 2]), [c1, c2], dict=True)
    assert len(solved) == 1
    return solved[0][c1], solved[0][c2]


def _medium(q):
    m = _matrix(q.text)
    assert m.det() == 0 and m[:, :2].rank() == 2
    assert _relation(m)[0] == q.answer


def _hard(q):
    m = _matrix(q.text)
    roots = sympy.solve(m.det(), K)
    assert len(roots) == 2 and "k" in q.text
    positive = [r for r in roots if r > 0]
    assert positive == [q.answer] and m.subs(K, q.answer).rank() == 2


def test_easy_matches_the_oracle():
    check_practice(PRACTICE, "easy", _easy)


def test_medium_matches_the_oracle():
    check_practice(PRACTICE, "medium", _medium)


def test_hard_matches_the_oracle_and_states_the_condition():
    check_practice(PRACTICE, "hard", _hard)
    for seed in range(40):
        q = make(PRACTICE, seed=seed, difficulty="hard")
        assert "positive" in q.text or "k > 0" in q.text
        assert str(next(r for r in sympy.solve(_matrix(q.text).det(), K) if r < 0)) in q.distractors


def _exam_oracle(q):
    m = _matrix(q.stem)
    det_part, k_part, c1_part, c2_part = q.parts
    assert sympy.expand(parse(det_part.answer_text, k=K) - sympy.expand(m.det())) == 0
    assert sympy.solve(m.det(), K) == [k_part.answer]
    c1, c2 = _relation(m.subs(K, k_part.answer))
    assert (c1_part.answer, c2_part.answer) == (c1, c2)


def test_exam_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [2, 1, 2, 2])


def test_k_appears_in_every_vector_across_seeds():
    where = set()
    for seed in range(60):
        text = make(EXAM, seed=seed, difficulty="exam").stem
        for name in "uvw":
            if re.search(rf"\b{name} = \([^)]*k", text):
                where.add(name)
    assert where == {"u", "v", "w"}


def test_variety_over_200_seeds():
    for tier in ("easy", "medium", "hard"):
        check_variety(PRACTICE, tier, minimum=100)
    check_variety(EXAM, "exam", minimum=100)


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "number")
