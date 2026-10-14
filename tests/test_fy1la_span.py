"""`ucv1-span-membership`: the coefficients that write v as a combination of u and w, and the k that puts v in the span.

The oracle reads the vectors out of the rendered text and solves with SymPy; it never imports the generator.
"""

from __future__ import annotations

import re

import sympy

from fy1la_support import (
    check_exam,
    check_practice,
    check_skill,
    check_variety,
    column,
    exam,
    parse,
    vectors_in,
)
from mathgen import make

SKILL = "ucv1-span-membership"
PRACTICE = "fy1la_span"
EXAM = "exd_x_fy1la_span"
K = sympy.Symbol("k")


def _coefficients(u, w, v):
    a, b = sympy.symbols("a b")
    solved = sympy.solve([a * x + b * y - z for x, y, z in zip(u, w, v)], [a, b], dict=True)
    assert len(solved) == 1, "v is not in the span, or u and w are dependent"
    return sympy.Matrix([solved[0][a], solved[0][b]])


def _oracle(dimension):
    def check(q):
        vs = vectors_in(q.text, ["u", "w", "v"])
        assert all(len(vec) == dimension for vec in vs.values())
        assert column(q.answer) == _coefficients(vs["u"], vs["w"], vs["v"]), q.text

    return check


def test_easy_reads_off_two_coordinates():
    check_practice(PRACTICE, "easy", _oracle(3))
    for seed in range(40):
        vs = vectors_in(make(PRACTICE, seed=seed, difficulty="easy").text, ["u", "w"])
        zeros = sorted(i for i in range(3) if vs["u"][i] == 0 or vs["w"][i] == 0)
        assert len(zeros) == 2, "u and w each have a 1 and a 0 in two shared coordinates"


def test_medium_needs_elimination():
    check_practice(PRACTICE, "medium", _oracle(3))
    for seed in range(40):
        vs = vectors_in(make(PRACTICE, seed=seed, difficulty="medium").text, ["u", "w"])
        minors = [vs["u"][i] * vs["w"][j] - vs["u"][j] * vs["w"][i] for i in range(3) for j in range(i + 1, 3)]
        assert sum(1 for m in minors if m != 0) >= 2


def test_hard_is_four_equations_in_two_unknowns():
    check_practice(PRACTICE, "hard", _oracle(4))


def _exam_oracle(q):
    vs = vectors_in(q.stem, ["u", "w", "v"])
    det_part, k_part, coeff_part, count_part = q.parts
    m = sympy.Matrix.hstack(*[sympy.Matrix(vs[n]) for n in ("u", "w", "v")])
    det = sympy.expand(m.det())
    assert sympy.expand(parse(det_part.answer_text, k=K) - det) == 0
    roots = sympy.solve(det, K)
    assert roots == [k_part.answer]
    v0 = [x.subs(K, k_part.answer) for x in vs["v"]]
    assert column(coeff_part.answer) == _coefficients(vs["u"], vs["w"], v0)
    k1 = int(re.search(r"when k = (-?\d+)", count_part.prompt).group(1))
    assert k1 != k_part.answer
    v1 = sympy.Matrix([x.subs(K, k1) for x in vs["v"]])
    assert sympy.Matrix.hstack(sympy.Matrix(vs["u"]), sympy.Matrix(vs["w"])).rank() == 2
    assert sympy.Matrix.hstack(sympy.Matrix(vs["u"]), sympy.Matrix(vs["w"]), v1).rank() == 3
    assert count_part.answer == "No pair (a, b)"


def test_exam_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [2, 1, 3, 2])


def test_exam_puts_k_in_different_positions():
    positions = {re.search(r"v = \((.*?)\) in", exam(EXAM, s).stem).group(1).split(", ").index("k") for s in range(40)}
    assert positions == {0, 1, 2}


def test_variety_over_200_seeds():
    for tier in ("medium", "hard"):
        check_variety(PRACTICE, tier, minimum=100)
    check_variety(EXAM, "exam", minimum=100)


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "matrix")
