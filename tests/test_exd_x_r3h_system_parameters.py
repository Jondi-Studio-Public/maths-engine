"""Tests for the exam tier `exd_x_r3h_system_parameters` (senior-examine-the-three-cases-for-solutions-of-systems)."""

from __future__ import annotations

import re

import sympy

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_system_parameters"
SKILL = "senior-examine-the-three-cases-for-solutions-of-systems"
X, Y, Z, K, M = sympy.symbols("x y z k m")


def read(q):
    body = re.search(r"constants: (.*)\.$", q.stem).group(1)
    equations = []
    for side in body.split("; "):
        left, right = side.split(" = ")
        equations.append(sympy.Eq(sup.parse_math(left, x=X, y=Y, z=Z, k=K), sup.parse_math(right, k=K, m=M)))
    assert len(equations) == 3
    x0 = sup.first_int(r"where x = (-?\d+)", q.parts[2].prompt)
    return equations, x0


def matrix_and_rhs(equations):
    a, b = sympy.linear_eq_to_matrix([e.lhs - e.rhs for e in equations], [X, Y, Z])
    return a, b


def critical_k(equations):
    a, _ = matrix_and_rhs(equations)
    return sympy.solve(a.det(), K)[0]


def oracle(q):
    equations, x0 = read(q)
    k_star = critical_k(equations)
    a, rhs = matrix_and_rhs([e.subs(K, k_star) for e in equations])
    null = a.T.nullspace()
    assert len(null) == 1
    m_star = sympy.solve((null[0].T * rhs)[0], M)[0]
    full = [e.subs({K: k_star, M: m_star}) for e in equations]
    solution = sympy.solve([e.subs(X, x0) for e in full], [Y, Z], dict=True)
    assert len(solution) == 1
    return {"a": str(k_star), "b": str(m_star), "c": sup.rnd(solution[0][Y], 2)}


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_unique_solution_off_the_critical_value_and_none_off_m():
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        equations, _ = read(q)
        k_star = critical_k(equations)
        a, rhs = matrix_and_rhs([e.subs(K, k_star + 1) for e in equations])
        assert a.det() != 0
        a, rhs = matrix_and_rhs([e.subs({K: k_star, M: oracle(q)["b"]}) for e in equations])
        assert a.rank() == 2 and a.row_join(rhs).rank() == 2
        a, rhs = matrix_and_rhs([e.subs({K: k_star, M: sympy.Rational(oracle(q)["b"]) + 1}) for e in equations])
        assert a.row_join(rhs).rank() == 3


def test_explicit_follow_through_with_a_wrong_k():
    skill = sup.skill(SKILL)
    checked = 0
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        equations, x0 = read(q)
        k_wrong = critical_k(equations) + 1
        a, rhs = matrix_and_rhs([e.subs(K, k_wrong) for e in equations])
        # the learner's own m: the third equation is l1 (eq 1) + l2 (eq 2) on the left, so m = l1 b1 + l2 b2'
        l1, l2 = sympy.symbols("l1 l2")
        combo = sympy.solve([a[2, 0] - l1 * a[0, 0] - l2 * a[1, 0], a[2, 1] - l1 * a[0, 1] - l2 * a[1, 1]], [l1, l2])
        m_wrong = combo[l1] * rhs[0] + combo[l2] * rhs[1]
        pair = [e.subs({K: k_wrong, X: x0}) for e in equations[:2]]
        y_wrong = sympy.solve(pair, [Y, Z], dict=True)[0][Y]
        reply = {"a": str(k_wrong), "b": str(m_wrong), "c": sup.rnd(y_wrong, 2)}
        if len({k_wrong, m_wrong, sympy.Rational(sup.rnd(y_wrong, 2))}) < 3 or 0 in (k_wrong, m_wrong, y_wrong):
            continue
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
        checked += 1
    assert checked >= 20
