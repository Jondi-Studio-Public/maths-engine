"""Tests for the exam tier `exd_x_r3h_ideal_unknown_coefficients` (ucv1-ideal-membership)."""

from __future__ import annotations

import re

import sympy

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_ideal_unknown_coefficients"
SKILL = "ucv1-ideal-membership"
X, H, K = sympy.symbols("x h k")


def read(q):
    g = sup.parse_math(re.search(r"g\(x\) = (.*?)\. The polynomial", q.stem).group(1), x=X)
    f = sup.parse_math(re.search(r"f\(x\) = (.*?), where h and k", q.stem).group(1), x=X, h=H, k=K)
    n = sup.first_int(r"find f\((-?\d+)\)", q.parts[2].prompt)
    return sympy.expand(g), sympy.expand(f), n


def solve_unknowns(g, f):
    remainder = sympy.Poly(sympy.rem(f, g, X), X)
    solution = sympy.solve(remainder.all_coeffs(), [H, K], dict=True)
    assert len(solution) == 1
    return solution[0]


def oracle(q):
    g, f, n = read(q)
    solution = solve_unknowns(g, f)
    value = f.subs(solution).subs(X, n)
    return {"a": str(solution[H]), "b": str(solution[K]), "c": str(value)}


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_the_completed_polynomial_is_a_multiple_of_the_generator():
    for seed in range(80):
        q = sup.draw(TOPIC, seed)
        g, f, _ = read(q)
        full = f.subs(solve_unknowns(g, f))
        assert sympy.rem(full, g, X) == 0
        assert sympy.degree(sympy.quo(full, g, X), X) == 2


def test_explicit_follow_through_from_a_wrong_h():
    skill = sup.skill(SKILL)
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        g, f, n = read(q)
        true = solve_unknowns(g, f)
        h_wrong = true[H] + (2 if true[H] != -2 else 3)
        # the learner's own k: make the x coefficient of the remainder vanish with their h
        remainder = sympy.Poly(sympy.rem(f.subs(H, h_wrong), g, X), X)
        k_wrong = sympy.solve(remainder.coeff_monomial(X), K)[0]
        value = f.subs({H: h_wrong, K: k_wrong}).subs(X, n)
        reply = {"a": str(h_wrong), "b": str(k_wrong), "c": str(value)}
        if 0 in (h_wrong, k_wrong, value):
            continue
        if len({reply["a"], reply["b"], reply["c"]}) < 3:
            continue
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
