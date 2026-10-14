"""Tests for the exam tier `exd_x_r3h_bifurcation_threshold` (ucv1-bifurcation-sign-check)."""

from __future__ import annotations

import re

import sympy

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_bifurcation_threshold"
SKILL = "ucv1-bifurcation-sign-check"
X, R = sympy.symbols("x r")


def read(q):
    rhs = re.search(r"dx/dt = (.*?), where r", q.stem).group(1)
    f = sup.parse_math(rhs, x=X, r=R)
    r1 = sup.first_int(r"For r = (-?\d+)", q.parts[2].prompt)
    return sympy.expand(f), r1


def oracle(q):
    f, r1 = read(q)
    threshold = sympy.solve(sympy.discriminant(f, X), R)[0]
    a = f.coeff(X, 2)
    b = f.coeff(X, 1)
    roots = sympy.solve(f.subs(R, r1), X)
    return {
        "a": sup.rnd(threshold, 2),
        "b": str(-b / (2 * a)),
        "c": sup.rnd(abs(roots[0] - roots[1]), 2),
    }


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_explicit_follow_through_in_the_equilibrium_distance():
    skill = sup.skill(SKILL)
    checked = 0
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        f, r1 = read(q)
        a, p = f.coeff(X, 2), f.coeff(R, 1)
        for shift in (sympy.Rational(3, 2), -sympy.Rational(3, 2)):
            mine = sympy.Rational(sup.rnd(sympy.Rational(q.parts[0].answer_text) + shift, 2))
            radicand = -4 * a * p * (r1 - mine)
            if radicand > 0 and mine != 0:
                break
        else:
            continue
        if sup.rnd(sympy.sqrt(radicand) / abs(a), 2) == sup.rnd(mine, 2):
            continue  # a reply that repeats an earlier one is not follow-through
        reply = {"a": sup.rnd(mine, 2), "b": q.parts[1].answer_text, "c": sup.rnd(sympy.sqrt(radicand) / abs(a), 2)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
        checked += 1
    assert checked >= 20
