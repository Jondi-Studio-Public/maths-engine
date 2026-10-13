"""Exam source of `autonomous-equilibrium-stability`: a harvested population's two equilibria."""

from __future__ import annotations

import re

import sympy
from sympy import Symbol
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)

from mathgen.engine import grade
from tests import exd_x_sc1_support as sup

SKILL_ID = "autonomous-equilibrium-stability"
TOPIC = "exd_x_sc1_harvested_equilibria"
P = Symbol("P")
_T = standard_transformations + (implicit_multiplication_application, rationalize)
_MODEL = re.compile(r"f\(P\) = (.*?) and t is in years")


def model(q):
    return parse_expr(_MODEL.search(q.stem).group(1).replace("^", "**"), transformations=_T, local_dict={"P": P})


def oracle(q):
    f = model(q)
    roots = sorted(sympy.solve(f, P))
    slope = sympy.diff(f, P).subs(P, roots[1])
    assert slope < 0 < sympy.diff(f, P).subs(P, roots[0])  # larger equilibrium stable, smaller unstable
    return {"a": str(roots[1]), "b": f"{float(slope):.2f}", "c": str(roots[0])}


def test_wiring_shapes_and_text():
    sup.check_wiring(SKILL_ID, TOPIC)
    sup.check_shapes(SKILL_ID)
    sup.check_no_text_defects(SKILL_ID)


def test_shape_of_the_question():
    for seed in range(60):
        q = sup.exam(SKILL_ID, seed)
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        assert {p.label: p.answer_text for p in q.parts} == oracle(q), (seed, q.text)


def test_grading_and_tags():
    sup.check_grading(SKILL_ID)
    sup.check_tags(SKILL_ID)


def test_follow_through_from_a_wrong_equilibrium():
    """A wrong larger equilibrium, used correctly for f' and for the smaller equilibrium, still earns (b) and (c)."""
    for seed in range(100):
        q = sup.exam(SKILL_ID, seed)
        f = model(q)
        p_coef = f.coeff(P, 1)
        q_coef = -f.coeff(P, 2)
        truth = oracle(q)
        bad = int(truth["a"]) + 3
        slope = p_coef - 2 * q_coef * bad
        smaller = p_coef / q_coef - bad
        reply = {"a": str(bad), "b": f"{float(slope):.2f}", "c": str(sympy.nsimplify(smaller))}
        result = grade(sup.skill(SKILL_ID), q, reply)
        assert result.parts["a"].marks == 0
        # a smaller equilibrium that is not positive is outside the sensible range, so (c) earns nothing then
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (seed, reply)
        if smaller > 0:
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (seed, reply)


def test_answer_space_and_determinism():
    sup.check_answer_space(SKILL_ID)
    sup.check_deterministic(SKILL_ID)
