"""Exam source of `combine-like-terms`: collect the like terms in a perimeter, then substitute."""

from __future__ import annotations

import re

import sympy
from sympy import Symbol
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import exd_x_sc1_support as sup

SKILL_ID = "combine-like-terms"
TOPIC = "exd_x_sc1_perimeter_collect"
x, y = Symbol("x"), Symbol("y")
_T = standard_transformations + (implicit_multiplication_application,)
_SIDES = re.compile(r"side lengths (.*?), all in metres")


def sides(q):
    text = _SIDES.search(q.stem).group(1)
    pieces = re.split(r", | and ", text)
    return [parse_expr(p, transformations=_T, local_dict={"x": x, "y": y}) for p in pieces]


def oracle(q):
    perimeter = sympy.expand(sum(sides(q)))
    cx, cy = (int(v) for v in re.search(r"when x = (\d+) and y = (\d+)", q.parts[3].prompt).groups())
    return {
        "a": int(perimeter.coeff(x, 1)),
        "b": int(perimeter.coeff(y, 1)),
        "c": int(perimeter.subs({x: 0, y: 0})),
        "d": int(perimeter.subs({x: cx, y: cy})),
    }


def test_wiring_shapes_and_text():
    sup.check_wiring(SKILL_ID, TOPIC)
    sup.check_shapes(SKILL_ID)
    sup.check_no_text_defects(SKILL_ID)


def test_shape_of_the_question():
    for seed in range(60):
        q = sup.exam(SKILL_ID, seed)
        assert [p.marks for p in q.parts] == [1, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        assert {p.label: int(p.answer_text) for p in q.parts} == oracle(q), (seed, q.text)
        assert all(s.subs({x: 1, y: 1}) is not None for s in sides(q))


def test_grading_and_tags():
    sup.check_grading(SKILL_ID)
    sup.check_tags(SKILL_ID)


def test_follow_through_from_a_wrong_coefficient():
    """A wrong coefficient of x, used correctly in the substitution, still earns (d)."""
    checked = 0
    for seed in range(150):
        q = sup.exam(SKILL_ID, seed)
        truth = oracle(q)
        cx, cy = (int(v) for v in re.search(r"when x = (\d+) and y = (\d+)", q.parts[3].prompt).groups())
        bad_a = truth["a"] + 2
        reply_d = bad_a * cx + truth["b"] * cy + truth["c"]
        if len({bad_a, truth["b"], truth["c"], reply_d}) < 4:
            continue
        result = grade(
            sup.skill(SKILL_ID), q, {"a": str(bad_a), "b": str(truth["b"]), "c": str(truth["c"]), "d": str(reply_d)}
        )
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 1
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (seed, bad_a, reply_d)
        checked += 1
    assert checked >= 100


def test_every_side_is_positive_at_the_given_values():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        cx, cy = (int(v) for v in re.search(r"when x = (\d+) and y = (\d+)", q.parts[3].prompt).groups())
        assert all(s.subs({x: cx, y: cy}) > 0 for s in sides(q)), (seed, q.stem)
