"""Tests for the exam tier `exd_x_r3h_least_squares_basis` (ucv1-function-space-approximation)."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_least_squares_basis"
SKILL = "ucv1-function-space-approximation"


def read(q):
    pairs = [(int(a), int(b)) for a, b in re.findall(r"f\((-?\d+)\) = (-?\d+)", q.stem)]
    assert len(pairs) == 3
    return pairs


def text(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def oracle(q):
    pairs = read(q)
    design = sympy.Matrix([[x**2, x**3] for x, _ in pairs])
    y = sympy.Matrix([v for _, v in pairs])
    gram = design.T * design
    c = gram.LUsolve(design.T * y)
    residual = y - design * c
    return {
        "a": str(gram.det()),
        "b": text(c[0]),
        "c": text(c[1]),
        "d": text((residual.T * residual)[0]),
    }


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_sample_points_are_distinct_and_the_fit_is_not_exact():
    for seed in range(120):
        pairs = read(sup.draw(TOPIC, seed))
        assert len({x for x, _ in pairs}) == 3
        assert Fraction(oracle(sup.draw(TOPIC, seed))["d"]) > 0


def test_explicit_follow_through_through_to_the_residual():
    skill = sup.skill(SKILL)
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        pairs = read(q)
        g11, g12 = sum(x**4 for x, _ in pairs), sum(x**5 for x, _ in pairs)
        b1 = sum(x * x * y for x, y in pairs)
        c1 = Fraction(oracle(q)["b"]) + 1
        c2 = Fraction(b1 - g11 * c1) / g12
        rss = sum((y - c1 * x * x - c2 * x**3) ** 2 for x, y in pairs)
        reply = {"a": q.parts[0].answer_text, "b": text(c1), "c": text(c2), "d": text(rss)}
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
        assert result.parts["d"].marks == q.parts[3].marks, (seed, reply)
