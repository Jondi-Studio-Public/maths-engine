"""Tests for the exam tier `exd_x_r3h_hessian_definiteness` (ucv1-convexity-hessian-sign)."""

from __future__ import annotations

import re

import sympy

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_hessian_definiteness"
SKILL = "ucv1-convexity-hessian-sign"
X, Y = sympy.symbols("x y")


def read(q):
    rhs = re.search(r"f\(x, y\) = (.*?), with Hessian", q.stem).group(1)
    f = sup.parse_math(rhs, x=X, y=Y)
    return sympy.hessian(f, (X, Y))


def oracle(q):
    h = read(q)
    eigenvalues = sorted(float(v) for v, mult in h.eigenvals().items() for _ in range(mult))
    return {
        "a": str(h.det()),
        "b": str(h.trace()),
        "c": sup.rnd(eigenvalues[0], 2),
        "d": str(sum(1 for v in eigenvalues if v < 0)),
    }


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_all_three_verdicts_are_drawn():
    counts = {sup.draw(TOPIC, s).parts[3].answer_text for s in range(60)}
    assert counts == {"0", "1", "2"}


def test_explicit_follow_through_through_to_the_count():
    skill = sup.skill(SKILL)
    checked = 0
    for seed in range(80):
        q = sup.draw(TOPIC, seed)
        h = read(q)
        det, trace = h.det() + 4, h.trace()
        radicand = trace**2 - 4 * det
        if radicand < 0:
            continue
        low = sympy.Rational(sup.rnd((trace - sympy.sqrt(radicand)) / 2, 2))
        count = (1 - sympy.sign(low)) / 2 + (1 - sympy.sign(trace - low)) / 2
        reply = {"a": str(det), "b": str(trace), "c": sup.rnd(low, 2), "d": str(count)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
        assert result.parts["d"].marks == q.parts[3].marks, (seed, reply)
        checked += 1
    assert checked >= 30
