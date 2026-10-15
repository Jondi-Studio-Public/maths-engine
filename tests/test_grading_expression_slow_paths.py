"""Expression replies that used to stall the grader: a probe on a branch-cut function, and undefined values.

`sqrt(acos(x)^2)` hung `grade` for over 30 s: the numeric probe substituted x = 11/5 symbolically, which makes
sympy rewrite `acos(11/5)` exactly before any number appears. `1/0`, `log(0)` and `tan(pi/2)` parse to zoo or
infinity, and were sent through `simplify` against the canonical answer for seconds before being refused.
"""

from __future__ import annotations

import time

import pytest

from mathgen import engine
from mathgen.core import symbolically_equal

SKILLS = {s.id: s for s in engine.load_skills()[1]}
LIMIT_SECONDS = 2.0


@pytest.mark.parametrize(
    ("skill_id", "seed", "reply"),
    [
        ("differentiate-trig", 1, "sqrt(acos(x)^2)"),
        ("ucv1-heat-equation-mode", 0, "1/0"),
        ("ucv1-heat-equation-mode", 0, "tan(pi/2)"),
        ("ucv1-heat-equation-mode", 0, "log(0)"),
        ("ucv1-heat-equation-mode", 0, "nan"),
        ("ucv1-heat-equation-mode", 0, "zoo"),
        ("ucv1-heat-equation-mode", 0, "oo"),
    ],
)
def test_slow_reply_is_refused_quickly(skill_id, seed, reply):
    skill = SKILLS[skill_id]
    question = engine.make_for_skill(skill, seed)
    assert engine.grade(skill, question, question.answer_text).correct
    started = time.perf_counter()
    result = engine.grade(skill, question, reply)
    assert time.perf_counter() - started < LIMIT_SECONDS
    assert not result.correct


def test_infinity_equals_only_itself():
    from sympy import S, Symbol

    x = Symbol("x")
    assert symbolically_equal(S.Infinity, S.Infinity)
    assert not symbolically_equal(S.ComplexInfinity, x)
    assert not symbolically_equal(x / x - 1 + S.NaN, 0)
