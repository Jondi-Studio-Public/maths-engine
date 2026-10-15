"""Tests the exam chain for two colliding particles: collision time, point and the speed of B."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_r2c_collision"
SKILL_ID = "senior-consider-position-of-vectors-as-a-function-of"
HARD_TOPIC = "consider_position_of_vectors_as_a_function_of"
HARD_KWARGS = {"tier": "hard"}
t = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"t": t})


def oracle(q):
    """Collision time (the positive t where both coordinates agree), the point, and B's speed there."""
    ax, ay = (_parse(s) for s in re.search(r"r_A\(t\) = \((.+?), (.+?)\)", q.stem).groups())
    bx, by = (_parse(s) for s in re.search(r"r_B\(t\) = \((.+?), (.+?)\)", q.stem).groups())
    times = [r for r in sympy.solve(sympy.Eq(ax, bx), t) if r > 0 and sympy.simplify(ay - by).subs(t, r) == 0]
    assert len(times) == 1, (q.stem, times)
    at = times[0]
    speed = sympy.sqrt(sympy.diff(bx, t).subs(t, at) ** 2 + sympy.diff(by, t).subs(t, at) ** 2)
    return at, ax.subs(t, at), ay.subs(t, at), speed


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "number"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        at, x, y, speed = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        if not (
            sympy.Rational(a_t) == at
            and sympy.Rational(b_t) == x
            and sympy.Rational(c_t) == y
            and sympy.Rational(d_t) == speed
        ):
            bad.append((q.stem, canonical(q), (at, x, y, speed)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_time():
    """A wrong collision time used in (b), (c) and (d) earns those parts."""

    def carried(q):
        ax, ay = (_parse(s) for s in re.search(r"r_A\(t\) = \((.+?), (.+?)\)", q.stem).groups())
        bx, by = (_parse(s) for s in re.search(r"r_B\(t\) = \((.+?), (.+?)\)", q.stem).groups())
        for shift in (2, 3, 4, 5, 6):  # a wrong time whose follow-on answers are neither zero nor repeats
            wrong = sympy.Rational(q.parts[0].answer_text) + shift
            x_wrong, y_wrong = ax.subs(t, wrong), ay.subs(t, wrong)
            if len({wrong, x_wrong, y_wrong}) == 3 and 0 not in (x_wrong, y_wrong):
                break
        speed = sympy.sqrt(sympy.diff(bx, t).subs(t, wrong) ** 2 + sympy.diff(by, t).subs(t, wrong) ** 2)
        return "a", str(wrong), {"b": str(x_wrong), "c": str(y_wrong), "d": f"{sympy.N(speed, 12):.1f}"}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
