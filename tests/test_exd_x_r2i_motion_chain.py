"""Tests for the multi-part exam source of senior-recognise-acceleration-as-the-second-derivative-of-position: the
later rest time of a cubic position, the acceleration there, the time of zero acceleration and the extreme velocity.
Keys are re-derived from the rendered stem with sympy."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.exd_x_r2i_motion_chain  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_motion_chain"
SKILL = "senior-recognise-acceleration-as-the-second-derivative-of-position"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 1, 1]
FT_LABELS = {"b", "d"}
T = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _facts(q):
    poly = re.search(r"s\(t\) = ([^m]*) metres", q.stem).group(1).replace("^", "**")
    s = parse_expr(poly, local_dict={"t": T}, transformations=_TRANSFORMS)
    v, a = sympy.diff(s, T), sympy.diff(s, T, 2)
    rest = sorted(sympy.solve(v, T))
    assert len(rest) == 2
    (zero_acc,) = sympy.solve(a, T)
    return s, v, a, rest, zero_acc


def oracle(q):
    s, v, a, rest, zero_acc = _facts(q)
    return {
        "a": str(rest[1]),
        "b": str(a.subs(T, rest[1])),
        "c": str(zero_acc),
        "d": str(v.subs(T, zero_acc)),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, FT_LABELS)


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_oracle_agrees_with_every_part():
    CHECKS["oracle_agrees"]()


def test_composed_reply_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_are_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_rest_times_are_positive_and_the_midpoint_is_between_them():
    for s_ in range(120):
        s, v, a, rest, zero_acc = _facts(exam(TOPIC, s_))
        assert 0 < rest[0] < zero_acc < rest[1]


def test_wrong_rest_time_carries_into_the_acceleration():
    def chain(q):
        s, v, a, rest, zero_acc = _facts(q)
        wrong_a = rest[1] + 1
        replies = {
            "a": str(wrong_a),
            "b": str(a.subs(T, wrong_a)),
            "c": str(zero_acc),
            "d": str(v.subs(T, zero_acc)),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_zero_acceleration_time_carries_into_the_velocity():
    def chain(q):
        s, v, a, rest, zero_acc = _facts(q)
        wrong_c = zero_acc + 1
        replies = {
            "a": str(rest[1]),
            "b": str(a.subs(T, rest[1])),
            "c": str(wrong_c),
            "d": str(v.subs(T, wrong_c)),
        }
        return "c", replies, sum(MARKS) - MARKS[2]

    follow_through_check(TOPIC, SKILL, chain)
