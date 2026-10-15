"""Exam-tier checks for `f10-parallel-line-angle-theorems`: solve for x, angle PQR, then angle QPR."""

import re

import sympy

from mathgen import make
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_variety,
    check_wired,
    text_of,
)

TOPIC = "exd_x_sc2_parallel_triangle"
SKILL = "f10-parallel-line-angle-theorems"
X = sympy.Symbol("x")


def _expr(text: str):
    coef, sign, const = re.fullmatch(r"(\d+)x ([+−]) (\d+)", text).groups()
    return int(coef) * X + (int(const) if sign == "+" else -int(const))


def _facts(q):
    text = text_of(q)
    first = re.search(r"Angle (SPQ|TPQ) = \(([^)]*)\)°", text)
    second = re.search(r"angle PQR = \(([^)]*)\)°", text)
    e = int(re.search(r"Angle TPR = (\d+)°", text).group(1))
    relation = "alternate" if "These are alternate angles" in text else "co-interior"
    assert (first.group(1) == "SPQ") == (relation == "alternate")
    return _expr(first.group(2)), _expr(second.group(1)), e, relation


def _solve(q):
    left, right, e, relation = _facts(q)
    eq = sympy.Eq(left, right) if relation == "alternate" else sympy.Eq(left + right, 180)
    (x,) = sympy.solve(eq, X)
    return x, right, e


def oracle(q):
    x, right, e = _solve(q)
    alpha = right.subs(X, x)
    return {"a": str(x), "b": str(alpha), "c": str(180 - alpha - e)}


def wrong_chain(q):
    x, right, e = _solve(q)
    wrong_x = x + 2
    alpha = right.subs(X, wrong_x)
    return {"a": str(wrong_x), "b": str(alpha), "c": str(180 - alpha - e)}


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("p11d_parallel_line_angle_theorems", seed=s, variant="algebraic"))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_x():
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_triangle_angles_are_possible_and_both_relationships_appear():
    seen = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        x, right, e = _solve(q)
        alpha = right.subs(X, x)
        assert alpha > 0 and e > 0 and alpha + e < 180
        seen.add(_facts(q)[3])
    assert seen == {"alternate", "co-interior"}


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "901", "b": "902", "c": "903"})
