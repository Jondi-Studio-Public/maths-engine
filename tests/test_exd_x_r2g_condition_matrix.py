"""Tests for the multi-part exam source of ucv1-condition-number-sensitivity: infinity norms of a 2 x 2 matrix and
its inverse, the condition number and the error bound. Keys are re-derived from the rendered stem with sympy."""

from __future__ import annotations

import re

import sympy

import mathgen.generators.exd_x_r2g_condition_matrix  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_condition_matrix"
SKILL = "ucv1-condition-number-sensitivity"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 1, 2]


def _text(v) -> str:
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def _facts(q):
    entries = [
        int(x) for x in re.findall(r"\((-?\d+), (-?\d+)\)", q.stem)[0] + re.findall(r"\((-?\d+), (-?\d+)\)", q.stem)[1]
    ]
    rho = sympy.Rational(re.search(r"(\d+(?:\.\d+)?)%", q.stem).group(1)) / 100
    matrix = sympy.Matrix(2, 2, entries)
    inverse = matrix.inv()
    norm_a = max(sum(abs(v) for v in matrix.row(i)) for i in range(2))
    norm_inv = max(sum(abs(v) for v in inverse.row(i)) for i in range(2))
    return norm_a, norm_inv, rho


def oracle(q):
    norm_a, norm_inv, rho = _facts(q)
    kappa = norm_a * norm_inv
    return {"a": _text(norm_a), "b": _text(norm_inv), "c": _text(kappa), "d": _text(kappa * rho)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"c", "d"}, "p14c_condition_sensitivity")


def test_source_wired():
    CHECKS["wired"]()


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_follow_through_from_forgotten_determinant():
    """An inverse norm that forgot to divide by the determinant is carried through (c) and (d)."""

    def wrong_chain(q):
        norm_a, norm_inv, rho = _facts(q)
        det = abs(
            sympy.Matrix(2, 2, [int(x) for pair in re.findall(r"\((-?\d+), (-?\d+)\)", q.stem) for x in pair]).det()
        )
        wrong_b = norm_inv * det
        wrong_c = norm_a * wrong_b
        replies = {"a": _text(norm_a), "b": _text(wrong_b), "c": _text(wrong_c), "d": _text(wrong_c * rho)}
        return "b", replies, 1 + 1 + 2

    follow_through_check(TOPIC, SKILL, wrong_chain)


def test_follow_through_into_final_bound_only():
    def wrong_chain(q):
        norm_a, norm_inv, rho = _facts(q)
        wrong_c = norm_a + norm_inv  # added instead of multiplied
        replies = {"a": _text(norm_a), "b": _text(norm_inv), "c": _text(wrong_c), "d": _text(wrong_c * rho)}
        return "c", replies, 1 + 2 + 2

    follow_through_check(TOPIC, SKILL, wrong_chain)
