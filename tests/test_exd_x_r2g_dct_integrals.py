"""Tests for the multi-part exam source of ucv1-dominated-convergence-check: integrals of the limit, of the smallest
dominating function and of f_n0 on a three-point space, and the n that brings the integrals close. Keys are
re-derived from the rendered stem with sympy."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy

import mathgen.generators.exd_x_r2g_dct_integrals  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_dct_integrals"
SKILL = "ucv1-dominated-convergence-check"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 1, 2]
N = sympy.Symbol("n")


def _text(v) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _facts(q):
    masses = [int(x) for x in re.findall(r"mu\(\{x\d\}\) = (\d+)", q.stem)]
    if not masses:
        masses = [int(x) for x in re.search(r"carry masses (\d+), (\d+) and (\d+)", q.stem).groups()]
    forms = [sympy.sympify(m) for m in re.findall(r"f_n\(x\d\) = (-?\d+ [+-] \d+/n)", q.stem)]
    n0 = int(re.search(r"integral of f_(\d+) with", q.text).group(1))
    eps = sympy.Rational(re.search(r"by less than (\d+(?:\.\d+)?)", q.text).group(1))
    return masses, forms, n0, eps


def _solve(q):
    masses, forms, n0, eps = _facts(q)
    assert len(masses) == 3 and len(forms) == 3, q.stem
    limits = [Fraction(int(sympy.limit(f, N, sympy.oo))) for f in forms]
    offsets = [Fraction(int(f.subs(N, 1))) - lim for f, lim in zip(forms, limits)]
    value = lambda i, k: limits[i] + offsets[i] / k  # noqa: E731
    integral_f = sum(w * lim for w, lim in zip(masses, limits))
    # the smallest dominating function: the largest |f_n(x)| over the first 200 indices and the limit
    integral_g = sum(
        w * max([abs(value(i, k)) for k in range(1, 200)] + [abs(limits[i])]) for i, w in enumerate(masses)
    )
    integral_n0 = sum(w * value(i, n0) for i, w in enumerate(masses))
    eps = Fraction(int(eps.p), int(eps.q))
    n = next(k for k in range(1, 5000) if abs(sum(w * value(i, k) for i, w in enumerate(masses)) - integral_f) < eps)
    return integral_f, integral_g, integral_n0, n


def oracle(q):
    f, g, n0, n = _solve(q)
    return {"a": _text(f), "b": _text(g), "c": _text(n0), "d": str(n)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"d"}, "p15d_dominated_convergence")


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


def test_follow_through_from_wrong_limit_integral():
    """A wrong integral of the limit changes the gap and so the n in (d); (d) is marked on the learner's own (a)."""

    def wrong_chain(q):
        integral_f, _g, integral_n0, _n = _solve(q)
        _masses, _forms, n0, eps = _facts(q)
        eps = Fraction(int(eps.p), int(eps.q))
        wrong_a = integral_f + (1 if integral_f != -1 else 2)  # never zero: markers withhold follow-through from 0
        k = math.floor(n0 * abs(integral_n0 - wrong_a) / eps) + 1
        replies = {"a": _text(wrong_a), "b": _text(_g), "c": _text(integral_n0), "d": str(k)}
        return "a", replies, 2 + 1 + 2

    follow_through_check(TOPIC, SKILL, wrong_chain)
