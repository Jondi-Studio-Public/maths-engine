"""Tests for the multi-part exam source of ucv1-completeness-cauchy-check: limit, tail spread, least N, gap.
Keys are re-derived from the rendered stem with exact fractions."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import mathgen.generators.exd_x_r2g_cauchy_tail  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_cauchy_tail"
SKILL = "ucv1-completeness-cauchy-check"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]


def _fraction_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _parse(q):
    m = re.search(r"a_n = \((-?\d+)n ([+-]) (\d+)\)/\((\d+)n\)", q.stem)
    alpha, sign, beta, gamma = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    beta *= 1 if sign == "+" else -1
    n0 = int(re.search(r"m, n >= (\d+)\.", q.text).group(1))
    eps = Fraction(re.search(r"< (\d+(?:\.\d+)?) for all", q.text).group(1))
    return alpha, beta, gamma, n0, eps


def _facts(q):
    alpha, beta, gamma, n0, eps = _parse(q)
    limit = Fraction(alpha, gamma)
    c = Fraction(abs(beta), gamma)
    n = math.floor(c / eps) + 1
    return limit, c, n0, eps, n


def oracle(q):
    limit, c, n0, eps, n = _facts(q)
    return {
        "a": _fraction_text(limit),
        "b": _fraction_text(c / n0),
        "c": str(n),
        "d": _fraction_text(c / n),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"c", "d"}, "p15b_completeness_cauchy_check")


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


def test_brute_force_least_n_is_the_key():
    """Check the epsilon-N threshold directly: the spread over indices up to a large cut-off first drops below
    epsilon at the keyed N (the gap c/N is the supremum, so the finite maximum sits just under it)."""
    from tests.exd_x_r2g_support import exam

    for s in range(40):
        q = exam(TOPIC, s)
        _limit, c, _n0, eps, n = _facts(q)
        assert c / n < eps <= c / (n - 1)


def test_follow_through_from_wrong_spread():
    """A tail spread of twice the true value is carried through (c) and (d)."""

    def wrong_chain(q):
        _limit, c, n0, eps, _n = _facts(q)
        wrong_b = 2 * c / n0
        wrong_n = math.floor(n0 * wrong_b / eps) + 1
        wrong_d = n0 * wrong_b / wrong_n
        replies = {
            "a": _fraction_text(_limit),
            "b": _fraction_text(wrong_b),
            "c": str(wrong_n),
            "d": _fraction_text(wrong_d),
        }
        return "b", replies, 4

    follow_through_check(TOPIC, SKILL, wrong_chain)
