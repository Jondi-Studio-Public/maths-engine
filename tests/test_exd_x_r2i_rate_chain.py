"""Tests for the multi-part exam source of senior-interpret-the-derivative-as-the-instantaneous-rate-of: average
rate, end rate, the time the two agree, and the percentage by which they differ. Keys are re-derived from the
rendered stem with sympy."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.exd_x_r2i_rate_chain  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_rate_chain"
SKILL = "senior-interpret-the-derivative-as-the-instantaneous-rate-of"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 3, 2]
FT_LABELS = {"c", "d"}
T = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _round(x, places: int) -> str:
    exact = Fraction(str(sympy.N(sympy.nsimplify(x), 30)))
    quantum = Decimal(1).scaleb(-places)
    return str((Decimal(exact.numerator) / Decimal(exact.denominator)).quantize(quantum, rounding=ROUND_HALF_UP))


def _facts(q):
    poly = re.search(r"modelled by [A-Z]\(t\) = ([^,]*),", q.stem).group(1).replace("^", "**")
    f = parse_expr(poly, local_dict={"t": T}, transformations=_TRANSFORMS)
    p, qq = (int(v) for v in re.search(r"from t = (\d+) to t = (\d+)", q.stem).groups())
    avg = (f.subs(T, qq) - f.subs(T, p)) / (qq - p)
    end = sympy.diff(f, T).subs(T, qq)
    roots = [r for r in sympy.solve(sympy.diff(f, T) - avg, T) if p < r < qq]
    assert len(roots) == 1
    return f, p, qq, avg, end, roots[0]


def oracle(q):
    f, p, qq, avg, end, root = _facts(q)
    return {
        "a": str(avg),
        "b": str(end),
        "c": _round(root, 2),
        "d": _round(100 * (end - avg) / avg, 1),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, FT_LABELS)


def test_source_wired():
    CHECKS["wired"]()


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


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_average_rate_is_positive_and_exceeded_by_the_end_rate():
    for s in range(120):
        f, p, qq, avg, end, root = _facts(exam(TOPIC, s))
        assert 0 < avg < end
        assert p < root < qq


def test_wrong_average_carries_into_the_time_and_the_percentage():
    def chain(q):
        f, p, qq, avg, end, root = _facts(q)
        wrong_a = avg + 4
        disc_roots = [r for r in sympy.solve(sympy.diff(f, T) - wrong_a, T) if r.is_real and r > 0]
        # follow the one root of the wrong equation that lies inside the model's domain
        candidates = [r for r in disc_roots if r < qq + 2]
        if len(candidates) != 1:
            return None
        replies = {
            "a": str(wrong_a),
            "b": str(end),
            "c": _round(candidates[0], 2),
            "d": _round(100 * (end - wrong_a) / wrong_a, 1),
        }
        return "a", replies, MARKS[1] + MARKS[2] + MARKS[3]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_end_rate_carries_into_the_percentage():
    def chain(q):
        f, p, qq, avg, end, root = _facts(q)
        wrong_b = end + 10
        replies = {
            "a": str(avg),
            "b": str(wrong_b),
            "c": _round(root, 2),
            "d": _round(100 * (wrong_b - avg) / avg, 1),
        }
        return "b", replies, sum(MARKS) - MARKS[1]

    follow_through_check(TOPIC, SKILL, chain)
