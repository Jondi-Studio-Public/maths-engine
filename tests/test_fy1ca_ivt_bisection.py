"""Tests `ucv1-ivt-bisection` (topics `fy1ca_ivt_bisection` and `fy1ca_ivt_bisection_exam`).

The oracle reads the function and the interval back out of the question TEXT and re-derives each key with SymPy:
the root is bracketed by exact rational bisection, the midpoint value is evaluated exactly, and step counts are
worked out from the stated width and tolerance.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_ivt_bisection"
EXAM = "fy1ca_ivt_bisection_exam"
SKILL = load_skill("ucv1-ivt-bisection")
x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _function(text: str):
    body = re.search(r"[fgh]\(x\) = (x³ [+-] \d*x [+-] \d+|e\^x \+ x - \d+)", text).group(1)
    return parse_expr(body.replace("³", "**3").replace("e^x", "exp(x)"), transformations=_TRANSFORMS)


def _sign(expr, value):
    v = expr.subs(x, value)
    return sympy.sign(sympy.N(v, 30))


def _bisect(expr, lo, hi, steps):
    lo, hi = sympy.Rational(lo), sympy.Rational(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sign(expr, mid) < 0 else (lo, mid)
    return lo, hi


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "number")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_key_is_the_integer_below_the_root():
    kinds = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        f = _function(q.text)
        n = int(q.answer)
        assert _sign(f, n) < 0 < _sign(f, n + 1), (seed, q.text)
        kinds["exp" if "e^x" in q.text else "cubic"] += 1
    assert kinds["exp"] >= 10 and kinds["cubic"] >= 40


def test_medium_key_is_the_kept_endpoint():
    sides = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        f = _function(q.text)
        lo, hi = (int(v) for v in re.search(r"\[(-?\d+), (-?\d+)\]", q.text).groups())
        assert _sign(f, lo) < 0 < _sign(f, hi)
        new_lo, new_hi = _bisect(f, lo, hi, 1)
        asked_left = "left endpoint" in q.text
        expected = new_lo if asked_left else new_hi
        assert sympy.Rational(q.answer_text) == expected, (seed, q.text)
        sides[(asked_left, expected in (lo, hi))] += 1
    assert len(sides) == 4, sides  # both ends asked, and both halves kept


def test_hard_key_is_three_steps_or_a_step_count():
    seen = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        if "three" in q.text:
            f = _function(q.text)
            lo, hi = (int(v) for v in re.search(r"\[(-?\d+), (-?\d+)\]", q.text).groups())
            new_lo, _ = _bisect(f, lo, hi, 3)
            assert sympy.Rational(q.answer_text) == new_lo, (seed, q.text)
            assert len(q.answer_text.split(".")[1]) == 3
            seen["steps"] += 1
        else:
            lo, hi = (int(v) for v in re.search(r"\[(-?\d+), (-?\d+)\]", q.text).groups())
            tol = Fraction(re.search(r"(?:than|less than) (0\.\d+)", q.text).group(1))
            width = hi - lo
            count = next(k for k in range(60) if Fraction(width, 2**k) < tol)
            assert int(q.answer) == count, (seed, q.text)
            seen["count"] += 1
    assert seen["steps"] >= 40 and seen["count"] >= 40, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 4, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [2, 2, 1, 1], ["number"] * 4)


def test_exam_keys_match_independent_derivation():
    for seed in range(120):
        q = _exam(seed)
        f = _function(q.stem)
        n = int(q.parts[0].answer_text)
        assert _sign(f, n) < 0 < _sign(f, n + 1)
        mid = sympy.Rational(2 * n + 1, 2)
        assert f.subs(x, mid) == sympy.Rational(q.parts[1].answer_text)
        new_lo, _ = _bisect(f, n, n + 1, 1)
        assert new_lo == sympy.Rational(q.parts[2].answer_text)
        tol = Fraction(re.search(r"narrower than (0\.\d+)", q.parts[3].prompt).group(1))
        assert (
            int(q.parts[3].answer_text) == math.ceil(math.log2(1 / tol))
            or Fraction(1, 2 ** int(q.parts[3].answer_text)) < tol
        )


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_exam_endpoint_follows_a_wrong_midpoint_value():
    credited = checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        n = int(canon["a"])
        flipped = "1.000" if sympy.Rational(canon["b"]) < 0 else "-1.000"  # a value of the opposite sign
        expected = n + sympy.Rational(1, 2) if flipped.startswith("-") else n
        text = f"{float(expected):.2f}"
        result = grade(SKILL, q, {**canon, "b": flipped, "c": text})
        checked += 1
        if flipped.startswith("-"):  # the kept half moves right: a new number, so the grader follows it through
            assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, seed
            credited += 1
    assert credited >= 15


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"
