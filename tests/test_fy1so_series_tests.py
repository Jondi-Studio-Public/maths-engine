"""`ucv1-series-comparison-tests`: the p-series test, limit comparison, absolute and conditional convergence, the integral test.

The oracle reads each series out of the question's own text and decides it by the numerical behaviour of its terms (the
exponent p that |a_n| settles to, and the limit of a_n / b_n at a very large n); it never imports the generator module. The
exam chain is re-derived with SymPy's integrals and exact partial sums.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mpmath
import pytest
import sympy

from mathgen import make
from tests.fy1so_support import (
    SEEDS,
    check_chain,
    check_follow_through,
    check_single,
    check_tiers_differ,
    parse_text,
)

TOPIC = "fy1so_series_tests"
EXAM = "fy1so_exd_series_tests"
n = sympy.Symbol("n")
mpmath.mp.dps = 40
_ROOTS = {"√": 2, "∛": 3, "∜": 4}


def _term(text: str, k: int):
    """The general term in `text` with k put in: ``3n²/∛(n^k)`` or ``n/n^(k/3)`` as a SymPy expression in n."""
    body = re.search(r"Σ_\(n=1\)\^∞ (?:\(-1\)\^n )?(.+?)(?: converges\.|\. For how many)", text).group(1)
    body = re.sub(r"([√∛∜])\(n\^k\)", lambda m: f"n^({k}/{_ROOTS[m.group(1)]})", body)
    body = body.replace("n^(k/", f"n^({k}/")
    return parse_text(body, {"n": n})


def _exponent(text: str, k: int) -> float:
    """The exponent p of a term, from the slope of log |a_n| between two huge n (never from a formula for p)."""
    f = sympy.lambdify(n, abs(_term(text, k)), "mpmath")
    a, b = mpmath.mpf(10) ** 12, mpmath.mpf(10) ** 14
    return float(-(mpmath.log(f(b)) - mpmath.log(f(a))) / (mpmath.log(b) - mpmath.log(a)))


def test_easy_least_k_oracle():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        first = next(k for k in range(1, 60) if _exponent(q.text, k) > 1 + 1e-6)
        assert int(q.answer_text) == first, (seed, q.text, q.answer_text, first)
        assert _exponent(q.text, first - 1) <= 1 + 1e-6  # just below it the series is the harmonic series or worse


def _limit_of_ratio(text: str):
    body = re.search(r"a_n = (.+?) and b_n = (.+?)\. Find", text)
    a = parse_text(body.group(1), {"n": n})
    b = parse_text(body.group(2).replace("1/", "1/"), {"n": n})
    f = sympy.lambdify(n, a / b, "mpmath")
    return f(mpmath.mpf(10) ** 30)


def test_medium_limit_comparison_oracle():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        want = _limit_of_ratio(q.text)
        got = mpmath.mpf(Fraction(q.answer_text).numerator) / Fraction(q.answer_text).denominator
        assert abs(got - want) < mpmath.mpf(10) ** -12, (seed, q.text, q.answer_text, want)


def test_hard_counts_oracle():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        upper = int(re.search(r"to (\d+)", q.text).group(1))
        question = q.text.split("For how many")[1]
        ask = "conditional" if "conditionally" in question else "absolute" if "absolutely" in question else "divergent"
        counts = {"absolute": 0, "conditional": 0, "divergent": 0}
        for k in range(1, upper + 1):
            p = _exponent(q.text, k)
            counts["divergent" if p < 1e-6 else "conditional" if p < 1 + 1e-6 else "absolute"] += 1
        assert int(q.answer_text) == counts[ask], (seed, q.text, q.answer_text, counts)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "fraction")


def test_tiers_differ_in_shape():
    pools = check_tiers_differ(TOPIC)
    assert all(t.startswith("Find the smallest positive integer") for t in pools["easy"])
    assert all("lim" in t for t in pools["medium"])
    assert all(t.startswith("For each integer") for t in pools["hard"])


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction"] * 4, [2, 1, 2, 2])
    check_follow_through(EXAM)


def test_exam_oracle():
    k = sympy.Symbol("x")
    for seed in SEEDS:
        q = make(EXAM, seed=seed, difficulty="exam")
        match = re.search(r"a_n = (\d+)/(?:\(n \+ (\d+)\)|n)([²³⁴])", q.stem)
        c = int(match.group(1))
        b = int(match.group(2) or 0)
        power = {"²": 2, "³": 3, "⁴": 4}[match.group(3)]
        f = sympy.Rational(c) / (k + b) ** power
        a, bb, cc, dd = q.parts
        integral = sympy.integrate(f, (k, 1, sympy.oo))
        a1 = f.subs(k, 1)
        s3 = sum(f.subs(k, j) for j in (1, 2, 3))
        tail = sympy.integrate(f, (k, 3, sympy.oo))
        assert Fraction(a.answer_text) == Fraction(int(integral.p), int(integral.q)), (seed, q.stem)
        assert Fraction(bb.answer_text) == Fraction(int((a1 + integral).p), int((a1 + integral).q))
        assert Fraction(cc.answer_text) == Fraction(int(s3.p), int(s3.q))
        assert Fraction(dd.answer_text) == Fraction(int((s3 + tail).p), int((s3 + tail).q))
        total = sum(sympy.Rational(c, (j + b) ** power) for j in range(1, 3000)) + sympy.Rational(
            c, (power - 1) * (3000 + b) ** (power - 1)
        )
        assert s3 < total <= s3 + tail < a1 + integral


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-series-comparison-tests")
    assert skill.status == "active" and skill.answer == "fraction"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
