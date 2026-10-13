"""`ucv1-series-comparison-tests`: p-series, comparison, limit comparison, the integral test, absolute and conditional
convergence.

The oracle reads the series out of the question's own text and decides it by its own numerical behaviour (the exponent p
and constant c that a_n n^p settles to); it never imports the generator module.
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


def _series_term(text: str):
    """The general term of a ``Σ_(n=a)^∞ ...`` in a prompt, as a SymPy expression in n."""
    body = re.search(r"Σ_\(n=\d+\)\^∞ (.+?)(?:\. |\.$| as absolutely| where)", text).group(1)
    body = body.replace("∛n", "n^(1/3)").replace("(-1)^n", "(-1)**n")
    return parse_text(body, {"n": n})


def _behaviour(term, sign: bool = False):
    """(p, c) with |term| ~ c / n^p, read from the term at two huge n."""
    f = sympy.lambdify(n, abs(term) if not sign else term, "mpmath")
    a, b = mpmath.mpf(10) ** 12, mpmath.mpf(10) ** 14
    p = -(mpmath.log(f(b)) - mpmath.log(f(a))) / (mpmath.log(b) - mpmath.log(a))
    c = f(b) * b**p
    return Fraction(str(round(float(p), 4))).limit_denominator(12), Fraction(str(round(float(c), 4))).limit_denominator(
        12
    )


def _p_in(statement: str) -> Fraction:
    return Fraction(re.search(r"p = (\d+(?:/\d+)?)", statement).group(1))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "mcq")


def test_exactly_one_option_per_question_agrees_with_the_oracle():
    """The easy and medium oracles: the verdict and the exponent (and constant) in the true option are the series' own."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        p, c = _behaviour(_series_term(q.text))
        assert _p_in(q.answer_text) == p, (seed, q.text, q.answer_text)
        assert q.answer_text.startswith("Converges" if p > 1 else "Diverges"), (seed, q.text)
        for d in q.distractors:  # a wrong option misstates the exponent, the verdict or the rule
            assert d.startswith("Converges: the terms tend") or _p_in(d) != p or not d.startswith(q.answer_text[:9])

        q = make(TOPIC, seed=seed, difficulty="medium")
        if "direct comparison" in q.text:
            term = _series_term(q.text.replace("where a_n = ", "").replace("Σ_(n=1)^∞ a_n", "Σ_(n=1)^∞ "))
            continue
        term = _series_term(q.text)
        p, c = _behaviour(term)
        statement = re.search(r"behave like (\S+), and", q.answer_text).group(1)
        match = re.fullmatch(r"(?:(\d+)/)?(?:\((\d+)/(\d+)n[^)]*\)|(\d+)/)?.*", statement)
        assert match
        shown_p = Fraction(1)
        if "n^(" in statement:
            shown_p = Fraction(re.search(r"n\^\((\d+/\d+)\)", statement).group(1))
        elif "²" in statement:
            shown_p = Fraction(2)
        elif "³" in statement:
            shown_p = Fraction(3)
        assert shown_p == p, (seed, q.text, q.answer_text, p)
        assert q.answer_text.startswith("Converges" if p > 1 else "Diverges")


def test_direct_comparison_questions_are_sound():
    seen = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        if "direct comparison" not in q.text:
            continue
        seen += 1
        body = re.search(r"a_n = (.+?)\. Use direct", q.text).group(1)
        a = sympy.lambdify(n, parse_text(body.replace("(-1)^n", "(-1)**n"), {"n": n}), "mpmath")
        values = [a(k) for k in range(1, 400)]
        assert all(v >= 0 for v in values)
        if q.answer_text.startswith("Converges"):
            assert "0 ≤ a_n ≤ 1/n²" in q.answer_text
            assert all(a(k) <= mpmath.mpf(1) / k**2 for k in range(1, 400))
        else:
            k0 = int(re.search(r"a_n ≥ (\d)/n", q.answer_text).group(1))
            assert all(a(k) >= mpmath.mpf(k0) / k for k in range(1, 400))
    assert seen >= 8


def test_hard_classification_oracle():
    labels = {"Absolutely convergent", "Conditionally convergent", "Divergent"}
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert q.answer_text in labels and set(q.distractors) < labels
        term = _series_term(q.text)
        if "ln n" in q.text:
            expected = "Conditionally convergent" if "(ln n)²" not in q.text else "Absolutely convergent"
        else:
            magnitude = abs(term)
            f = sympy.lambdify(n, magnitude, "mpmath")
            if f(mpmath.mpf(10) ** 14) > f(mpmath.mpf(10) ** 12) / 2:  # the terms do not tend to 0
                expected = "Divergent"
            else:
                p, _c = _behaviour(term)
                expected = "Absolutely convergent" if p > 1 else "Conditionally convergent"
        assert q.answer_text == expected, (seed, q.text, q.answer_text, expected)


def test_tiers_differ_in_shape():
    pools = check_tiers_differ(TOPIC)
    assert all("Which" in t or "Decide" in t for t in pools["easy"])
    assert all(t.startswith("Classify") for t in pools["hard"])
    assert all(("limit comparison" in t or "direct comparison" in t or "justification" in t) for t in pools["medium"])


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
    assert skill.status == "active" and skill.answer == "mcq"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
