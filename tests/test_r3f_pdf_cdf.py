"""Tests for `pdf-cdf-from-polynomial`: tiers (`r3f_pdf_cdf`) and the exam chain (`exd_x_r3f_pdf_cdf`), with the answers
re-derived by parsing the density or distribution function out of the rendered stem and integrating it with SymPy."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tests import r3f_support as sup

SKILL = "pdf-cdf-from-polynomial"
TOPIC = "r3f_pdf_cdf"
EXAM = "exd_x_r3f_pdf_cdf"

x, k, t = sympy.symbols("x k t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(text):
    return parse_expr(text.replace("−", "-"), local_dict={"x": x, "k": k}, transformations=_TRANSFORMS)


def _density(text):
    """(f as a SymPy expression in x with k eliminated, lo, hi) from a stem that states f(x) = ... for lo <= x <= hi."""
    m = re.search(r"f\(x\) = (.+?) for (\d+) ≤ x ≤ (\d+)", text)
    shape, lo, hi = _parse(m.group(1)), int(m.group(2)), int(m.group(3))
    if shape.has(k):
        const = sympy.solve(sympy.integrate(shape, (x, lo, hi)) - 1, k)[0]
        shape = shape.subs(k, const)
    return sympy.expand(shape), lo, hi


def _cdf_from_stem(text):
    m = re.search(r", F\(x\) = (.+?) for (\d+) ≤ x ≤ (\d+)", text)
    return _parse(m.group(1)), int(m.group(2)), int(m.group(3))


def _same(a, b):
    return sympy.simplify(a - b) == 0


def _frac(v):
    v = sympy.nsimplify(v)
    return Fraction(int(v.p), int(v.q))


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def _quantile(F, lo, hi, p):
    f = sympy.lambdify(x, F - p, "mpmath")
    import mpmath

    return mpmath.findroot(f, (lo, hi), solver="anderson")


def test_easy_keys_from_the_stem():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "easy", 150):
        t_ = q.text
        if "Find the cumulative distribution function" in t_:
            seen.add("cdf")
            f, lo, hi = _density(t_)
            F = sympy.integrate(f.subs(x, t), (t, lo, x))
            if not _same(_parse(q.answer_text), F):
                bad.append((t_, q.answer_text))
        elif "Find P(" in t_:
            seen.add("prob")
            F, lo, hi = _cdf_from_stem(t_)
            a, b = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", t_).groups())
            if Fraction(q.answer_text) != _frac(F.subs(x, b) - F.subs(x, a)):
                bad.append((t_, q.answer_text))
        else:
            seen.add("pdf")
            F, lo, hi = _cdf_from_stem(t_)
            if not _same(_parse(q.answer_text), sympy.diff(F, x)):
                bad.append((t_, q.answer_text))
    assert seen == {"cdf", "prob", "pdf"}
    assert bad == [], bad[:2]


def test_medium_keys_from_the_stem():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "medium", 200):
        t_ = q.text
        f, lo, hi = _density(t_)
        F = lambda v: sympy.integrate(f.subs(x, t), (t, lo, v))  # noqa: E731
        if "Find the cumulative distribution function" in t_:
            seen.add("cdf")
            if not _same(_parse(q.answer_text), F(x)):
                bad.append((t_, q.answer_text))
        elif "Find P(X >" in t_:
            seen.add("tail")
            c = int(re.search(r"P\(X > (\d+)\)", t_).group(1))
            if Fraction(q.answer_text) != _frac(1 - F(c)):
                bad.append((t_, q.answer_text))
        else:
            seen.add("conditional")
            c, a = (int(v) for v in re.search(r"P\(X < (\d+) \| X > (\d+)\)", t_).groups())
            want = _frac((F(c) - F(a)) / (1 - F(a)))
            if Fraction(q.answer_text) != want:
                bad.append((t_, q.answer_text))
    assert seen == {"cdf", "tail", "conditional"}
    assert bad == [], bad[:2]


def test_hard_keys_from_the_stem():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "hard", 150):
        t_ = q.text
        f, lo, hi = _density(t_)
        F = sympy.integrate(f.subs(x, t), (t, lo, x))
        if "median" in t_:
            seen.add("median")
            want = _quantile(F, lo, hi, sympy.Rational(1, 2))
        elif "P(X > c)" in t_:
            seen.add("percentile")
            tail = float(re.search(r"= (0\.\d+),", t_).group(1))
            want = _quantile(F, lo, hi, 1 - sympy.nsimplify(tail))
        else:
            seen.add("iqr")
            want = _quantile(F, lo, hi, sympy.Rational(3, 4)) - _quantile(F, lo, hi, sympy.Rational(1, 4))
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((t_, q.answer_text, float(want)))
    assert seen == {"median", "percentile", "iqr"}
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [2, 2, 2, 3], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in z.parts] == [False, True, True, True] for z in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation():
    bad = []
    for q in sup.questions(EXAM, 200):
        f, lo, hi = _density(q.text)
        F = sympy.integrate(f.subs(x, t), (t, lo, x))
        kk = sympy.simplify(f / (x - lo) ** sympy.degree(f, x))
        got = sup.canonical(q)
        c1, c2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", " ".join(p.prompt for p in q.parts)).groups())
        median = _quantile(F, lo, hi, sympy.Rational(1, 2))
        if Fraction(got["a"]) != _frac(kk) or not _same(_parse(got["b"]), F):
            bad.append((q.text, got))
        if Fraction(got["c"]) != _frac(F.subs(x, c2) - F.subs(x, c1)) or abs(float(got["d"]) - float(median)) > 0.0051:
            bad.append((q.text, got))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_k():
    def replies(q):
        wrong_k = Fraction(q.parts[0].distractors[0])
        f, lo, hi = _density(q.text)
        shape_degree = sympy.degree(f, x)
        n = shape_degree
        a = lo
        bfun = wrong_k * (x - a) ** (n + 1) / (n + 1)
        c1, c2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", " ".join(p.prompt for p in q.parts)).groups())
        pc = wrong_k * (Fraction(c2 - a) ** (n + 1) - Fraction(c1 - a) ** (n + 1)) / (n + 1)
        if not 0 < pc < 1:
            return None
        pd = a + float(((n + 1) / (2 * wrong_k)) ** (1 / (n + 1)))
        if not a < pd < hi:
            return None
        return (
            {
                "a": q.parts[0].distractors[0],
                "b": str(sympy.factor(bfun)).replace("**", "^"),
                "c": str(pc),
                "d": f"{pd:.2f}",
            },
            {"b", "c", "d"},
            {"a"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, n=120, at_least=10)


def test_real_grader_rejects_every_distractor_over_many_seeds():
    """The generator checks formula options by exact polynomial comparison, not by the grader; this runs the real
    grader on every option of every part, over far more seeds than the other tests use."""
    sup.check_grading(SKILL, EXAM, n=500)
    sup.check_tier_grading(SKILL, TOPIC, n=400)
