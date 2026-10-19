"""Tests for `piecewise-pdf-quantile`: tiers (`r3f_piecewise_pdf`) and the exam chain (`exd_x_r3f_piecewise_pdf`), with
every answer re-derived by numerical integration of the density read from the rendered stem."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

from tests import r3f_support as sup

SKILL = "piecewise-pdf-quantile"
TOPIC = "r3f_piecewise_pdf"
EXAM = "exd_x_r3f_piecewise_pdf"


def simpson(f, a, b, n=1500):
    h = (b - a) / (2 * n)
    total = f(a) + f(b)
    for i in range(1, 2 * n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def bisect(g, lo, hi):
    for _ in range(80):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tri_pdf(text):
    """The triangular density a stem describes: modes and ends read from its two `for ... ≤ x ≤ ...` ranges."""
    c, d = (int(v) for v in re.search(r"for 0 ≤ x ≤ (\d+), f\(x\) = .+? for \1 ≤ x ≤ (\d+)", text).groups())

    def f(x):
        if 0 <= x <= c:
            return 2 * x / (c * d)
        if c < x <= d:
            return 2 * (d - x) / (d * (d - c))
        return 0.0

    return f, c, d


def half_up(value, places):
    """`value` rounded half up, as the grader and the generator round."""
    return str(Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def cdf_of(f, lo, t):
    return simpson(f, lo, t) if t > lo else 0.0


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def _check(q, want, tol, bad):
    if abs(float(q.answer_text) - want) > tol:
        bad.append((q.text, q.answer_text, want))


def test_easy_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "easy", 150):
        t = q.text
        if "triangular" in t:
            seen.add("mean")
            c, d = (int(v) for v in re.search(r"peak at x = (\d+).*?to 0 at x = (\d+)", t).groups())
            f = lambda x: 2 * x / (c * d) if x <= c else 2 * (d - x) / (d * (d - c))  # noqa: E731
            _check(q, simpson(lambda x: x * f(x), 0, d), 0.0051, bad)
        elif "value of k" in t:
            seen.add("k")
            m = re.search(r"f\(x\) = kx/(\d+) for 0 ≤ x ≤ (\d+), f\(x\) = k\((\d+) - x\)/(\d+)", t)
            c, d = int(m.group(2)), int(m.group(3))
            # k is the height at the mode; the area of the triangle with base d is 1
            _check(q, 2 / d, 0.0005, bad)
        else:
            seen.add("prob")
            f, c, d = tri_pdf(t)
            tt = int(re.search(r"P\(X ≤ (\d+)\)", t).group(1))
            _check(q, cdf_of(f, 0, tt), 0.00006, bad)
    assert seen == {"mean", "k", "prob"}
    assert bad == [], bad[:2]


def test_medium_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "medium", 200):
        t = q.text
        if "sin(" in t:
            seen.add("sin")
            length = int(re.search(r"sin\(πx/(\d+)\)", t).group(1))
            k = 1 / simpson(lambda x: math.sin(math.pi * x / length), 0, length)
            if "value of k" in t:
                _check(q, k, 0.00006, bad)
            else:
                a = int(re.search(r"P\(X < (\d+)\)", t).group(1))
                _check(q, k * simpson(lambda x: math.sin(math.pi * x / length), 0, a), 0.00006, bad)
        else:
            f_c_d = tri_from_k_text(t)
            f, c, d = f_c_d
            if "median" in t:
                seen.add("median")
                _check(q, bisect(lambda m: cdf_of(f, 0, m) - 0.5, 0, d), 0.0051, bad)
            else:
                seen.add("across")
                t1, t2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", t).groups())
                _check(q, simpson(f, t1, t2), 0.00006, bad)
    assert seen == {"sin", "median", "across"}
    assert bad == [], bad[:2]


def tri_from_k_text(text):
    m = re.search(r"f\(x\) = kx/(\d+) for 0 ≤ x ≤ (\d+), f\(x\) = k\((\d+) - x\)/(\d+)", text)
    c, d = int(m.group(2)), int(m.group(3))

    def f(x):
        if 0 <= x <= c:
            return 2 * x / (c * d)
        if c < x <= d:
            return 2 * (d - x) / (d * (d - c))
        return 0.0

    return f, c, d


def test_hard_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "hard", 200):
        t = q.text
        if "e^(-" in t:
            lam = float(re.search(r"e\^\(-(\d+(?:\.\d+)?)x\)", t).group(1))
            b = int(re.search(r"0 ≤ x ≤ (\d+)", t).group(1))
            k = 1 / simpson(lambda x: math.exp(-lam * x), 0, b)
            dens = lambda x: k * math.exp(-lam * x)  # noqa: E731
            if "median" in t:
                seen.add("exp-median")
                _check(q, bisect(lambda m: simpson(dens, 0, m) - 0.5, 0, b), 0.0051, bad)
            elif "E(X)" in t:
                seen.add("exp-mean")
                _check(q, simpson(lambda x: x * dens(x), 0, b), 0.0051, bad)
            else:
                seen.add("exp-prob")
                a = int(re.search(r"P\(X < (\d+)\)", t).group(1))
                _check(q, simpson(dens, 0, a), 0.00006, bad)
        elif "sin(" in t:
            seen.add("sin-quantile")
            length = int(re.search(r"sin\(πx/(\d+)\)", t).group(1))
            k = 1 / simpson(lambda x: math.sin(math.pi * x / length), 0, length)
            p = float(re.search(r"= (0\.\d+),", t).group(1))
            want = bisect(lambda m: k * simpson(lambda x: math.sin(math.pi * x / length), 0, m) - p, 0, length)
            _check(q, want, 0.0051, bad)
        else:
            seen.add("tri-quantile")
            f, c, d = tri_from_k_text(t)
            p = float(re.search(r"= (0\.\d+),", t).group(1))
            _check(q, bisect(lambda m: cdf_of(f, 0, m) - p, 0, d), 0.0051, bad)
    assert seen == {"exp-median", "exp-mean", "exp-prob", "sin-quantile", "tri-quantile"}
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [2, 2, 3, 3], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in z.parts] == [False, True, True, True] for z in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_numerical_integration():
    bad = []
    for q in sup.questions(EXAM, 200):
        f, c, d = tri_from_k_text(q.text)
        got = sup.canonical(q)
        t1 = int(re.search(r"P\(X ≤ (\d+)\)", " ".join(p.prompt for p in q.parts)).group(1))
        lo, hi = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", " ".join(p.prompt for p in q.parts)).groups())
        if abs(float(got["a"]) - 2 / d) > 0.0005:
            bad.append((q.text, got))
        if abs(float(got["b"]) - simpson(f, 0, t1)) > 0.00006:
            bad.append((q.text, got))
        if abs(float(got["c"]) - bisect(lambda m: cdf_of(f, 0, m) - 0.5, 0, d)) > 0.0051:
            bad.append((q.text, got))
        if abs(float(got["d"]) - simpson(f, lo, hi)) > 0.00006:
            bad.append((q.text, got))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_k():
    def replies(q):
        c, d = tri_from_k_text(q.text)[1:]
        wrong_k = float(q.parts[0].distractors[0])
        prompts = " ".join(p.prompt for p in q.parts)
        t1 = int(re.search(r"P\(X ≤ (\d+)\)", prompts).group(1))
        lo, hi = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", prompts).groups())
        b = wrong_k * t1**2 / (2 * c)
        m = math.sqrt(c / wrong_k)
        dd = 1 - wrong_k * lo**2 / (2 * c) - wrong_k * (d - hi) ** 2 / (2 * (d - c))
        if not (0 < b < 1 and 0 < m < d and 0 < dd < 1):
            return None
        return (
            {"a": q.parts[0].distractors[0], "b": half_up(b, 4), "c": half_up(m, 2), "d": half_up(dd, 4)},
            {"b", "c", "d"},
            {"a"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, n=150, at_least=8)
