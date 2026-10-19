"""Tests for `binomial-mean-variance-sd`: tiers (`r3a_binomial_moments`) and the exam chain
(`exd_x_r3a_binomial_moments`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "binomial-mean-variance-sd"
TOPIC = "r3a_binomial_moments"
EXAM = "exd_x_r3a_binomial_moments"


def _n_p(text):
    n = int(re.search(r"\d+", text).group())
    p = Fraction(re.search(r"probability (0\.\d+)", text).group(1))
    return n, p


def _pmf(n, p, k):
    return Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k)


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_keys_match_the_formulas_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "easy", 150):
        n, p = _n_p(q.text)
        mean, var = n * p, n * p * (1 - p)
        if "mean of X" in q.text:
            want = mean
        elif "variance" in q.text:
            want = var
        else:
            want = Fraction(math.sqrt(var))
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_medium_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "medium", 150):
        n, p = _n_p(q.text)
        mean, var = n * p, n * p * (1 - p)
        sd = math.sqrt(var)
        if "E(X^2)" in q.text:
            want = float(var + mean * mean)
        elif "upper end" in q.text:
            want = float(mean) + 2 * sd
        elif "lower end" in q.text:
            want = float(mean) - 2 * sd
        elif "mean of Y" in q.text:
            want = float(n * (1 - p))
        else:
            assert "standard deviation of Y" in q.text, q.text
            want = sd
        if abs(float(q.answer_text) - want) > 0.0051:
            bad.append((q.text, q.answer_text, want))
    assert bad == [], bad[:2]


def test_hard_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "hard", 150):
        n1, n2 = (int(x) for x in re.findall(r"\b(\d+)\b", q.text)[:2])
        found = re.findall(r"probability (0\.\d+)", q.text)
        p1, p2 = Fraction(found[0]), Fraction(found[-1])
        want = math.sqrt(float(n1 * p1 * (1 - p1) + n2 * p2 * (1 - p2)))
        if abs(float(q.answer_text) - want) > 0.0051:
            bad.append((q.text, q.answer_text, want))
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [1, 1, 1, 3], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in x.parts] == [False, True, True, False] for x in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        n, p = _n_p(q.text)
        mean, var = n * p, n * p * (1 - p)
        sd = math.sqrt(var)
        lo, hi = math.ceil(float(mean) - sd), math.floor(float(mean) + sd)
        inside = sum(_pmf(n, p, k) for k in range(lo, hi + 1))
        got = sup.canonical(q)
        if (float(got["a"]), float(got["b"]), float(got["c"])) != (float(mean), round(float(var), 2), round(sd, 2)):
            bad.append((q.text, got))
        if abs(float(got["d"]) - float(inside)) > 0.00006:
            bad.append((q.text, got["d"], float(inside)))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_mean():
    def replies(q):
        n, p = _n_p(q.text)
        wrong_mean = q.parts[0].distractors[0]
        b = f"{float(Fraction(wrong_mean) * (1 - p)):.2f}"
        return (
            {"a": wrong_mean, "b": b, "c": f"{math.sqrt(float(b)):.2f}", "d": q.parts[3].answer_text},
            {"b", "c", "d"},
            {"a"},
        )

    sup.check_follow_through(SKILL, EXAM, replies)
