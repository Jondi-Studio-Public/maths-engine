"""Tests for `hypothesis-test-errors-power`: tiers (`r3f_test_errors_power`) and the exam chain
(`exd_x_r3f_test_errors_power`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re

from tests import r3f_support as sup

SKILL = "hypothesis-test-errors-power"
TOPIC = "r3f_test_errors_power"
EXAM = "exd_x_r3f_test_errors_power"


def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _setup(text):
    mu0 = float(re.search(r"H0: μ = (\d+)", text).group(1))
    sign = re.search(r"H1: μ (.) ", text).group(1)
    n = int(re.search(r"sample of (\d+)", text).group(1))
    sigma = float(re.search(r"standard deviation (\d+)", text).group(1))
    z = float(re.search(r"critical z value (?:is )?(\d+(?:\.\d+)?)", text).group(1))
    return mu0, sign, n, sigma, z


def _beta(mu0, sign, n, sigma, z, mu1):
    se = sigma / math.sqrt(n)
    if sign == ">":
        return _phi((mu0 + z * se - mu1) / se)
    if sign == "<":
        return 1 - _phi((mu0 - z * se - mu1) / se)
    return _phi((mu0 + z * se - mu1) / se) - _phi((mu0 - z * se - mu1) / se)


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "easy", 200):
        t = q.text
        if "Type I error" in t:
            seen.add("alpha")
            mu0, sign, n, sigma, _z = (
                float(re.search(r"H0: μ = (\d+)", t).group(1)),
                re.search(r"H1: μ (.) ", t).group(1),
                int(re.search(r"sample of (\d+)", t).group(1)),
                float(re.search(r"standard deviation (\d+)", t).group(1)),
                None,
            )
            cut = float(re.search(r"sample mean is (?:above|below) (\d+(?:\.\d+)?)", t).group(1))
            se = sigma / math.sqrt(n)
            want = 1 - _phi((cut - mu0) / se) if sign == ">" else _phi((cut - mu0) / se)
            tol = 0.00006
        else:
            seen.add("critical")
            mu0, sign, n, sigma, z = _setup(t)
            se = sigma / math.sqrt(n)
            want = mu0 - z * se if sign == "<" else mu0 + z * se
            tol = 0.0006
        if abs(float(q.answer_text) - want) > tol:
            bad.append((t, q.answer_text, want))
    assert seen == {"alpha", "critical"}
    assert bad == [], bad[:2]


def test_medium_keys():
    bad = []
    for q in sup.tier_questions(TOPIC, "medium", 250):
        t = q.text
        mu0, sign, n, sigma, z = _setup(t)
        mu1 = float(re.search(r"true mean is (\d+)", t).group(1))
        beta = _beta(mu0, sign, n, sigma, z, mu1)
        want = beta if "Type II error" in t else 1 - beta
        if abs(float(q.answer_text) - want) > 0.00006:
            bad.append((t, q.answer_text, want))
    assert bad == [], bad[:2]


def test_hard_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "hard", 250):
        t = q.text
        if "smallest sample size" in t:
            seen.add("size")
            mu0, sign = float(re.search(r"H0: μ = (\d+)", t).group(1)), re.search(r"H1: μ (.) ", t).group(1)
            z_alpha = float(re.search(r"critical z value (\d+(?:\.\d+)?)", t).group(1))
            z_beta = float(re.search(r"use z = (\d+(?:\.\d+)?)", t).group(1))
            sigma = float(re.search(r"standard deviation (\d+)\.", t).group(1))
            mu1 = float(re.search(r"true mean is (\d+)", t).group(1))
            delta = abs(mu1 - mu0)
            # directly: the least n whose power, computed with the stated z values, reaches the target
            want = next(m for m in range(1, 1000) if delta / (sigma / math.sqrt(m)) - z_alpha >= z_beta)
            if int(q.answer_text) != want:
                bad.append((t, q.answer_text, want))
        elif "increased to" in t:
            seen.add("quadrupled")
            mu0, sign, n, sigma, z = _setup(t)
            n2 = int(re.search(r"increased to (\d+)", t).group(1))
            mu1 = float(re.search(r"true mean is (\d+)", t).group(1))
            want = 1 - _beta(mu0, sign, n2, sigma, z, mu1)
            if n2 != 4 * n or abs(float(q.answer_text) - want) > 0.00006:
                bad.append((t, q.answer_text, want))
        else:
            seen.add("two-sided")
            mu0, sign, n, sigma, z = _setup(t)
            mu1 = float(re.search(r"true mean is (\d+)", t).group(1))
            want = _beta(mu0, sign, n, sigma, z, mu1)
            if sign != "≠" or abs(float(q.answer_text) - want) > 0.00006:
                bad.append((t, q.answer_text, want))
    assert seen == {"size", "quadrupled", "two-sided"}
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [2, 3, 1, 4], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in z.parts] == [False, True, True, False] for z in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation():
    bad = []
    for q in sup.questions(EXAM, 250):
        mu0, sign, n, sigma, z = _setup(q.text)
        prompts = " ".join(p.prompt for p in q.parts)
        mu1 = float(re.search(r"true mean is (\d+)", prompts).group(1))
        n2 = int(re.search(r"increased to (\d+)", prompts).group(1))
        se = sigma / math.sqrt(n)
        beta = _beta(mu0, sign, n, sigma, z, mu1)
        want = (mu0 + z * se, beta, 1 - beta, 1 - _beta(mu0, sign, n2, sigma, z, mu1))
        got = sup.canonical(q)
        for key, w in zip("abcd", want):
            if abs(float(got[key]) - w) > (0.0006 if key == "a" else 0.00006):
                bad.append((q.text, key, got[key], w))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_critical_value():
    def replies(q):
        mu0, sign, n, sigma, z = _setup(q.text)
        prompts = " ".join(p.prompt for p in q.parts)
        mu1 = float(re.search(r"true mean is (\d+)", prompts).group(1))
        se = sigma / math.sqrt(n)
        wrong_c = float(q.parts[0].distractors[0])
        beta = _phi((wrong_c - mu1) / se)
        if not 0.001 < beta < 0.999:
            return None
        return (
            {"a": q.parts[0].distractors[0], "b": f"{beta:.4f}", "c": f"{1 - beta:.4f}", "d": q.parts[3].answer_text},
            {"b", "c", "d"},
            {"a"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, n=150, at_least=10)
