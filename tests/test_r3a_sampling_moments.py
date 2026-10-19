"""Tests for `sampling-distribution-moments`: tiers (`r3a_sampling_moments`) and the exam chain
(`exd_x_r3a_sampling_moments`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "sampling-distribution-moments"
TOPIC = "r3a_sampling_moments"
EXAM = "exd_x_r3a_sampling_moments"
_NUM = r"(\d+(?:\.\d+)?)"


def _r2(x) -> str:
    """x to 2 places, half up, as the engine states a rounded follow-through value."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _least_n(sigma, target):
    n = 1
    while sigma * sigma / n > target * target:
        n += 1
    return n


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "easy", 150):
        sigma = Fraction(re.search(rf"population standard deviation {_NUM}", q.text).group(1))
        n = int(re.search(r"samples of (\d+)", q.text).group(1))
        want = float(sigma * sigma / n) if "variance of the sample mean" in q.text else float(sigma) / math.sqrt(n)
        if abs(float(q.answer_text) - want) > 0.0051:
            bad.append((q.text, q.answer_text, want))
    assert bad == [], bad[:2]


def test_medium_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "medium", 150):
        if "sample proportion" in q.text:
            p = float(re.search(rf"proportion of [\w ]+ is {_NUM}", q.text).group(1))
            n = int(re.search(r"sample of (\d+)", q.text).group(1))
            want = math.sqrt(p * (1 - p) / n)
            tol = 0.00006
        else:
            se = float(re.search(rf"standard deviation {_NUM}", q.text).group(1))
            n_from = int(re.search(r"samples of (\d+) observations", q.text).group(1))
            n_to = int(re.search(r"(?:increased to|cut to) (\d+)", q.text).group(1))
            want = se * math.sqrt(n_from / n_to)
            tol = 0.0051
        if abs(float(q.answer_text) - want) > tol:
            bad.append((q.text, q.answer_text, want))
    assert bad == [], bad[:2]


def test_hard_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "hard", 200):
        if "difference" in q.text:
            s1, s2 = (
                Fraction(x)
                for x in re.search(
                    rf"Site A has population standard deviation {_NUM} \w+ and Site B has {_NUM}", q.text
                ).groups()
            )
            n1, n2 = (int(x) for x in re.search(r"samples of (\d+) from A and (\d+) from B", q.text).groups())
            want = math.sqrt(float(s1 * s1 / n1 + s2 * s2 / n2))
            tol = 0.0051
        elif "sample proportion" in q.text:
            p = Fraction(re.search(rf"proportion of [\w ]+ is {_NUM}", q.text).group(1))
            target = Fraction(re.search(rf"at most {_NUM}", q.text).group(1))
            n = 1
            while p * (1 - p) / n > target * target:
                n += 1
            want, tol = n, 0
        else:
            sigma = Fraction(re.search(rf"standard deviation {_NUM} ", q.text).group(1))
            target = Fraction(re.search(rf"at most {_NUM}", q.text).group(1))
            want, tol = _least_n(sigma, target), 0
        if abs(float(q.answer_text) - float(want)) > tol:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_exam_shape():
    sup.check_marks(EXAM, [1, 1, 3, 1], 40)
    assert all(
        [p.follow_through is not None for p in x.parts] == [False, True, True, True] for x in sup.questions(EXAM, 40)
    )
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        sigma = Fraction(re.search(rf"population standard deviation {_NUM}", q.text).group(1))
        n0 = int(re.search(r"samples of (\d+)", q.text).group(1))
        target = Fraction(re.search(rf"is at most {_NUM}", q.text).group(1))
        n = _least_n(sigma, target)
        se = float(sigma) / math.sqrt(n0)
        got = sup.canonical(q)
        want = {"a": f"{se:.2f}", "b": f"{se / 2:.2f}", "c": str(n), "d": f"{float(sigma) / math.sqrt(n):.2f}"}
        if got != want:
            bad.append((q.text, got, want))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_standard_error():
    def replies(q):
        n0 = int(re.search(r"samples of (\d+)", q.text).group(1))
        target = float(re.search(rf"is at most {_NUM}", q.text).group(1))
        wrong = q.parts[0].distractors[0]
        sigma = float(wrong) * math.sqrt(n0)
        n = math.ceil((sigma / target) ** 2)
        d = float(wrong) * math.sqrt(n0) / math.sqrt(n)
        return (
            {"a": wrong, "b": _r2(Fraction(wrong) / 2), "c": str(n), "d": f"{d:.2f}"},
            {"b", "c", "d"},
            {"a"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, at_least=10)
