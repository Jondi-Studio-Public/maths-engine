"""Tests for `binomial-find-n-or-p`: tiers (`r3a_binomial_find_np`) and the exam chain (`exd_x_r3a_binomial_find_np`),
with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "binomial-find-n-or-p"
TOPIC = "r3a_binomial_find_np"
EXAM = "exd_x_r3a_binomial_find_np"


def _mean_var(text):
    m = re.search(r"mean (\d+(?:\.\d+)?) and (?:variance|standard deviation) (\d+(?:\.\d+)?)", text)
    return Fraction(m.group(1)), Fraction(m.group(2))


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
        mean = Fraction(re.search(r"The mean of X is (\d+(?:\.\d+)?)", q.text).group(1))
        if q.text.endswith("Find n."):
            p = Fraction(re.search(r"probability (0\.\d+)", q.text).group(1))
            want = mean / p
        else:
            n = int(re.search(r"(\d+) (?:seeds|calls|phone screens|shots)", q.text).group(1))
            want = mean / n
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_medium_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "medium", 150):
        mean, spread = _mean_var(q.text)
        var = spread * spread if "standard deviation" in q.text else spread
        p = 1 - var / mean
        want = p if q.text.endswith("decimal places.") else mean / p
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_hard_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "hard", 200):
        numbers = [Fraction(x) for x in re.findall(r"\d+\.\d+|\d+", q.text)]
        if "least value of n" in q.text:
            p, level = numbers[0], numbers[1]
            n = 1
            while 1 - (1 - p) ** n <= level:
                n += 1
            want = n
        elif "none of" in q.text:
            n, value = int(numbers[0]), numbers[1]
            want = 1 - Fraction(round(float(value ** Fraction(1, n)) * 100), 100)
        else:
            n, value = int(numbers[0]), numbers[1]
            want = Fraction(round(float(value ** Fraction(1, n)) * 100), 100)
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_exam_shape():
    sup.check_marks(EXAM, [2, 1, 3, 2], 40)
    assert all(
        [p.follow_through is not None for p in x.parts] == [False, True, True, True] for x in sup.questions(EXAM, 40)
    )
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        mean, var = _mean_var(q.text)
        p = 1 - var / mean
        n = mean / p
        one = n * p * (1 - p) ** (int(n) - 1)
        least = 1
        while 1 - (1 - p) ** least <= Fraction(99, 100):
            least += 1
        got = sup.canonical(q)
        want = {"a": f"{float(p):.2f}", "b": str(int(n)), "c": f"{float(one):.4f}", "d": str(least)}
        if n.denominator != 1 or got != want:
            bad.append((q.text, got, want))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_p():
    def replies(q):
        mean, _var = _mean_var(q.text)
        wrong = q.parts[0].distractors[0]
        p = Fraction(wrong)
        n = round(float(mean / p))
        if n < 2:
            return None
        one = n * p * (1 - p) ** (n - 1)
        least = math.ceil(math.log(0.01) / math.log(float(1 - p)))
        return (
            {"a": wrong, "b": str(n), "c": f"{float(one):.4f}", "d": str(least)},
            {"b", "c", "d"},
            {"a"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, at_least=10)
