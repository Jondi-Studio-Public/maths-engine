"""Tests for `expectation-variance-linear-transform`: tiers (`r3a_linear_transform`) and the exam chain
(`exd_x_r3a_linear_transform`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "expectation-variance-linear-transform"
TOPIC = "r3a_linear_transform"
EXAM = "exd_x_r3a_linear_transform"


def _table(text):
    pairs = re.findall(r"P\(X = (\d+)\) = (0\.\d+)", text)
    return [Fraction(v) for v, _ in pairs], [Fraction(p) for _, p in pairs]


def _rule(text):
    """(a, b) of 'Y = aX + b' as written in the stem."""
    m = re.search(r"Y = (-?\d*)X(?: ([+-]) (\d+))?", text)
    a = {"": 1, "-": -1}.get(m.group(1)) or int(m.group(1))
    return a, (int(m.group(3)) * (1 if m.group(2) == "+" else -1) if m.group(3) else 0)


def _moments(values, probs):
    mean = sum(v * p for v, p in zip(values, probs))
    return mean, sum(v * v * p for v, p in zip(values, probs)) - mean * mean


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
        a, b = _rule(q.text)
        spread = Fraction(re.search(r"(?:variance|standard deviation) (\d+(?:\.\d+)?)", q.text).group(1))
        mean = Fraction(re.search(r"has mean (\d+(?:\.\d+)?)", q.text).group(1))
        want = a * mean + b if "E(Y)" in q.text else a * a * spread
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_easy_asks_for_variance_from_a_variance_and_never_a_standard_deviation():
    for q in sup.tier_questions(TOPIC, "easy", 100):
        if "Var(Y)" in q.text:
            assert "variance" in q.text and "standard deviation" not in q.text


def test_medium_keys_are_unit_conversions_of_the_stated_mean_and_sd():
    bad = []
    rules = {
        "F = 1.8C + 32": (Fraction(9, 5), 32),
        "M = 10L": (10, 0),
        "S = 60T": (60, 0),
        "K = C + 273.15": (1, Fraction(27315, 100)),
        "Q = 2.2W + 4.5": (Fraction(11, 5), Fraction(9, 2)),
    }
    for q in sup.tier_questions(TOPIC, "medium", 150):
        a, b = next(v for k, v in rules.items() if k in q.text)
        mean = Fraction(re.search(r"has mean (\d+(?:\.\d+)?)", q.text).group(1))
        sd = Fraction(re.search(r"standard deviation (\d+(?:\.\d+)?)", q.text).group(1))
        if "mean of the new" in q.text:
            want = a * mean + b
        elif "variance of the new" in q.text:
            want = a * a * sd * sd
        else:
            want = a * sd
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_hard_keys_from_the_table():
    bad = []
    for q in sup.tier_questions(TOPIC, "hard", 150):
        values, probs = _table(q.text)
        mean, var = _moments(values, probs)
        if "Let Y" in q.text:
            a, _b = _rule(q.text)
            want = abs(a) * math.sqrt(float(var))
        else:
            poly = re.search(r"Find E\((-?\d+)X\^2 ([+-]) (\d*)X? ?([+-])? ?(\d+)?\)", q.text)
            a = int(poly.group(1))
            second = var + mean * mean
            terms = re.search(r"Find E\((.*?)\)\.", q.text).group(1)
            b = re.search(r"([+-]) (\d*)X(?!\^)", terms)
            bb = int(b.group(2) or 1) * (1 if b.group(1) == "+" else -1)
            c = re.search(r"([+-]) (\d+)$", terms)
            cc = int(c.group(2)) * (1 if c.group(1) == "+" else -1) if c else 0
            want = float(a * second + bb * mean + cc)
        if abs(float(q.answer_text) - float(want)) > 0.0051:
            bad.append((q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_exam_shape():
    sup.check_marks(EXAM, [1, 2, 1, 2], 40)
    assert all(
        [p.follow_through is not None for p in x.parts] == [False, True, True, True] for x in sup.questions(EXAM, 40)
    )
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        values, probs = _table(q.text)
        a, b = _rule(q.text)
        mean, var = _moments(values, probs)
        got = sup.canonical(q)
        want = {
            "a": f"{float(mean):.2f}",
            "b": f"{float(var):.2f}",
            "c": f"{float(a * mean + b):.2f}",
            "d": f"{a * math.sqrt(float(var)):.2f}",
        }
        if got != want:
            bad.append((q.text, got, want))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_variance():
    def replies(q):
        a, b = _rule(q.text)
        wrong = q.parts[1].distractors[0]
        return (
            {
                "a": q.parts[0].answer_text,
                "b": wrong,
                "c": q.parts[2].answer_text,
                "d": f"{a * math.sqrt(float(wrong)):.2f}",
            },
            {"a", "c", "d"},
            {"b"},
        )

    sup.check_follow_through(SKILL, EXAM, replies)
