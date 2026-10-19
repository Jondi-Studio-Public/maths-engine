"""Tests for `discrete-unknown-from-expectation`: tiers (`r3a_discrete_unknown`) and the exam chain
(`exd_x_r3a_discrete_unknown`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import re
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "discrete-unknown-from-expectation"
TOPIC = "r3a_discrete_unknown"
EXAM = "exd_x_r3a_discrete_unknown"


def _frac(text):
    return Fraction(text)


def _cells(text):
    """[(value text, cell text)] from 'P(X = v) = cell' pairs in the stem."""
    return re.findall(r"P\(X = (-?\w+)\) = ([^,]+?)(?=, P\(X|, and |\. |\.$| and E)", text)


def _mean_given(text):
    m = re.search(r"E\(X\) = (-?\d+(?:/\d+)?)", text)
    return Fraction(m.group(1)) if m else None


def _solve_pair(values, cells, mean):
    """Probabilities a and b from 'sum = 1' and 'E(X) = mean', by elimination."""
    known_p = sum(_frac(c) for c in cells if c not in ("a", "b"))
    known_e = sum(Fraction(v) * _frac(c) for v, c in zip(values, cells) if c not in ("a", "b"))
    va = Fraction(values[cells.index("a")])
    vb = Fraction(values[cells.index("b")])
    s, e = 1 - known_p, mean - known_e
    a = (e - vb * s) / (va - vb)
    return a, s - a


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_and_medium_keys_from_the_stem():
    bad = []
    for tier in ("easy", "medium"):
        for q in sup.tier_questions(TOPIC, tier, 150):
            pairs = _cells(q.text)
            values = [Fraction(v) for v, _ in pairs]
            cells = [c for _, c in pairs]
            if any(c.endswith("k") for c in cells):
                coeffs = [int(c[:-1] or 1) for c in cells]
                probs = [Fraction(c, sum(coeffs)) for c in coeffs]
                unknown = Fraction(1, sum(coeffs))
            else:
                known = sum(_frac(c) for c in cells if c != "p")
                unknown = 1 - known
                probs = [unknown if c == "p" else _frac(c) for c in cells]
            want = unknown if tier == "easy" else sum(v * p for v, p in zip(values, probs))
            if _frac(q.answer_text) != want:
                bad.append((tier, q.text, q.answer_text, str(want)))
    assert bad == [], bad[:2]


def test_hard_keys_from_the_stem():
    bad = []
    for q in sup.tier_questions(TOPIC, "hard", 200):
        pairs = re.findall(r"P\(X = (-?\w+)\) = (-?[\w/]+)", q.text)
        mean = _mean_given(q.text)
        if "Find x" in q.text:
            known = sum(Fraction(v) * _frac(c) for v, c in pairs if v != "x")
            px = next(_frac(c) for v, c in pairs if v == "x")
            want = (mean - known) / px
        else:
            values = [v for v, _ in pairs]
            cells = [c for _, c in pairs]
            a, b = _solve_pair(values, cells, mean)
            probs = [a if c == "a" else b if c == "b" else _frac(c) for c in cells]
            if q.text.rstrip().endswith("Find a."):
                want = a
            elif q.text.rstrip().endswith("Find b."):
                want = b
            else:
                want = sum(Fraction(v) ** 2 * p for v, p in zip(values, probs)) - mean * mean
        if _frac(q.answer_text) != want:
            bad.append((q.text, q.answer_text, str(want)))
    assert bad == [], bad[:2]


def test_exam_shape():
    sup.check_marks(EXAM, [3, 1, 3, 2], 40)
    assert all(
        [p.follow_through is not None for p in x.parts] == [False, True, True, True] for x in sup.questions(EXAM, 40)
    )
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        pairs = re.findall(r"P\(X = (-?\w+)\) = (-?[\w/]+)", q.text)
        values = [v for v, _ in pairs]
        cells = [c for _, c in pairs]
        mean = _mean_given(q.text)
        a, b = _solve_pair(values, cells, mean)
        probs = [a if c == "a" else b if c == "b" else _frac(c) for c in cells]
        assert sum(probs) == 1 and all(p > 0 for p in probs)
        second = sum(Fraction(v) ** 2 * p for v, p in zip(values, probs))
        want = {"a": a, "b": b, "c": second - mean * mean, "d": 2 * second - 3}
        got = {k: _frac(v) for k, v in sup.canonical(q).items()}
        if got != want:
            bad.append((q.text, {k: str(v) for k, v in got.items()}, {k: str(v) for k, v in want.items()}))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_a():
    def replies(q):
        pairs = re.findall(r"P\(X = (-?\w+)\) = (-?[\w/]+)", q.text)
        values = [Fraction(v) for v, _ in pairs]
        cells = [c for _, c in pairs]
        mean = _mean_given(q.text)
        wrong = Fraction(q.parts[0].distractors[0])
        known_p = sum(_frac(c) for c in cells if c not in ("a", "b"))
        b = 1 - known_p - wrong
        probs = [wrong if c == "a" else b if c == "b" else _frac(c) for c in cells]
        second = sum(v * v * p for v, p in zip(values, probs))
        var = second - mean * mean
        y = 2 * (var + mean * mean) - 3
        text = lambda f: f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)  # noqa: E731
        return {"a": text(wrong), "b": text(b), "c": text(var), "d": text(y)}, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, EXAM, replies, at_least=10)
