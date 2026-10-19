"""Tests for `binomial-at-least-complement`: tiers (`r3a_binomial_at_least`) and the exam chain
(`exd_x_r3a_binomial_at_least`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "binomial-at-least-complement"
TOPIC = "r3a_binomial_at_least"
EXAM = "exd_x_r3a_binomial_at_least"
_WORDS = {"one": 1, "two": 2}


def _n_p(text):
    n = int(re.search(r"\d+", text).group())
    p = Fraction(re.search(r"probability (0\.\d+)", text).group(1))
    return n, p


def _prob(n, p, lo, hi):
    return sum(Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(max(lo, 0), min(hi, n) + 1))


def _num(word):
    return _WORDS.get(word) or int(word)


def _event_value(n, p, text):
    """The probability asked for, read off the sentence 'Find the probability that X is ...'."""
    m = re.search(r"probability that X is (.*?), correct|probability that X is (.*?)\. Give", text)
    event = (m.group(1) or m.group(2)).strip()
    if m := re.fullmatch(r"at least (\w+), given that X is at least (\w+)", event):
        k, j = _num(m.group(1)), _num(m.group(2))
        return _prob(n, p, k, n) / _prob(n, p, j, n)
    if m := re.fullmatch(r"at most (\w+), given that X is at least (\w+)", event):
        k, j = _num(m.group(1)), _num(m.group(2))
        return _prob(n, p, j, k) / _prob(n, p, j, n)
    if m := re.fullmatch(r"at least (\w+)", event):
        return _prob(n, p, _num(m.group(1)), n)
    if m := re.fullmatch(r"at most (\w+)", event):
        return _prob(n, p, 0, _num(m.group(1)))
    if m := re.fullmatch(r"more than (\w+)", event):
        return _prob(n, p, _num(m.group(1)) + 1, n)
    m = re.fullmatch(r"between (\d+) and (\d+) inclusive", event)
    return _prob(n, p, int(m.group(1)), int(m.group(2)))


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_every_tier_key_matches_independent_derivation():
    bad = []
    for tier in sup.TIERS:
        for q in sup.tier_questions(TOPIC, tier, 150):
            n, p = _n_p(q.text)
            want = _event_value(n, p, q.text)
            if abs(float(q.answer_text) - float(want)) > 0.00006:
                bad.append((tier, q.text, q.answer_text, float(want)))
    assert bad == [], bad[:2]


def test_hard_tier_is_always_conditional():
    assert all("given that" in q.text for q in sup.tier_questions(TOPIC, "hard", 60))


def test_exam_shape():
    sup.check_marks(EXAM, [2, 1, 2, 2], 40)
    assert all(
        [p.follow_through is not None for p in x.parts] == [False, True, True, True] for x in sup.questions(EXAM, 40)
    )
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        n, p = _n_p(q.text)
        k = int(re.search(r"Find P\(X <= (\d+)\)", q.text).group(1)) + 1
        days = int(re.search(r"each of (\d+) independent days", q.text).group(1))
        lower = _prob(n, p, 0, k - 1)
        upper = _prob(n, p, k, n)
        cond = upper / _prob(n, p, 1, n)
        some_day = 1 - (1 - upper) ** days
        got = {key: float(v) for key, v in sup.canonical(q).items()}
        want = {"a": lower, "b": upper, "c": cond, "d": some_day}
        if any(abs(got[key] - float(want[key])) > 0.00006 for key in want):
            bad.append((q.text, got, {key: float(v) for key, v in want.items()}))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_lower_tail():
    def replies(q):
        n, p = _n_p(q.text)
        k = int(re.search(r"Find P\(X <= (\d+)\)", q.text).group(1)) + 1
        days = int(re.search(r"each of (\d+) independent days", q.text).group(1))
        wrong = q.parts[0].distractors[0]
        b = round(1 - float(wrong), 4)
        c = b / float(_prob(n, p, 1, n))
        d = 1 - (1 - b) ** days
        return {"a": wrong, "b": f"{b:.4f}", "c": f"{c:.4f}", "d": f"{d:.4f}"}, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, EXAM, replies, at_least=10)
