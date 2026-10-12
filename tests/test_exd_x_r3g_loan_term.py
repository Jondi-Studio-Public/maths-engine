"""Tests for the multi-part exam source `exd_x_r3g_loan_term` (flat-interest loan: term, interest, saving)."""

from __future__ import annotations

import re
from fractions import Fraction

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_loan_term"
SKILL = "senior-investigate-the-effect-of-the-interest-rate-and"


def _facts(q):
    """Everything the stem states, read back from the rendered text alone."""
    amounts = []
    for m in re.findall(r"\$(\d+)", q.text):
        if int(m) not in amounts:
            amounts.append(int(m))
    P, M1, M2 = amounts
    r = int(re.search(r"(\d+)%", q.text).group(1))
    return P, r, M1, M2


def _months(P, r, M):
    """Months to clear a flat-interest loan: n * M = P * (1 + r/100 * n/12)."""
    return Fraction(P) / (M - Fraction(P * r, 1200))


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p12b_loan_term_from_repayment", {"mode": "term_difference"})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape_marks_and_chain():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [3, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        P, r, M1, M2 = _facts(q)
        n1, n2 = _months(P, r, M1), _months(P, r, M2)
        interest1 = Fraction(P * r, 1200) * n1
        want = {"a": n1, "b": interest1, "c": interest1 - Fraction(P * r, 1200) * n2}
        got = {k: Fraction(v) for k, v in sup.canonical(q).items()}
        if got != want:
            bad.append((q.text, got, want))
    assert bad == [], bad[:2]


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_term():
    def replies(q):
        P, r, M1, M2 = _facts(q)
        I = Fraction(P * r, 1200)
        wrong_a = int(q.parts[0].distractors[0])
        b = I * wrong_a
        c = b - I * _months(P, r, M2)
        return {"a": str(wrong_a), "b": str(round(b)), "c": str(round(c))}, {"b", "c"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_a_wrong_b_does_not_cost_c():
    def replies(q):
        P, r, M1, M2 = _facts(q)
        I = Fraction(P * r, 1200)
        wrong_b = int(q.parts[1].distractors[0])
        c = wrong_b - I * _months(P, r, M2)
        return {"a": q.parts[0].answer_text, "b": str(wrong_b), "c": str(round(c))}, {"a", "c"}, {"b"}

    sup.check_follow_through(SKILL, TOPIC, replies)
