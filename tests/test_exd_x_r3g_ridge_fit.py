"""Tests for the multi-part exam source `exd_x_r3g_ridge_fit` (ridge: Sxx, Sxy, coefficient, prediction)."""

from __future__ import annotations

import re
from fractions import Fraction

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_ridge_fit"
SKILL = "ucv1-ridge-coefficient"


def _facts(q):
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", q.text)]
    lam = int(re.search(r"λ = (\d+)", q.text).group(1))
    new_x = int(re.search(r"when x = (-?\d+)", q.text).group(1))
    return pairs, lam, new_x


def _text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p13f_ridge_coefficient", {})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "fraction", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        pairs, lam, new_x = _facts(q)
        assert len(pairs) == 5 and sum(x for x, _ in pairs) == 0 and sum(y for _, y in pairs) == 0
        sxx = sum(x * x for x, _ in pairs)
        sxy = sum(x * y for x, y in pairs)
        beta = Fraction(sxy, sxx + lam)
        want = {"a": str(sxx), "b": str(sxy), "c": _text(beta), "d": _text(beta * new_x)}
        if sup.canonical(q) != want:
            bad.append((q.text, sup.canonical(q), want))
    assert bad == [], bad[:2]


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_wrong_sums():
    def replies(q):
        pairs, lam, new_x = _facts(q)
        sxx = int(q.parts[0].distractors[0])
        sxy = int(q.parts[1].distractors[0])
        beta = Fraction(sxy, sxx + lam)
        return {"a": str(sxx), "b": str(sxy), "c": _text(beta), "d": _text(beta * new_x)}, {"c", "d"}, {"a", "b"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_coefficient():
    def replies(q):
        pairs, lam, new_x = _facts(q)
        beta = Fraction(q.parts[2].distractors[0])
        return {**sup.canonical(q), "c": _text(beta), "d": _text(beta * new_x)}, {"d"}, {"c"}

    sup.check_follow_through(SKILL, TOPIC, replies)
