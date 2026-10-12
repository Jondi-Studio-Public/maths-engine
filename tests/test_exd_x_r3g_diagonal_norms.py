"""Tests for the multi-part exam source `exd_x_r3g_diagonal_norms` (norm, power, inverse, condition number)."""

from __future__ import annotations

import re
from fractions import Fraction

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_diagonal_norms"
SKILL = "ucv1-diagonal-operator-norm"


def _entry(text: str) -> Fraction:
    m = re.fullmatch(r"(-?)(\d*)[xy](?:/(\d+))?", text.strip())
    assert m, text
    return Fraction(-1 if m.group(1) else 1) * int(m.group(2) or 1) / int(m.group(3) or 1)


def _facts(q):
    m = re.search(r"T\(x, y\) = \(([^,]+), ([^)]+)\)", q.text)
    k = int(re.search(r"T\^(\d)", q.text).group(1))
    return _entry(m.group(1)), _entry(m.group(2)), k


def _text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _derive(q):
    d1, d2, k = _facts(q)
    # The Euclidean operator norm of a diagonal map is its largest |entry|; T^k and T^-1 are diagonal too.
    norm = lambda *e: max(abs(v) for v in e)  # noqa: E731
    big = norm(d1, d2)
    inverse = norm(1 / d1, 1 / d2)
    return {"a": _text(big), "b": _text(norm(d1**k, d2**k)), "c": _text(inverse), "d": _text(big * inverse)}


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p15c_diagonal_operator_norm", {"variant": "hard"})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction"] * 4
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        if sup.canonical(q) != _derive(q):
            bad.append((q.text, sup.canonical(q), _derive(q)))
    assert bad == [], bad[:2]


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_norm():
    def replies(q):
        d1, d2, k = _facts(q)
        a = Fraction(q.parts[0].distractors[0])
        c = 1 / min(abs(d1), abs(d2))
        if a <= 0:
            return None
        reply = {**sup.canonical(q), "a": _text(a), "b": _text(a**k), "d": _text(a * c)}
        if len(set(reply.values())) < 4:
            return None
        return reply, {"b", "d"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_inverse_norm():
    def replies(q):
        c = Fraction(q.parts[2].distractors[0])
        a = Fraction(q.parts[0].answer_text)
        if c <= 0:
            return None
        reply = {**sup.canonical(q), "c": _text(c), "d": _text(a * c)}
        if len(set(reply.values())) < 4:
            return None
        return reply, {"d"}, {"c"}

    sup.check_follow_through(SKILL, TOPIC, replies)
