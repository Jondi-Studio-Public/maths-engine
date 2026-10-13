"""Tests for the multi-part exam source `exd_x_r3g_recurrence_fit` (recurrence from three records)."""

from __future__ import annotations

import re
from fractions import Fraction

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_recurrence_fit"
SKILL = "senior-use-first-order-linear-recurrence-relations-to-model"


def _facts(q):
    found = re.findall(r"u\(\d\) = (\d+(?:\.\d+)?)", q.text)
    if len(found) < 3:
        found = re.search(r"are ([\d.]+), ([\d.]+) and ([\d.]+)\.", q.text).groups()
    return tuple(Fraction(v) for v in found[:3])


def _text(value: Fraction, places: int) -> str:
    scaled = int(value * 10**places + (Fraction(1, 2) if value >= 0 else -Fraction(1, 2)))
    return f"{scaled / 10**places:.{places}f}"


def _derive(q):
    u0, u1, u2 = _facts(q)
    a = (u2 - u1) / (u1 - u0)
    b = u1 - a * u0
    limit = b / (1 - a)
    # step the recurrence forward from u(0) until it is within 1 of the limit
    value, n = u0, 0
    while abs(value - limit) >= 1:
        value, n = a * value + b, n + 1
        assert n < 500
    return {"a": _text(a, 1), "b": _text(b, 1), "c": _text(limit, 0), "d": str(n)}


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "pv0a_recurrence_steady_state", {"variant": "hard"})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        if sup.canonical(q) != _derive(q):
            bad.append((q.text, sup.canonical(q), _derive(q)))
    assert bad == [], bad[:2]


def test_records_satisfy_the_recurrence_and_approach_from_both_sides():
    sides = set()
    for q in sup.questions(TOPIC, 100):
        u0, u1, u2 = _facts(q)
        a, limit = Fraction(sup.canonical(q)["a"]), Fraction(sup.canonical(q)["c"])
        assert u1 - limit == a * (u0 - limit) and u2 - limit == a * (u1 - limit)
        sides.add(u0 > limit)
    assert sides == {True, False}


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_constant_a():
    def replies(q):
        u0, u1, u2 = _facts(q)
        for option in q.parts[0].distractors:
            a = Fraction(option)
            if not 0 < a < 1 or a == Fraction(sup.canonical(q)["a"]):
                continue
            b = Fraction(_text(u1 - a * u0, 1))
            limit = Fraction(_text(b / (1 - a), 0))
            if b == 0 or limit == 0:  # markers (and the engine) withhold follow-through from a zero
                continue
            return {**sup.canonical(q), "a": option, "b": _text(b, 1), "c": _text(limit, 0)}, {"b", "c"}, {"a"}
        return None

    sup.check_follow_through(SKILL, TOPIC, replies, at_least=5)


def test_follow_through_from_a_wrong_limit_into_the_time():
    import math

    def replies(q):
        u0, u1, u2 = _facts(q)
        a = Fraction(sup.canonical(q)["a"])
        for option in q.parts[2].distractors:
            limit = Fraction(option)
            if abs(u0 - limit) <= 1 or limit <= 0:
                continue
            n = math.ceil(math.log(1 / float(abs(u0 - limit))) / math.log(float(a)))
            reply = {**sup.canonical(q), "c": option, "d": str(n)}
            if n > 0 and n != int(sup.canonical(q)["d"]) and len(set(reply.values())) == 4:
                return reply, {"d"}, {"c"}
        return None

    sup.check_follow_through(SKILL, TOPIC, replies, at_least=5)
