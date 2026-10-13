"""Tests for the multi-part exam source `exd_x_r3g_dose_units` (dosage: mg per dose, per day, in g, tub left)."""

from __future__ import annotations

import re

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_dose_units"
SKILL = "senior-recognise-the-need-for-milligrams"


def _facts(q):
    r = int(re.search(r"(\d+) mg per kilogram", q.text).group(1))
    w = int(re.search(r"(\d+) kg", q.text).group(1))
    n = int(re.search(r"(\d+) doses", q.text).group(1))
    tub = int(re.search(r"(\d+) g tub", q.text).group(1))
    days = int(re.search(r"after (\d+) days", q.text).group(1))
    return r, w, n, tub, days


def _derive(q):
    r, w, n, tub, days = _facts(q)
    dose = r * w
    daily = n * dose
    return {
        "a": str(dose),
        "b": str(daily),
        "c": f"{daily / 1000:.3f}",
        "d": str(tub * 1000 - days * daily),
    }


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p19h_need_for_milligrams", {"tier": "hard"})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        if sup.canonical(q) != _derive(q):
            bad.append((q.text, sup.canonical(q), _derive(q)))
    assert bad == [], bad[:2]


def test_the_tub_is_never_overdrawn():
    for q in sup.questions(TOPIC, 300):
        assert int(sup.canonical(q)["d"]) > 0


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_dose():
    def replies(q):
        r, w, n, tub, days = _facts(q)
        dose = int(q.parts[0].distractors[0])
        daily = n * dose
        reply = {"a": str(dose), "b": str(daily), "c": f"{daily / 1000:.3f}", "d": str(tub * 1000 - days * daily)}
        if tub * 1000 - days * daily <= 0:
            return None
        return reply, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_daily_total():
    def replies(q):
        r, w, n, tub, days = _facts(q)
        daily = int(q.parts[1].distractors[0])
        left = tub * 1000 - days * daily
        if left <= 0:
            return None
        return {**sup.canonical(q), "b": str(daily), "c": f"{daily / 1000:.3f}", "d": str(left)}, {"c", "d"}, {"b"}

    sup.check_follow_through(SKILL, TOPIC, replies)
