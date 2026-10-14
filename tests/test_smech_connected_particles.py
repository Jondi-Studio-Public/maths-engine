"""Tests `connected-particles-pulleys-and-towing`: oracles re-derive every answer from the question text alone."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
from smech_support import (
    assert_exam_rules,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    fixed,
    frac,
    questions,
    skill,
)

from mathgen import make
from mathgen.engine import grade

SK = skill("connected-particles-pulleys-and-towing")
SEEDS = range(60)
G = Fraction(98, 10)


def oracle(q) -> str:
    t = q.text
    if m := re.search(
        r"of mass (\d+) kg tows a \w+ of mass (\d+) kg .* driving force of (\d+) N\. Resistances of (\d+) N on .* and (\d+) N on",
        t,
    ):
        big, small, drive, r1, r2 = map(int, m.groups())
        a = Fraction(drive - r1 - r2, big + small)
        if "Find the acceleration" in t:
            return fixed(a, 2)
        assert "tension in the tow bar" in t
        return fixed(small * a + r2, 2)
    if m := re.search(r"block of mass (\d+) kg rests on a smooth .* mass of (\d+) kg hanging", t):
        m1, m2 = int(m.group(1)), int(m.group(2))
        a = m2 * G / (m1 + m2)
        return fixed(a if "acceleration" in t else m1 * a, 2)
    if m := re.search(r"Masses of (\d+) kg and (\d+) kg", t):
        light, heavy = sorted(map(int, m.groups()))
        if "acceleration" in t:
            return fixed((heavy - light) * G / (heavy + light), 2)
        return fixed(2 * light * heavy * G / (heavy + light), 2)
    raise AssertionError(f"no oracle for {t}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_grading_rules(difficulty):
    assert_single_answer_rules(SK, difficulty, SEEDS)
    for q in questions(SK, difficulty, SEEDS):
        assert_number_distractors_match_places(q)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_agrees_with_the_key(difficulty):
    for q in questions(SK, difficulty, range(200)):
        assert oracle(q) == q.answer_text, (q.text, q.answer_text)


def test_tiers_differ_in_the_decision_they_ask_for():
    easy = [q.text for q in questions(SK, "easy", range(80))]
    medium = [q.text for q in questions(SK, "medium", range(80))]
    hard = [q.text for q in questions(SK, "hard", range(80))]
    assert all("tows" in t and "acceleration" in t for t in easy)
    assert any("tow bar" in t for t in medium) and any("smooth pulley" in t for t in medium)
    assert any("both hang freely" in t for t in hard) and any("tension" in t for t in hard)
    assert not any("tows" in t for t in hard)


def test_the_pulley_tension_is_neither_weight():
    for q in questions(SK, "hard", range(120)):
        if m := re.search(r"Masses of (\d+) kg and (\d+) kg", q.text):
            if "tension" in q.text:
                light, heavy = sorted(map(int, m.groups()))
                tension = frac(q.answer_text)
                assert light * G < tension < heavy * G


def _exam():
    return [make("smech_connected_particles_exam", seed=s, difficulty="exam") for s in range(120)]


def exam_oracle(q):
    t = q.text
    m1 = int(re.search(r"block of mass (\d+) kg", t).group(1))
    mu = frac(re.search(r"friction between the block and the table is (\d+(?:\.\d+)?)", t).group(1))
    m2 = int(re.search(r"to a mass of (\d+) kg hanging", t).group(1))
    h = frac(re.search(r"fallen (\d+(?:\.\d+)?) m", q.parts[3].prompt).group(1))
    friction = mu * m1 * G
    a = (m2 * G - friction) / (m1 + m2)
    tension = m2 * (G - a)
    return {
        "a": fixed(friction, 2),
        "b": fixed(a, 2),
        "c": fixed(tension, 2),
        "d": fixed(math.sqrt(2 * a * h), 2),
    }


def test_exam_chain():
    assert_exam_rules(SK, range(60), labels="abcd", marks=[1, 3, 2, 2])
    for q in _exam():
        assert exam_oracle(q) == {p.label: p.answer_text for p in q.parts}, q.text


def test_the_string_stays_taut_and_the_block_moves():
    for q in _exam():
        a, tension = frac(q.parts[1].answer_text), frac(q.parts[2].answer_text)
        assert a > 0 and tension > 0 and tension < frac(q.parts[0].answer_text) + 100 * G


def test_exam_follows_a_wrong_friction_through():
    checked = 0
    for q in _exam()[:80]:
        t = q.text
        m1 = int(re.search(r"block of mass (\d+) kg", t).group(1))
        m2 = int(re.search(r"to a mass of (\d+) kg hanging", t).group(1))
        h = frac(re.search(r"fallen (\d+(?:\.\d+)?) m", q.parts[3].prompt).group(1))
        wrong = frac(q.parts[0].distractors[0])
        a = Fraction(fixed((m2 * G - wrong) / (m1 + m2), 2))  # the learner carries the rounded value
        if a <= 0:
            continue
        tension = m2 * (G - a)
        reply = {
            "a": q.parts[0].distractors[0],
            "b": fixed(a, 2),
            "c": fixed(tension, 2),
            "d": fixed(math.sqrt(2 * a * h), 2),
        }
        result = grade(SK, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == q.parts["abcd".index(label)].marks, (q.text, label, reply)
            assert result.parts[label].follow_through
        checked += 1
    assert checked >= 40


def test_exam_distractors_keep_the_precision_and_vary():
    for q in _exam()[:40]:
        for p in q.parts:
            assert_number_distractors_match_places(p)
    assert len({q.text for q in _exam()}) >= 60


def test_deterministic():
    for topic, d in (("smech_connected_particles", "hard"), ("smech_connected_particles_exam", "exam")):
        assert make(topic, seed=7, difficulty=d).to_dict() == make(topic, seed=7, difficulty=d).to_dict()
