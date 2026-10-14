"""Tests `inclined-plane-friction`: oracles re-derive every answer from the question text alone."""

from __future__ import annotations

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

SK = skill("inclined-plane-friction")
SEEDS = range(60)
G = Fraction(98, 10)
NUM = r"(\d+(?:\.\d+)?)"


def slope(t):
    s = re.search(r"sin θ = (\d+)/(\d+)", t)
    sin = Fraction(int(s.group(1)), int(s.group(2)))
    # cosine from Pythagoras with exact integer triples: 3-4-5 only
    cos = {Fraction(3, 5): Fraction(4, 5), Fraction(4, 5): Fraction(3, 5)}[sin]
    assert sin * sin + cos * cos == 1
    return sin, cos


def mass(t):
    return int(re.search(r"mass (\d+) kg", t).group(1))


def mu_of(t):
    return frac(re.search(rf"coefficient of friction (?:between [^.]*? and the plane )?is {NUM}", t).group(1))


def oracle(q) -> str:
    t = q.text
    sin, cos = slope(t)
    m = mass(t)
    if "normal reaction" in t:
        return fixed(m * G * cos, 2)
    if "weight acting down the slope" in t:
        return fixed(m * G * sin, 2)
    if "greatest friction force" in t:
        return fixed(mu_of(t) * m * G * cos, 2)
    if "on the point of slipping" in t:
        return fixed(sin / cos, 2)
    if "friction force acting on the" in t:
        assert mu_of(t) > sin / cos  # the block stays at rest, so friction is the weight component only
        return fixed(m * G * sin, 2)
    if "acceleration of the" in t and "down the plane" in t:
        return fixed(G * (sin - mu_of(t) * cos), 2)
    if "pulling force" in t:
        return fixed(m * G * (sin + mu_of(t) * cos), 2)
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
    easy = " ".join(q.text for q in questions(SK, "easy", range(80)))
    medium = " ".join(q.text for q in questions(SK, "medium", range(80)))
    hard = " ".join(q.text for q in questions(SK, "hard", range(80)))
    assert "normal reaction" in easy and "weight acting down the slope" in easy and "greatest friction" in easy
    assert "released" not in easy and "constant speed" not in easy
    assert "point of slipping" in medium and "stays at rest" in medium
    assert "released from rest" in hard and "constant speed" in hard


def test_static_friction_is_not_the_limiting_value():
    for q in questions(SK, "medium", range(120)):
        if "stays at rest" in q.text:
            sin, cos = slope(q.text)
            limiting = mu_of(q.text) * mass(q.text) * G * cos
            assert frac(q.answer_text) < limiting
            assert fixed(limiting, 2) in q.distractors


def test_the_sliding_block_really_accelerates_down():
    for q in questions(SK, "hard", range(120)):
        if "acceleration" in q.text:
            sin, cos = slope(q.text)
            assert mu_of(q.text) < sin / cos and frac(q.answer_text) > 0


def _exam():
    return [make("smech_inclined_plane_exam", seed=s, difficulty="exam") for s in range(120)]


def exam_oracle(q):
    t = q.text
    sin, cos = slope(t)
    m = mass(t)
    mu = mu_of(t)
    secs = int(re.search(r"first (\d+) seconds", q.parts[3].prompt).group(1))
    normal = m * G * cos
    friction = mu * normal
    a = (m * G * sin - friction) / m
    return {
        "a": fixed(normal, 2),
        "b": fixed(friction, 2),
        "c": fixed(a, 2),
        "d": fixed(a * secs * secs / 2, 2),
    }


def test_exam_chain():
    assert_exam_rules(SK, range(60), labels="abcd", marks=[1, 1, 3, 2])
    for q in _exam():
        assert exam_oracle(q) == {p.label: p.answer_text for p in q.parts}, q.text


def test_exam_follows_a_wrong_normal_reaction_through():
    checked = 0
    for q in _exam()[:80]:
        t = q.text
        sin, cos = slope(t)
        m, mu = mass(t), mu_of(t)
        secs = int(re.search(r"first (\d+) seconds", q.parts[3].prompt).group(1))
        wrong = frac(q.parts[0].answer_text) + Fraction(31, 10)  # a slip the distractors do not cover
        raw = [mu * wrong]
        friction = Fraction(fixed(raw[0], 2))
        raw.append((m * G * sin - friction) / m)
        a = Fraction(fixed(raw[1], 2))
        raw.append(a * secs * secs / 2)
        if a <= 0 or any((r * 100) % 1 == Fraction(1, 2) for r in raw):
            continue  # a tie at the last place sits exactly on the follow-through tolerance
        reply = {
            "a": fixed(wrong, 2),
            "b": fixed(friction, 2),
            "c": fixed(a, 2),
            "d": fixed(a * secs * secs / 2, 2),
        }
        if any(reply[label] in q.parts["abcd".index(label)].distractors for label in "bcd"):
            continue  # a carried value equal to a tagged slip is ruled as that slip
        result = grade(SK, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == q.parts["abcd".index(label)].marks, (q.text, label, reply)
        checked += 1
    assert checked >= 25


def test_exam_distractors_keep_the_precision_and_text_varies():
    for q in _exam()[:40]:
        for p in q.parts:
            assert_number_distractors_match_places(p)
    assert len({q.text for q in _exam()}) >= 40


def test_deterministic():
    for topic, d in (("smech_inclined_plane", "hard"), ("smech_inclined_plane_exam", "exam")):
        assert make(topic, seed=7, difficulty=d).to_dict() == make(topic, seed=7, difficulty=d).to_dict()
