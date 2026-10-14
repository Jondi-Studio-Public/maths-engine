"""Tests `newtons-second-law-resultant-force`: oracles re-derive every answer from the question text alone."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
from smech_support import (
    assert_exam_rules,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    frac,
    questions,
    skill,
)

from mathgen import make
from mathgen.engine import grade

SKILL_ID = "newtons-second-law-resultant-force"
SK = skill(SKILL_ID)
SEEDS = range(60)
G = Fraction(98, 10)


def _shown(value: Fraction, places: int) -> str:
    scaled = value * 10**places
    assert scaled.denominator == 1, value
    sign = "-" if scaled < 0 else ""
    digits = str(abs(int(scaled))).rjust(places + 1, "0")
    return sign + (digits[:-places] + "." + digits[-places:] if places else digits)


def oracle(q) -> str:
    t = q.text
    if m := re.search(
        r"mass (\d+) kg moves .* forces in the direction of motion, (\d+) N and (\d+) N, and a resistance of (\d+) N", t
    ):
        mass, p1, p2, res = map(int, m.groups())
        return _shown(Fraction(p1 + p2 - res, mass), 1)
    if m := re.search(r"of mass (\d+) kg is driven .* of (\d+) N and (\d+) N .* accelerates at ([\d.]+) m/s²", t):
        mass, p1, p2 = map(int, m.groups()[:3])
        return str(p1 + p2 - mass * frac(m.group(4)))
    if m := re.search(r"forces of (\d+) N and (\d+) N .* resistance of (\d+) N\. Its acceleration is ([\d.]+) m/s²", t):
        p1, p2, res = map(int, m.groups()[:3])
        return str((p1 + p2 - res) / frac(m.group(4)))
    if m := re.search(
        r"mass (\d+) kg starts from rest .* reaches ([\d.]+) m/s after (\d+) seconds\b.*resistance of (\d+) N", t
    ):
        mass, sec, res = int(m.group(1)), int(m.group(3)), int(m.group(4))
        return str(mass * frac(m.group(2)) / sec + res)
    if m := re.search(r"mass (\d+) kg is travelling at (\d+) m/s .* after travelling (\d+) m", t):
        mass, u, d = map(int, m.groups())
        return str(Fraction(mass * u * u, 2 * d))
    if m := re.search(
        r"mass (\d+) kg is pushed from rest .* travels (\d+) m in the first (\d+) seconds\b.*resistance of (\d+) N", t
    ):
        mass, s, sec, res = map(int, m.groups())
        return str(Fraction(mass * 2 * s, sec * sec) + res)
    if m := re.search(r"load of mass (\d+) kg vertically .* upward at ([\d.]+) m/s²", t):
        return _shown(int(m.group(1)) * (G + frac(m.group(2))), 1)
    if m := re.search(r"lift of mass (\d+) kg .* downward at ([\d.]+) m/s²", t):
        return _shown(int(m.group(1)) * (G - frac(m.group(2))), 1)
    if m := re.search(r"load of mass (\d+) kg\. The tension in the cable is ([\d.]+) N", t):
        return _shown(frac(m.group(2)) / int(m.group(1)) - G, 1)
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


def test_every_variant_of_each_tier_occurs():
    asked = {
        d: {re.search(r"(Find [a-z ]+?)(,| in |\.)", q.text).group(1) for q in questions(SK, d, range(120))}
        for d in ("easy", "medium", "hard")
    }
    assert len(asked["easy"]) == 3, asked
    assert len(asked["medium"]) >= 3, asked
    assert len(asked["hard"]) == 2, asked  # the tension and the acceleration, with weight in both


def test_tiers_differ_in_shape():
    first = {d: {q.text.split(" ")[-1] for q in questions(SK, d, range(10))} for d in ("easy", "medium", "hard")}
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    medium = " ".join(q.text for q in questions(SK, "medium", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "g = 9.8" not in easy and "g = 9.8" not in medium and "g = 9.8" in hard
    assert "seconds" not in easy and ("seconds" in medium or "travelling at" in medium)
    assert first  # all three tiers generate


def test_hard_tension_is_above_or_below_the_weight_as_the_acceleration_says():
    seen = set()
    for q in questions(SK, "hard", range(120)):
        if m := re.search(r"mass (\d+) kg .*(upward|downward) at ([\d.]+)", q.text):
            weight = int(m.group(1)) * G
            tension = frac(q.answer_text)
            assert (tension > weight) == (m.group(2) == "upward")
            seen.add(m.group(2))
    assert seen == {"upward", "downward"}


def _exam():
    return [make("smech_newton_second_law_exam", seed=s, difficulty="exam") for s in range(120)]


def exam_oracle(q):
    t = q.text
    f1 = re.search(r"F1 = \((-?\d+), (-?\d+)\) N", t)
    f2 = re.search(r"F2 = \((-?\d+), (-?\d+)\) N", t)
    acc = re.search(r"acceleration is \((-?[\d.]+), (-?[\d.]+)\)", t)
    mass = int(re.search(r"(\d+) kg", t).group(1))
    px = mass * frac(acc.group(1)) - int(f1.group(1)) - int(f2.group(1))
    py = mass * frac(acc.group(2)) - int(f1.group(2)) - int(f2.group(2))
    assert px.denominator == 1 and py.denominator == 1
    mag = math.hypot(px, py)
    ang = round(math.degrees(math.atan2(py, px)))
    return {"a": str(px), "b": str(py), "c": f"{mag:.1f}", "d": str(ang)}


def test_exam_chain():
    assert_exam_rules(SK, range(60), labels="abcd", marks=[2, 1, 2, 2])
    for q in _exam():
        assert exam_oracle(q) == {p.label: p.answer_text for p in q.parts}, q.text


def test_exam_direction_needs_the_quadrant():
    quadrants = set()
    for q in _exam():
        px = int(q.parts[0].answer_text)
        quadrants.add(px > 0)
        if px < 0:  # the tempting arctangent of |y/x| must differ from the key in the second quadrant
            assert any(d != q.parts[3].answer_text for d in q.parts[3].distractors)
    assert quadrants == {True, False}


def test_exam_follows_a_wrong_component_through():
    checked = 0
    for q in _exam()[:60]:
        wrong_a = Fraction(q.parts[0].distractors[0])
        if wrong_a == 0:  # follow-through is withheld when an earlier reply is zero
            continue
        py = Fraction(q.parts[1].answer_text)
        mag = math.hypot(wrong_a, py)
        theta = math.degrees(math.atan(py / wrong_a)) + (180 if wrong_a < 0 else 0)
        if abs((mag * 10) % 1 - 0.5) < 0.02 or abs(theta % 1 - 0.5) < 0.02:
            continue
        if f"{mag:.1f}" == q.parts[2].answer_text or f"{mag:.1f}" in q.parts[2].distractors:
            continue  # the carried value coincides with the key or with a tagged slip, which the grader rules first
        reply = {"a": str(wrong_a), "b": q.parts[1].answer_text, "c": f"{mag:.1f}", "d": str(round(theta))}
        result = grade(SK, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through
        checked += 1
    assert checked >= 30


def test_exam_distractors_keep_the_precision():
    for q in _exam()[:40]:
        for p in q.parts:
            assert_number_distractors_match_places(p)


def test_deterministic_and_varied():
    for topic, d in (("smech_newton_second_law", "hard"), ("smech_newton_second_law_exam", "exam")):
        assert make(topic, seed=5, difficulty=d).to_dict() == make(topic, seed=5, difficulty=d).to_dict()
        assert len({make(topic, seed=s, difficulty=d).text for s in range(40)}) >= 30
