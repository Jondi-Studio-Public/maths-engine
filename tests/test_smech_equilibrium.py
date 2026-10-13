"""Tests `particle-equilibrium-concurrent-forces`: oracles re-derive every answer from the question text alone."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
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

SK = skill("particle-equilibrium-concurrent-forces")
SEEDS = range(60)
G = Fraction(98, 10)
COS = {
    Fraction(3, 5): Fraction(4, 5),
    Fraction(4, 5): Fraction(3, 5),
    Fraction(5, 13): Fraction(12, 13),
    Fraction(12, 13): Fraction(5, 13),
}


def ratio(text):
    a, b = text.split("/")
    return Fraction(int(a), int(b))


def vectors(t):
    return re.findall(r"F\d = \(([^,]+), ([^)]+)\) N", t)


def oracle(q) -> str:
    t = q.text
    if "Find k" in t:
        vs = vectors(t)
        axis = 0 if any(v[0] == "k" for v in vs) else 1
        known = sum(int(v[axis]) for v in vs if v[axis] != "k")
        return str(-known)
    if "magnitude of P" in t:
        (x1, y1), (x2, y2) = [(int(a), int(b)) for a, b in vectors(t)]
        return str(math.isqrt((x1 + x2) ** 2 + (y1 + y2) ** 2))
    if "slanting string" in t or "horizontal string" in t.split("Find")[1]:
        m = int(re.search(r"of mass (\d+) kg", t).group(1))
        sin = ratio(re.search(r"sin α = (\d+/\d+)", t).group(1))
        cos = COS[sin]
        weight = m * G
        return fixed(weight / sin if "tension in the slanting" in t else weight * cos / sin, 2)
    if "two light strings" in t:
        m = int(re.search(r"of mass (\d+) kg", t).group(1))
        a = ratio(re.search(r"sin α = (\d+/\d+)", t).group(1))
        b = ratio(re.search(r"sin β = (\d+/\d+)", t).group(1))
        t1, t2 = sympy.symbols("t1 t2")
        sol = sympy.solve(
            [
                sympy.Eq(
                    t1 * sympy.Rational(COS[a].numerator, COS[a].denominator),
                    t2 * sympy.Rational(COS[b].numerator, COS[b].denominator),
                ),
                sympy.Eq(
                    t1 * sympy.Rational(a.numerator, a.denominator) + t2 * sympy.Rational(b.numerator, b.denominator),
                    sympy.Rational(m * G.numerator, G.denominator),
                ),
            ],
            [t1, t2],
        )
        want = sol[t1] if "string PA" in t else sol[t2]
        return fixed(Fraction(int(want.p), int(want.q)), 2)
    raise AssertionError(f"no oracle for {t}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_grading_rules(difficulty):
    assert_single_answer_rules(SK, difficulty, SEEDS)
    for q in questions(SK, difficulty, SEEDS):
        assert_number_distractors_match_places(q)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_agrees_with_the_key(difficulty):
    for q in questions(SK, difficulty, range(150)):
        assert oracle(q) == q.answer_text, (q.text, q.answer_text)


def test_tiers_differ_in_the_decision_they_ask_for():
    easy = " ".join(q.text for q in questions(SK, "easy", range(80)))
    medium = " ".join(q.text for q in questions(SK, "medium", range(80)))
    hard = " ".join(q.text for q in questions(SK, "hard", range(80)))
    assert "equilibrium" in easy and "string" not in easy and "g = 9.8" not in easy
    assert "horizontal string" in medium and "two light strings" not in medium
    assert "two light strings" in hard and "PA" in hard and "PB" in hard


def test_the_symmetric_shortcut_is_always_wrong_in_the_hard_tier():
    for q in questions(SK, "hard", range(100)):
        m = int(re.search(r"of mass (\d+) kg", q.text).group(1))
        which = re.search(r"sin (α|β) = (\d+/\d+)", q.text)
        a = ratio(re.search(r"sin α = (\d+/\d+)", q.text).group(1))
        b = ratio(re.search(r"sin β = (\d+/\d+)", q.text).group(1))
        sin = a if "string PA" in q.text else b
        assert which
        assert fixed(m * G / (2 * sin), 2) != q.answer_text


def _exam():
    return [make("smech_equilibrium_exam", seed=s, difficulty="exam") for s in range(120)]


def exam_oracle(q):
    t = q.text
    m = int(re.search(r"of mass (\d+) kg", t).group(1))
    tan = ratio(re.search(r"tan θ = (\d+/\d+)", t).group(1))
    limit = int(re.search(r"exceeds (\d+) N", q.parts[3].prompt).group(1))
    hyp = {Fraction(3, 4): 5, Fraction(4, 3): 5, Fraction(5, 12): 13, Fraction(12, 5): 13}[tan]
    sin, cos = tan.numerator / Fraction(hyp), tan.denominator / Fraction(hyp)
    assert sin / cos == tan
    weight = m * G
    return {
        "a": fixed(weight, 2),
        "b": fixed(weight / cos, 2),
        "c": fixed(weight * tan, 2),
        "d": fixed(math.sqrt(limit * limit - float(weight) ** 2), 1),
    }


def test_exam_chain():
    assert_exam_rules(SK, range(60), labels="abcd", marks=[1, 2, 2, 2])
    for q in _exam():
        assert exam_oracle(q) == {p.label: p.answer_text for p in q.parts}, q.text


def test_exam_limit_is_above_the_current_tension():
    for q in _exam():
        limit = int(re.search(r"exceeds (\d+) N", q.parts[3].prompt).group(1))
        assert limit > frac(q.parts[1].answer_text)
        assert frac(q.parts[3].answer_text) > frac(q.parts[2].answer_text)


def test_exam_follows_a_wrong_weight_through():
    checked = 0
    for q in _exam()[:80]:
        tan = ratio(re.search(r"tan θ = (\d+/\d+)", q.text).group(1))
        limit = int(re.search(r"exceeds (\d+) N", q.parts[3].prompt).group(1))
        hyp = {Fraction(3, 4): 5, Fraction(4, 3): 5, Fraction(5, 12): 13, Fraction(12, 5): 13}[tan]
        cos = Fraction(tan.denominator, hyp)
        weight = frac(q.parts[0].answer_text) + Fraction(37, 10)  # a slip the distractors do not cover
        if weight >= limit:
            continue
        raw = [weight / cos, weight * tan]
        if any((r * 100) % 1 == Fraction(1, 2) for r in raw):
            continue
        hmax = math.sqrt(limit * limit - float(weight) ** 2)
        if abs((hmax * 10) % 1 - 0.5) < 0.02:
            continue
        reply = {"a": fixed(weight, 2), "b": fixed(raw[0], 2), "c": fixed(raw[1], 2), "d": fixed(hmax, 1)}
        if any(reply[k] in q.parts["abcd".index(k)].distractors for k in "bcd"):
            continue
        result = grade(SK, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == q.parts["abcd".index(label)].marks, (q.text, label, reply)
        checked += 1
    assert checked >= 30


def test_exam_distractors_keep_the_precision_and_text_varies():
    for q in _exam()[:40]:
        for p in q.parts:
            assert_number_distractors_match_places(p)
    assert len({q.text for q in _exam()}) >= 40


def test_deterministic():
    for topic, d in (("smech_equilibrium", "hard"), ("smech_equilibrium_exam", "exam")):
        assert make(topic, seed=7, difficulty=d).to_dict() == make(topic, seed=7, difficulty=d).to_dict()


def test_working_brackets_a_fraction_divisor():
    """"T = 58.8 / 3/5" reads as 58.8 / 3 / 5: a fractional divisor is bracketed."""
    for tier in ("medium", "hard", "exam"):
        for s in range(60):
            q = make("smech_equilibrium_exam" if tier == "exam" else "smech_equilibrium", seed=s, difficulty=tier)
            steps = list(q.working or [])
            for part in q.parts or []:
                steps += list(part.working or [])
            assert not re.search(r"/ \d+/\d+", " ".join(steps)), steps
