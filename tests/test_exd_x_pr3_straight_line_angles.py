"""Tests the multi-part exam source of `straight-line-angles`: solve for x on a straight line, find an angle, then bisect it.

The oracle re-derives every part's answer from the rendered stem alone, with exact arithmetic and no code shared
with the generator.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import PRESETS
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr3_straight_line_angles"
SKILL_ID = "straight-line-angles"
LABELS = ["a", "b", "c"]
MARKS = [2, 1, 2]
FOLLOWS = [False, True, True]

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def half_up(x) -> int:
    return math.floor(Fraction(x) + Fraction(1, 2))


def cents(x) -> str:
    """Two decimal places, ties away from zero, from an exact fraction."""
    scaled = Fraction(x) * 100
    return f"{half_up(scaled) / 100:.2f}"


import sympy

_X = sympy.Symbol("x")


def parse(text):
    def expr(name):
        coef, sign, const = re.search(rf"∠{name} = \((\d*)x ([+−]) (\d+)\)°", text).groups()
        return (int(coef) if coef else 1) * _X + (1 if sign == "+" else -1) * int(const)

    return dict(aoc=expr("AOC"), dob=expr("DOB"), k=int(re.search(r"∠COD = (\d+)°", text).group(1)))


def oracle(text):
    v = parse(text)
    x = sympy.solve(sympy.Eq(v["aoc"] + v["k"] + v["dob"], 180), _X)[0]
    aoc = v["aoc"].subs(_X, x)
    return {"a": str(x), "b": str(aoc), "c": str(aoc / 2 + v["k"])}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FOLLOWS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(p.distractors and len(p.distractors) <= 3 for p in q.parts)
        assert all(p.option_notes for p in q.parts)


def test_prompt_shapes_add_to_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]
    diff = src.difficulty if src.difficulty in PRESETS.get(src.topic, {}) else None
    hard = {prompt_shape(make(src.topic, seed=s, difficulty=diff, **src.params)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam source adds no prompt shape beyond the hard tier"


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.seed, q.stem) for q in questions if oracle(q.stem) != _canonical(q)]
    assert bad == []


def test_distractors_rejected(questions):
    for q in questions[:120]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.seed, p.label, d)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {label: "999999" for label in LABELS})
        assert result.marks == 0 and not result.correct


def test_follow_through_from_wrong_x(questions):
    checked = 0
    for q in questions[:150]:
        v = parse(q.stem)
        for d in q.parts[0].distractors:
            x_wrong = Fraction(d)
            aoc = half_up(v["aoc"].subs(_X, sympy.Rational(x_wrong.numerator, x_wrong.denominator)))
            if not 0 < aoc < 180:
                continue
            c = half_up(Fraction(aoc, 2) + v["k"])
            if c == aoc:
                continue  # a reply that only repeats the earlier one earns no follow-through
            result = grade(_SKILL, q, {"a": d, "b": str(aoc), "c": str(c)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 1 and result.parts["b"].follow_through
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
            assert result.marks == 3 and not result.correct
            checked += 1
            break
    assert checked >= 100


def test_follow_through_from_wrong_angle(questions):
    for q in questions[:60]:
        v = parse(q.stem)
        aoc_wrong = int(q.parts[1].distractors[0])
        c = half_up(Fraction(aoc_wrong, 2) + v["k"])
        result = grade(_SKILL, q, {**_canonical(q), "b": str(aoc_wrong), "c": str(c)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2
