"""Tests the four-part exam chain on |pa + qb| from lengths and an angle (a . b, the squared length, the exact length, an angle)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_vector_magnitude_chain"
SKILL_ID = "vector-magnitude"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _hard(seed):
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]  # the hardest practice tier
    return make(src.topic, seed=seed, **src.params)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_FLAGS
        assert all(p.distractors and all(d in p.option_notes for d in p.distractors) for p in q.parts)


def test_prompt_shapes_beyond_the_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_oracle_agrees_with_the_rendered_prompt(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == [], bad[:2]


def test_full_marks_for_the_key(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_every_distractor_is_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_all_wrong_earns_nothing(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: f"9999{i}" for i, p in enumerate(q.parts)})
        assert result.marks == 0 and not result.correct


import math

import sympy

LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 1, 3]
FT_FLAGS = [False, True, True, True]


def _facts(q):
    la = int(re.search(r"\|a\| = (\d+)", q.stem).group(1))
    lb = int(re.search(r"\|b\| = (\d+)", q.stem).group(1))
    deg = int(re.search(r"(\d+)°", q.stem).group(1))
    expr = re.search(r"\|([^|]*b)\|", q.parts[1].prompt).group(1)
    coef = {}
    for sign, digits, name in re.findall(r"([+-]?)\s*(\d*)([ab])", expr.replace(" ", "")):
        coef[name] = (-1 if sign == "-" else 1) * (int(digits) if digits else 1)
    return la, lb, deg, coef["a"], coef["b"]


def oracle(q):
    la, lb, deg, p, qq = _facts(q)
    ab = sympy.Rational(la * lb) * sympy.cos(sympy.pi * deg / 180)
    n = p * p * la * la + qq * qq * lb * lb + 2 * p * qq * ab
    cos_phi = (p * la * la + qq * ab) / (sympy.sqrt(n) * la)
    phi = math.floor(math.degrees(math.acos(float(cos_phi))) + 0.5)
    return {"a": str(ab), "b": str(n), "c": str(sympy.sqrt(n)), "d": str(phi)}


def _from(q, a_value):
    """Parts b, c and d worked from a (possibly wrong) value of a . b, as a learner would."""
    la, lb, deg, p, qq = _facts(q)
    n = p * p * la * la + qq * qq * lb * lb + 2 * p * qq * a_value
    cos = (p * la * la + qq * a_value) / (math.sqrt(n) * la)
    return n, str(sympy.sqrt(n)), str(math.floor(math.degrees(math.acos(cos)) + 0.5))


def test_follow_through_from_a_wrong_dot_product(questions):
    checked = 0
    for q in questions[:100]:
        la, lb, deg, p, qq = _facts(q)
        usable = []
        for w in q.parts[0].distractors:
            wrong = int(w)
            n = p * p * la * la + qq * qq * lb * lb + 2 * p * qq * wrong
            if n > 0 and math.isqrt(n) ** 2 != n and abs((p * la * la + qq * wrong) / (math.sqrt(n) * la)) < 0.999:
                usable.append(wrong)
        if not usable:
            continue
        wrong = usable[0]
        n, c, d = _from(q, wrong)
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(n), "c": c, "d": d})
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True
        checked += 1
    assert checked >= 40


def test_a_wrong_squared_length_carries_into_the_exact_length(questions):
    for q in questions[:100]:
        wrong = int(q.parts[1].distractors[0])
        c = str(sympy.sqrt(wrong))
        result = grade(_SKILL, q, {**_canonical(q), "b": str(wrong), "c": c})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True


def test_the_squared_length_is_never_a_perfect_square(questions):
    assert all(math.isqrt(int(q.parts[1].answer_text)) ** 2 != int(q.parts[1].answer_text) for q in questions)
