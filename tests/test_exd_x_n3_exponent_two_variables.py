"""Exam source of `f10-exponent-laws-integer-variable`: a two-variable index expression, then its value."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy import symbols

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_exponent_two_variables"
SKILL_ID = "f10-exponent-laws-integer-variable"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
x, y = symbols("x y")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _printed(q):
    text = re.search(r"expression (.*)\. \(a\)|expression (.*)\.$", q.stem).group(0)
    text = re.sub(r"^expression ", "", text).rstrip(".")
    text = text.replace("÷", "/").replace("×", "*").replace("^", "**")
    text = re.sub(r"(?<=[xy0-9)]) (?=[xy(])", "*", text)
    return sympy.sympify(text)


def _values(q):
    bx, by = re.search(r"x = (-?\d+) and y = (-?\d+)", q.parts[1].prompt).groups()
    return int(bx), int(by)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        simplified = sympy.simplify(_printed(q))
        key = sympy.sympify(q.parts[0].answer_text.replace("^", "**"))
        assert sympy.simplify(simplified - key) == 0, (seed, q.stem)
        bx, by = _values(q)
        value = simplified.subs({x: bx, y: by})
        assert Fraction(int(value.p), int(value.q)) == Fraction(q.parts[1].answer_text), (seed, q.text)


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert p.distractors
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_follow_through_from_wrong_form():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        wrong = q.parts[0].distractors[0]
        bx, by = _values(q)
        v = sympy.sympify(wrong.replace("^", "**")).subs({x: bx, y: by})
        b_wrong = f"{v.p}/{v.q}" if v.q != 1 else str(v.p)
        result = grade(_SKILL, q, {"a": wrong, "b": b_wrong})
        assert result.parts["a"].marks == 0
        if b_wrong != q.parts[1].answer_text:
            assert result.parts["b"].marks == result.parts["b"].max_marks, (seed, q.text, wrong, b_wrong)
            assert result.parts["b"].follow_through
            checked += 1
    assert checked >= 80
