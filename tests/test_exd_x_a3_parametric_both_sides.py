"""Tests for the exam source `exd_x_a3_parametric_both_sides` (variables-on-both-sides).

The oracle re-derives every key from the RENDERED PROMPT TEXT alone, with sympy / exact fractions, and never imports
the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy import Symbol, parse_expr

from mathgen import make
from mathgen.grading import GRADERS, mcq_options
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a3_parametric_both_sides"
KIND = "number"
HARD = [("linear_equation", {"kind": "both_sides_bracket"})]


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _implicit(s: str) -> str:
    """Insert explicit multiplication into school-style algebra text, e.g. '3(x + 1)' -> '3*(x + 1)'."""
    s = s.replace("^", "**")
    return re.sub(r"(?<=[0-9a-z\)])(?=[a-z(])", "*", s)


def _expr(s: str):
    names = {c: Symbol(c) for c in set(re.findall(r"[a-z]", s))}
    return parse_expr(_implicit(s), local_dict=names)


def _equation(s: str):
    left, right = s.split("=")
    return _expr(left) - _expr(right)


def _grade(q, given):
    if KIND == "mcq":
        return GRADERS["mcq"](q, mcq_options(q).index(given)).correct
    return GRADERS[KIND](q, given).correct


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam = {prompt_shape(_make(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    hard = set()
    for topic, params in HARD:
        hard |= {prompt_shape(make(topic, seed=s, difficulty="hard", **params)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_determinism():
    for seed in (0, 1, 2, 77, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


def test_registered_exam_preset():
    make(TOPIC, seed=0, difficulty="exam")


def test_source_is_wired_into_the_skill():
    from mathgen.skills import load_skills

    skill = {s.id: s for s in load_skills()[1]}["variables-on-both-sides"]
    last = skill.sources[-1]
    assert last.topic == TOPIC and last.difficulty == "exam" and list(last.stages) == ["exam"]


def test_key_and_distractors_graded_over_150_seeds():
    for seed in range(150):
        q = _make(seed)
        assert _grade(q, q.answer_text), (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not _grade_text(q, d), (seed, q.text, q.answer_text, d)


def _grade_text(q, given):
    """Grade a raw typed reply (for mcq every option is graded by its own index)."""
    if KIND == "mcq":
        return given == q.answer_text
    return GRADERS[KIND](q, given).correct


_POLY = {"equilateral triangle": 3, "square": 4, "regular pentagon": 5, "regular hexagon": 6, "regular octagon": 8}


def _oracle(text: str) -> int:
    if "perimeter" in text and "cm" in text:
        sides = sorted(((text.index(name), n) for name, n in _POLY.items() if name in text), key=lambda t: t[0])
        (m, n) = sides[0][1], sides[1][1]
        found = re.findall(r"\(x ([+-]) (\d+)\)", text)
        a, b = (int(v) * (1 if sg == "+" else -1) for sg, v in found)
        x = Symbol("x")
        sol = sympy.solve(m * (x + a) - n * (x + b), x)
        assert len(sol) == 1
        return int(m * (sol[0] + a))
    letter = re.search(r"(?:Find|What is) the value of ([a-z])", text).group(1)
    sym = Symbol(letter)
    if "same" in text:
        pieces = re.split(r" and |Both |, | are solved| have the same", text)
        eqs = [p for p in pieces if "=" in p]
        assert len(eqs) == 2
        plain = next(e for e in eqs if letter not in e)
        param = next(e for e in eqs if letter in e)
        x = Symbol("x")
        x0 = sympy.solve(_equation(plain.replace("The equations ", "")), x)
        assert len(x0) == 1
        k = sympy.solve(_equation(param.replace("The equations ", "")).subs(x, x0[0]), sym)
        assert len(k) == 1
        return int(k[0])
    m = re.search(r"x = (-?\d+)", text)
    x0 = int(m.group(1))
    eq = re.search(r"(?:solution of |The equation )([^.]*?)(?: has the solution|\.)", text).group(1)
    k = sympy.solve(_equation(eq).subs(Symbol("x"), x0), sym)
    assert len(k) == 1
    return int(k[0])


def test_key_rederived_from_rendered_prompt_over_400_seeds():
    for seed in range(400):
        q = _make(seed)
        key = int(re.search(r"-?\d+", q.answer_text).group())
        assert _oracle(q.text) == key, (seed, q.text, q.answer_text)


def test_three_situations_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("perimeter" in t for t in texts)
    assert any("same" in t for t in texts)
    assert any("solution" in t for t in texts)
