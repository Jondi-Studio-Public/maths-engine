"""Tests for the exam source `exd_x_a3_equations_unknown_in_denominator` (linear-equations-with-fractions).

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

TOPIC = "exd_x_a3_equations_unknown_in_denominator"
KIND = "number"
HARD = [("linear_equation_fraction", {"kind": "both_sides", "h_hi": 6, "sol_hi": 9})]


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
        return GRADERS["mcq"](q, str(mcq_options(q).index(given))).correct
    return GRADERS[KIND](q, given).correct


def test_no_text_defects_over_60_seeds():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)


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

    skill = {s.id: s for s in load_skills()[1]}["linear-equations-with-fractions"]
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


def _oracle(text: str) -> Fraction:
    body = re.sub(r"^(Solve for x: |Solve: |Solve |Find x if )", "", text)
    x = Symbol("x")
    expr = sympy.together(_equation(body))
    numer, denom = sympy.fraction(expr)
    roots = [r for r in sympy.solve(numer, x) if denom.subs(x, r) != 0]
    assert len(roots) == 1, (text, roots)
    return Fraction(int(roots[0].p), int(roots[0].q))


def test_key_rederived_from_rendered_prompt_over_400_seeds():
    for seed in range(400):
        q = _make(seed)
        key = Fraction(re.search(r"-?\d+(?:/\d+)?", q.answer_text).group())
        assert _oracle(q.text) == key, (seed, q.text, q.answer_text)


def test_unknown_is_always_in_a_denominator():
    for seed in range(60):
        assert re.search(r"/\(x [+-] \d+\)", _make(seed).text), _make(seed).text


def test_excluded_value_is_never_the_key():
    for seed in range(300):
        q = _make(seed)
        key = int(re.search(r"-?\d+", q.answer_text).group())
        for sg, v in re.findall(r"\(x ([+-]) (\d+)\)", q.text):
            assert key != (-int(v) if sg == "+" else int(v)), q.text
