"""Tests for the exam source `exd_x_a3_rearrange_repeated_subject` (rearrange-simple-formulas).

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

TOPIC = "exd_x_a3_rearrange_repeated_subject"
KIND = "expression"
HARD = [("rearrange", {}), ("rearrange_powers_roots", {})]


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

    skill = {s.id: s for s in load_skills()[1]}["rearrange-simple-formulas"]
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


def _oracle(text: str):
    target = re.search(r"Make ([a-z]) the subject", text).group(1)
    formula = text.split("formula ")[1]
    sym = Symbol(target)
    left, right = formula.split("=")
    names = {c: Symbol(c) for c in set(re.findall(r"[a-z]", formula))}
    sols = sympy.solve(
        sympy.Eq(parse_expr(_implicit(left), local_dict=names), parse_expr(_implicit(right), local_dict=names)), sym
    )
    assert len(sols) == 1
    return sols[0]


def test_key_rederived_from_rendered_prompt_over_400_seeds():
    """The key satisfies the rendered formula exactly at fixed rational points; 80 of them are also re-solved
    symbolically and compared with the key."""
    points = [(2, 3, 5, 7, 11, 13), (3, -5, 7, 2, -9, 4), (-4, 9, 1, -6, 8, 5)]
    for seed in range(400):
        q = _make(seed)
        target = re.search(r"Make ([a-z]) the subject", q.text).group(1)
        left, right = q.text.split("formula ")[1].split("=")
        names = {c: Symbol(c) for c in set(re.findall(r"[a-z]", q.text.split("formula ")[1]))}
        lhs, rhs = parse_expr(_implicit(left), local_dict=names), parse_expr(_implicit(right), local_dict=names)
        key = sympy.sympify(q.answer_text, locals=names)
        others = [names[c] for c in sorted(names) if c != target]
        checked = 0
        for pt in points:
            env = {sym: sympy.Rational(v, 1) + sympy.Rational(i, 7) for i, (sym, v) in enumerate(zip(others, pt))}
            k = key.xreplace(env)
            if not k.is_finite:
                continue
            checked += 1
            assert sympy.nsimplify((lhs - rhs).xreplace(env).xreplace({names[target]: k})) == 0, (seed, q.text)
        assert checked >= 2, (seed, q.text)
        if seed < 80:
            assert sympy.cancel(sympy.together(_oracle(q.text) - key)) == 0, (seed, q.text, q.answer_text)


def test_target_is_repeated_or_sits_in_a_denominator():
    for seed in range(120):
        text = _make(seed).text
        target = re.search(r"Make ([a-z]) the subject", text).group(1)
        formula = text.split("formula ")[1]
        assert formula.count(target) >= 2 or re.search(rf"/[(]?[^=]*?{target}", formula), text


def test_equivalent_forms_of_the_key_are_accepted():
    for seed in range(100):
        q = _make(seed)
        expr = sympy.sympify(q.answer_text)
        for alt in (sympy.together(expr), sympy.cancel(expr)):
            assert GRADERS[KIND](q, str(alt)).correct, (seed, str(alt))
