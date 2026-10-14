"""Tests for the exam source `exd_x_a3_linear_for_which_parameter` (senior-identify-and-solve-linear-equations).

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

TOPIC = "exd_x_a3_linear_for_which_parameter"
KIND = "mcq"
HARD = [("identify_and_solve_linear_equation", {})]


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

    skill = {s.id: s for s in load_skills()[1]}["senior-identify-and-solve-linear-equations"]
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


def _oracle(text: str) -> str:
    eq = re.search(
        r"(?:The equation )?(.+?) (?:contains a constant k|\. For which)", text.replace(". For which", " . For which")
    )
    raw = eq.group(1).strip().rstrip(".").strip()
    k, x = Symbol("k"), Symbol("x")
    full = sympy.expand(_equation(raw))
    ks = [v for v in sympy.solve(sympy.Poly(full, x).nth(2), k) if v.is_real]
    if not ks:
        return "No value of k makes the equation linear"
    assert len(ks) == 1
    xs = sympy.solve(full.subs(k, ks[0]), x)
    assert len(xs) == 1
    sol = xs[0]
    sol_text = str(sol.p) if sol.q == 1 else f"{sol.p}/{sol.q}"
    return f"k = {ks[0]}; x = {sol_text}"


def test_key_rederived_from_rendered_prompt_over_400_seeds():
    for seed in range(400):
        q = _make(seed)
        assert _oracle(q.text) == q.answer_text, (seed, q.text, q.answer_text)


def test_exactly_one_option_is_graded_correct_and_options_are_distinct():
    for seed in range(150):
        q = _make(seed)
        options = mcq_options(q)
        assert len(options) == len(set(options)) == 4
        right = [i for i in range(len(options)) if GRADERS["mcq"](q, str(i)).correct]
        assert [options[i] for i in right] == [q.answer_text]


def test_no_value_option_never_matches_a_linearisable_equation():
    keys = set()
    for seed in range(100):
        q = _make(seed)
        keys.add(q.answer_text.startswith("k = "))
        assert q.answer_text.startswith("k = ") == (_oracle(q.text).startswith("k = "))
    assert keys == {True, False}


def test_no_value_is_keyed_on_roughly_a_quarter_of_seeds():
    n = sum(_make(s).answer_text.startswith("No value") for s in range(400))
    assert 60 <= n <= 140, n
