"""Tests for the exam source of `senior-recognise-and-use-the-recursive-definition-of-a`.

The oracle reads the rendered prompt only: it parses the three terms (linear in x), solves t2^2 = t1 * t3 for x,
evaluates the terms and reads off the first term and the ratio, then compares with the labelled-pair key.
"""

from __future__ import annotations

import re

import sympy
from sympy import Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

x = Symbol("x")
_T = standard_transformations + (implicit_multiplication_application, convert_xor)
_TERMS_RE = re.compile(r"(?:are|begins) (.+?)(?:\. Find|, where)")


def _p(text):
    return parse_expr(text, local_dict={"x": x}, transformations=_T)


TOPIC = "exd_x_f2_geometric_recursion"
SKILL_ID = "senior-recognise-and-use-the-recursive-definition-of-a"
KIND = "pair"


def _skill():
    _, skills = load_skills()
    return next(s for s in skills if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    source = next(s for s in _skill().sources if s.difficulty == "hard")
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (source.params or {}).items()}
    return make(source.topic, seed=seed, difficulty="hard", **params)


def _grade(q, given):
    if KIND == "mcq":
        options = mcq_options(q)
        given = str(options.index(given)) if given in options else "-1"
    return GRADERS[KIND](q, given).correct


# -- wiring, text quality and variety ------------------------------------------


def test_exam_source_is_the_last_source_of_the_skill():
    source = _skill().sources[-1]
    assert source.topic == TOPIC
    assert source.difficulty == "exam"
    assert list(source.stages) == ["exam"]
    assert dict(source.params) == {"tier": "exam"}
    assert _skill().answer == KIND


def test_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _exam(seed)
        defects = find_text_defects(q, include_working=True)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_exam_adds_shapes_not_present_in_the_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_determinism():
    for seed in (0, 1, 2, 3, 100, 12345):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_preset_registered_and_other_tiers_rejected():
    make(TOPIC, seed=0, difficulty="exam")
    try:
        make(TOPIC, seed=0, difficulty="hard")
    except KeyError:
        return
    raise AssertionError("only the exam preset should exist")


def test_three_distinct_distractors_and_none_equal_to_key():
    for seed in range(120):
        q = _exam(seed)
        assert len(q.distractors) == 3, (seed, q.distractors)
        assert len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors


# -- the grader: key accepted, every distractor rejected -----------------------


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    for seed in range(120):
        q = _exam(seed)
        assert _grade(q, q.answer_text), (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert not _grade(q, d), (seed, q.text, "distractor accepted", d)


def _expected(q):
    m = _TERMS_RE.search(q.text)
    assert m, q.text
    parts = m.group(1).replace(" and ", ", ").split(", ")
    assert len(parts) == 3, parts
    t1, t2, t3 = (_p(g) for g in parts)
    solutions = sympy.solve(sympy.expand(t2**2 - t1 * t3), x)
    assert len(solutions) == 1, (q.text, solutions)
    v1, v2, v3 = (t.subs(x, solutions[0]) for t in (t1, t2, t3))
    assert v2 / v1 == v3 / v2
    return int(v1), int(v2 / v1)


def test_key_matches_independent_recomputation_over_300_seeds():
    for seed in range(300):
        q = _exam(seed)
        first, ratio = _expected(q)
        assert q.answer == {"first_term": first, "r": ratio}, (seed, q.text, q.answer)
        assert q.answer_text == f"first_term = {first}, r = {ratio}"
