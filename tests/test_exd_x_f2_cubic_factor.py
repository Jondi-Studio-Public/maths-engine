"""Tests for the exam source of `senior-factorise-cubic-polynomials-in-cases-where-a-linear`.

The oracle reads the rendered prompt only. It recovers the cubic (solving for k from the stated factor where
there is a parameter), then checks that the key is a product of an integer linear factor and an integer
quadratic factor whose expansion is exactly that cubic.
"""

from __future__ import annotations

import re

import sympy
from sympy import Poly, Symbol
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
k = Symbol("k")
_T = standard_transformations + (implicit_multiplication_application, convert_xor)
_KEY_RE = re.compile(r"^\((.+?)\)\((.+)\)$")


def _p(text):
    return parse_expr(text, local_dict={"x": x, "k": k}, transformations=_T)


TOPIC = "exd_x_f2_cubic_factor"
SKILL_ID = "senior-factorise-cubic-polynomials-in-cases-where-a-linear"
KIND = "expression"


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


def test_prompt_variety_over_60_seeds():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


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


def _cubic_from_prompt(q):
    t = q.text
    if "where k is a constant" in t:
        factor = re.search(r"\((x [+-] \d+)\)", t).group(1)
        cubic = re.search(r"(?:The cubic |factor of )(.+?)(?: has |, where k)", t).group(1)
        expr = _p(cubic)
        root = sympy.solve(_p(factor), x)[0]
        k_value = sympy.solve(expr.subs(x, root), k)
        assert len(k_value) == 1
        return "parameter", sympy.expand(expr.subs(k, k_value[0]))
    cubic = re.match(r"(?:Factorise|Write|Express) (.+?) (?:as|into) ", t).group(1)
    return ("cubes" if cubic.count("x") == 1 or "x^2" not in cubic else "grouping"), sympy.expand(_p(cubic))


def test_key_expands_to_the_cubic_with_integer_linear_and_quadratic_factors_over_300_seeds():
    shapes = set()
    for seed in range(300):
        q = _exam(seed)
        shape, cubic = _cubic_from_prompt(q)
        shapes.add(shape)
        m = _KEY_RE.match(q.answer_text)
        assert m, (seed, q.answer_text)
        lin, quad = _p(m.group(1)), _p(m.group(2))
        assert Poly(lin, x).degree() == 1 and Poly(quad, x).degree() == 2
        assert all(c.is_integer for c in Poly(lin, x).all_coeffs() + Poly(quad, x).all_coeffs())
        assert sympy.expand(lin * quad - cubic) == 0, (seed, q.text, q.answer_text)
        # "completely": the quadratic factor must not factor further over the integers
        assert len(sympy.factor_list(quad)[1]) == 1 and sympy.factor_list(quad)[1][0][1] == 1, (seed, q.answer_text)
    assert shapes == {"cubes", "grouping", "parameter"}


def test_answer_form_is_factored_and_a_bracket_free_expansion_is_rejected():
    for seed in range(60):
        q = _exam(seed)
        assert q.answer_form == "factored"
        _, cubic = _cubic_from_prompt(q)
        expanded = str(cubic).replace("**", "^").replace("*", "")
        assert not _grade(q, expanded), (seed, expanded)
