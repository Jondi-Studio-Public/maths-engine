"""Tests for the exam source of `senior-examine-examples-of-variable-rates-of-change-of`.

The oracle reads the rendered prompt only: it parses the stated function(s) with sympy, finds the interval
endpoints in the remaining text and computes each average rate of change as an exact difference quotient.
"""

from __future__ import annotations

import re
from fractions import Fraction

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


def _p(text):
    return parse_expr(text, local_dict={"x": x}, transformations=_T)


def _split(text):
    """(function texts, remaining prompt text after the definitions)."""
    start = text.index("f(x) = ") + len("f(x) = ")
    stop_candidates = [i for i in (text.find(".", start), text.find(", find", start)) if i != -1]
    stop = min(stop_candidates)
    defs = text[start:stop]
    rest = text[stop:]
    if " and g(x) = " in defs:
        f_text, g_text = defs.split(" and g(x) = ")
        return f_text, g_text, rest
    return defs, None, rest


def _rate(expr, a, b):
    return Fraction(int(expr.subs(x, b) - expr.subs(x, a)), b - a)


TOPIC = "exd_x_f2_rate_of_change"
SKILL_ID = "senior-examine-examples-of-variable-rates-of-change-of"
KIND = "number"


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


def _expected(q):
    f_text, g_text, rest = _split(q.text)
    f = _p(f_text)
    nums = [int(v) for v in re.findall(r"-?\d+", rest)]
    if g_text is not None:
        a, b = nums[-2:]
        return "composite", _rate(f.subs(x, _p(g_text)), a, b)
    if len(nums) >= 4:
        a1, a2, b1, b2 = nums[-4:]
        return "compare", _rate(f, a1, a2) - _rate(f, b1, b2)
    a, b = nums[-2:]
    return "factored", _rate(f, a, b)


def test_key_matches_independent_recomputation_over_300_seeds():
    shapes = set()
    for seed in range(300):
        q = _exam(seed)
        shape, expected = _expected(q)
        shapes.add(shape)
        assert expected.denominator == 1, (seed, q.text)
        assert str(expected) == q.answer_text, (seed, q.text, expected, q.answer_text)
    assert shapes == {"composite", "compare", "factored"}
