"""Tests for the exam source of `senior-find-the-composition-of-two-functions`.

The oracle reads the rendered prompt only: it parses the stated f and g, builds the requested composite from
scratch with sympy and compares it numerically with the key at many points. It also checks the answer's
required form (rational denominator, single fraction) on the parsed key.
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
_F_RE = re.compile(r"f\(x\) = (1/\([^)]*\))")
_G_RE = re.compile(r"g\(x\) = (sqrt\([^)]*\))")
_ASK_RE = re.compile(r"(f\(g\(f\(x\)\)\)|f\(f\(x\)\)|f\(g\(x\)\)|g\(g\(x\)\))")


def _p(text):
    return parse_expr(text, local_dict={"x": x}, transformations=_T)


TOPIC = "exd_x_f2_compose"
SKILL_ID = "senior-find-the-composition-of-two-functions"
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


def _composite(q):
    f = _p(_F_RE.search(q.text).group(1)) if _F_RE.search(q.text) else None
    g = _p(_G_RE.search(q.text).group(1)) if _G_RE.search(q.text) else None
    ask = _ASK_RE.search(q.text).group(1)
    if ask == "f(g(f(x)))":
        return f.subs(x, g.subs(x, f)), ask
    if ask == "f(f(x))":
        return f.subs(x, f), ask
    if ask == "f(g(x))":
        return f.subs(x, g), ask
    return g.subs(x, g), ask


def _values(expr_a, expr_b):
    fa = sympy.lambdify(x, expr_a, "math")
    fb = sympy.lambdify(x, expr_b, "math")
    points = [i / 17 for i in range(-300, 600)]
    points += [c + 1 / m for c in range(-10, 12) for m in range(2, 40)]
    points += [i / 2 for i in range(0, 600)]
    agree = 0
    for pt in points:
        try:
            va, vb = fa(pt), fb(pt)
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
        if isinstance(va, complex) or isinstance(vb, complex):
            continue
        assert abs(va - vb) < 1e-6 * max(1.0, abs(vb)), (pt, va, vb)
        agree += 1
    return agree


def test_key_matches_independently_built_composite_over_300_seeds():
    asks = set()
    for seed in range(300):
        q = _exam(seed)
        expected, ask = _composite(q)
        asks.add(ask)
        assert _values(_p(q.answer_text), expected) >= 5, (seed, q.text, q.answer_text)
    assert len(asks) == 4


def test_required_forms_hold_on_the_parsed_key():
    seen = set()
    for seed in range(300):
        q = _exam(seed)
        key = _p(q.answer_text)
        if q.answer_form == "rationalised":
            _, den = sympy.fraction(sympy.together(key))
            assert not den.has(sympy.sqrt(x - 1).func), q.answer_text
            assert "sqrt" not in q.answer_text.split("/(")[-1], q.answer_text
            seen.add("rationalised")
        elif q.answer_form == "single_fraction":
            assert q.answer_text.count("/") == 1 and q.answer_text.startswith("("), q.answer_text
            seen.add("single_fraction")
    assert seen == {"rationalised", "single_fraction"}


def test_unrationalised_form_is_rejected_when_rationalised_is_required():
    checked = 0
    for seed in range(300):
        q = _exam(seed)
        if q.answer_form != "rationalised":
            continue
        sign, n = re.search(r"f\(x\) = 1/\(x ([+-]) (\d+)\)", q.text).groups()
        raw = f"1/({_G_RE.search(q.text).group(1)} {sign} {n})"
        assert not _grade(q, raw), (q.text, raw)
        checked += 1
    assert checked >= 20
