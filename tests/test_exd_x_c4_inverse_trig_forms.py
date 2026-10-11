"""Exam source for integrals that need completing the square before an inverse trig antiderivative applies.

The oracle integrates the rendered integrand by numerical quadrature and compares with the key.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import Question
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_c4_inverse_trig_forms"
HARD_TOPIC = "integrate_expressions_of_the_form_frac_pm1_sqrt"
GRADER = GRADERS["expression"]
_HARD_KW = {}


def _make(seed: int) -> Question:
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed: int) -> Question:
    return make(HARD_TOPIC, seed=seed, difficulty="hard", **_HARD_KW)


def test_no_text_defects():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert shapes - hard_shapes, (shapes, hard_shapes)


def test_determinism():
    for seed in (0, 1, 7, 99, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")


from mpmath import mp, quad
from sympy import Symbol, lambdify
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

_X = Symbol("x", real=True)
_TR = standard_transformations + (implicit_multiplication_application,)


def _expr(text: str):
    s = text.replace("π", "pi").replace("·", "*").replace("^", "**")
    s = re.sub(r"\be\*\*\(", "exp(", s)
    s = re.sub(r"\be\*\*x\b", "exp(x)", s)
    return parse_expr(s, transformations=_TR, local_dict={"x": _X})


@mp.workdps(25)  # set locally: a module-level mp.dps leaks into every test the worker runs
def _agrees(q: Question) -> bool:
    m = re.match(r"Find ∫ from x = (-?\d+) to x = (-?\d+) of (.+) dx\.", q.text)
    lo, hi, body = int(m.group(1)), int(m.group(2)), m.group(3)
    f = lambdify(_X, _expr(body), "mpmath")
    return abs(quad(f, [lo, hi]) - parse_expr(q.answer_text).evalf(25)) < 1e-8


def test_both_integrand_forms_occur():
    kinds = {"sqrt" in q.text for q in (_make(s) for s in range(60))}
    assert kinds == {True, False}


def test_independent_oracle_matches_key_over_400_seeds():
    bad = []
    for seed in range(400):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _make(seed)
        assert len(q.distractors) >= 1, (seed, q.text)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]
