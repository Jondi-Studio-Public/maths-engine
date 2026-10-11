"""Tests for the one-to-one exam source (`exd_x_f1_one_to_one_parameter`)."""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import Symbol, diff, solve
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_one_to_one_parameter"
SKILL_ID = "senior-determine-if-a-function-is-one-to-one"
x, p = Symbol("x"), Symbol("p")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    """The skill's highest existing tier (it has only an easy one)."""
    src = [s for s in _skill().sources if s.difficulty != "exam"][-1]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **src.params)


def _poly_from_prompt(text):
    m = re.search(r"f\(x\) = (.+?)(?: is defined| *,|, where|\Z)", text)
    return parse_expr(m.group(1).replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x, "p": p})


def _key_from_prompt(text):
    poly = _poly_from_prompt(text)
    upward = "≥" in text
    extreme = "least" if "least" in text else "greatest"
    a = poly.coeff(x, 2)
    if "value of k" in text:
        (h,) = solve(diff(poly, x), x)
        # one-to-one on [k, inf) iff k >= h, on (-inf, k] iff k <= h
        assert (extreme == "least") == upward
        return int(h)
    k = int(re.search(r"[≥≤] (-?\d+)", text).group(1))
    b = poly.coeff(x, 1)  # coefficient of x is p
    assert b == p

    # f(u) - f(v) = (u - v) * (a (u + v) + p): not one-to-one iff a s + p = 0 for some sum s of
    # two distinct points of the domain, i.e. s in (2k, inf) for x >= k and (-inf, 2k) for x <= k.
    def injective(pv):
        s0 = Fraction(-pv, 1) / int(a)
        return not (s0 > 2 * k if upward else s0 < 2 * k)

    good = [pv for pv in range(-200, 201) if injective(pv)]
    return min(good) if extreme == "least" else max(good)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_key_rederived_from_prompt():
    bad = [s for s in range(300) if _key_from_prompt(_exam(s).text) != _exam(s).answer]
    assert bad == []


def test_both_forms_and_both_extremes_occur():
    texts = [_exam(s).text for s in range(60)]
    assert any("value of p" in t for t in texts) and any("value of k" in t for t in texts)
    assert any("least" in t for t in texts) and any("greatest" in t for t in texts)


def test_grader_accepts_key_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        assert GRADERS["number"](q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) == 3
        for dist in q.distractors:
            assert not GRADERS["number"](q, dist).correct, (seed, dist, q.answer_text)


def test_skill_has_exam_source_last():
    src = _skill().sources[-1]
    assert (src.topic, src.difficulty, list(src.stages)) == (TOPIC, "exam", ["exam"])
