"""Tests for the conjugate-roots exam source (`exd_x_f1_conjugate_cubic`)."""

from __future__ import annotations

import re

from sympy import Symbol, linsolve, re as sym_re, im as sym_im, sympify

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_conjugate_cubic"
SKILL_ID = "senior-consider-conjugate-roots-for-polynomials-with-real-coefficients"
x = Symbol("x")
_REAL = {n: Symbol(n, real=True) for n in "abc"}


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard_like(seed):
    src = [s for s in _skill().sources if s.difficulty != "exam"][-1]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **src.params)


def _num(text):
    """Parse a stated number such as '-3 + i', '3i', '5'."""
    return sympify(re.sub(r"(\d)i", r"\1*I", text.strip()).replace("i", "I").replace(" ", ""))


def _key_from_prompt(text):
    poly_text = re.search(r"x\^3[^=]*=", text).group(0)[:-1]
    poly_text = re.sub(r"(\d)x", r"\1*x", poly_text)
    poly_text = re.sub(r"([abc])x", r"\1*x", poly_text).replace("^", "**")
    poly = sympify(poly_text.strip(), locals=_REAL)
    if text.startswith("Given that"):
        roots = re.search(r"Given that (.+?) and (.+?) are roots", text).groups()
    elif "Two of its roots are" in text:
        roots = re.search(r"roots are (.+?) and (.+?)\. Find", text).groups()
    else:
        roots = re.search(r"^(.+?) is a root|one root is (.+?)\. Find", text).groups()
        roots = [g for g in roots if g]
    eqs = []
    for rt in roots:
        v = _num(rt)
        val = sympify(poly.subs(x, v)).expand()
        eqs += [sym_re(val), sym_im(val)]
    eqs = [e for e in eqs if e != 0]
    unknowns = sorted(poly.free_symbols - {x}, key=str)
    sol = linsolve(eqs, unknowns)
    assert len(sol) == 1
    values = dict(zip(unknowns, next(iter(sol))))
    target = re.search(r"value of ([abc])", text).group(1)
    return values[_REAL[target]]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(_hard_like(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_key_rederived_from_prompt():
    bad = [s for s in range(300) if _key_from_prompt(_exam(s).text) != _exam(s).answer]
    assert bad == []


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
