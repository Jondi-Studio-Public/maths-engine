"""Tests for the inverse-function exam source (`exd_x_f1_inverse_mobius`)."""

from __future__ import annotations

import re

from sympy import Eq, Symbol, simplify, solve
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_inverse_mobius"
SKILL_ID = "senior-consider-inverses-of-one-to-one-function"
x, y = Symbol("x"), Symbol("y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    src = [s for s in _skill().sources if s.difficulty == "hard"][0]
    return make(src.topic, seed=seed, difficulty="hard", **src.params)


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x})


def _key_from_prompt(text):
    f = _parse(re.search(r"f\(x\) = (.+?)\.(?: |$)", text).group(1))
    sols = solve(Eq(y, f), x)
    assert len(sols) == 1
    return sols[0].subs(y, x)


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
    bad = [s for s in range(300) if simplify(_parse(_exam(s).answer_text) - _key_from_prompt(_exam(s).text)) != 0]
    assert bad == []


def test_grader_accepts_key_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        assert GRADERS["expression"](q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) == 3
        for dist in q.distractors:
            assert not GRADERS["expression"](q, dist).correct, (seed, dist, q.answer_text)


def test_skill_has_exam_source_last():
    src = _skill().sources[-1]
    assert (src.topic, src.difficulty, list(src.stages)) == (TOPIC, "exam", ["exam"])
