"""Tests for the midpoint exam source (`exd_x_f1_midpoint_parametric`)."""

from __future__ import annotations

import re

from sympy import Rational, Symbol, solve
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_midpoint_parametric"
SKILL_ID = "senior-determine-the-coordinates-of-the-midpoint-of-two"
t, x, y = Symbol("t"), Symbol("x"), Symbol("y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_ENV = {"t": t, "x": x, "y": y}


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    src = [s for s in _skill().sources if s.difficulty == "hard"][0]
    return make(src.topic, seed=seed, difficulty="hard", **src.params)


def _parse(text):
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=_ENV)


def _key_from_prompt(text):
    m = re.search(r"A\(([^,]+), ([^)]+)\) and B\(([^,]+), ([^)]+)\)", text)
    ax, ay, bx, by = (_parse(g) for g in m.groups())
    mx, my = Rational(1, 2) * (ax + bx), Rational(1, 2) * (ay + by)
    line = re.search(r"line (.+?)(?:\. Find|\.$)", text).group(1)
    lhs, rhs = (_parse(side) for side in line.split(" = "))
    eq = (lhs - rhs).subs({x: mx, y: my})
    (sol,) = solve(eq, t)
    return sol


def test_key_rederived_from_prompt():
    bad = [s for s in range(300) if _key_from_prompt(_exam(s).text) != _exam(s).answer]
    assert bad == []


def test_both_line_forms_occur():
    texts = [_exam(s).text for s in range(60)]
    assert any("line y = " in t for t in texts) and any("line y = " not in t for t in texts)


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
