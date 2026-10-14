"""Exam source of `function-notation`: the key is recomputed from the rendered functions with sympy."""

from __future__ import annotations

import re

from sympy import Symbol, solve, sympify

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f0_function_composition"
_x = Symbol("x")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("function_evaluation", seed=seed, difficulty="hard", a_lo=0)


def _expr(text: str):
    text = re.sub(r"(\d)x", r"\1*x", text.replace("^", "**"))
    return sympify(text, locals={"x": _x})


def derive(text: str) -> int:
    m = re.match(r"Let f\(x\) = (.+?) and g\(x\) = (.+?)\. Find (f\(g|g\(f)\((-?\d+)\)\)\.", text)
    if m:
        fe, ge, order, n = _expr(m.group(1)), _expr(m.group(2)), m.group(3), int(m.group(4))
        if order == "f(g":
            return int(fe.subs(_x, ge.subs(_x, n)))
        return int(ge.subs(_x, fe.subs(_x, n)))
    m = re.match(r"Given that f\((.+?)\) = (.+?), find f\((-?\d+)\)\.", text)
    arg, body, target = _expr(m.group(1)), _expr(m.group(2)), int(m.group(3))
    (x0,) = solve(arg - target, _x)
    # Independent route: recover f(t) as a polynomial in t and evaluate it at the target.
    t = Symbol("t")
    f_of_t = body.subs(_x, solve(arg - t, _x)[0])
    assert f_of_t.subs(t, target) == body.subs(_x, x0)
    return int(body.subs(_x, x0))


def test_exam_adds_shapes_over_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_q(s)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_key_matches_independent_derivation_over_300_seeds():
    bad = [(s, _q(s).text, _q(s).answer) for s in range(300) if derive(_q(s).text) != _q(s).answer]
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
