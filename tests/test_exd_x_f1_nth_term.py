"""Tests for the `nth-term-rule` exam source (`exd_x_f1_nth_term`)."""

from __future__ import annotations

import re

from sympy import Symbol, expand, linsolve, symbols, sympify

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_nth_term"
SKILL_ID = "nth-term-rule"
x = Symbol("x")
A, D = symbols("A D")


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    skill = _skill()
    src = [s for s in skill.sources if s.difficulty == "hard"][0]
    return make(src.topic, seed=seed, difficulty="hard", **src.params)


_SUM = re.compile(r"sum of terms (\d+) and (\d+) (?:of an arithmetic sequence )?is (-?\d+)")


def _t(n):
    return A + (n - 1) * D


def _key_from_prompt(text):
    ratio = re.search(r"term (\d+) (?:is|equal to) (\d+) times term (\d+)", text)
    r, k, s = (int(v) for v in ratio.groups())
    eqs = [_t(r) - k * _t(s)]
    summ = _SUM.search(text)
    if summ:
        p, q, total = (int(v) for v in summ.groups())
        eqs.append(_t(p) + _t(q) - total)
    else:
        # the value statement is the last "term N is V." / "Its term N is V."
        u, v = (int(g) for g in re.findall(r"term (\d+) (?:is|equal to) (-?\d+)\.", text)[-1])
        eqs.append(_t(u) - v)
    sol = linsolve(eqs, A, D)
    assert len(sol) == 1
    a, d = next(iter(sol))
    return expand(a + d * (x - 1))


def test_key_rederived_from_prompt():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        if expand(sympify(q.answer_text) - _key_from_prompt(q.text)) != 0:
            bad.append(seed)
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
