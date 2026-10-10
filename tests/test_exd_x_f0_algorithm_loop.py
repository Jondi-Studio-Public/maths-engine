"""Exam source of `f10-number-algorithms-decisions`: the trace is rerun from the rendered rule."""

from __future__ import annotations

import re

from sympy import Symbol, sympify

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f0_algorithm_loop"
_n = Symbol("n")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("number_algorithms_decisions", seed=seed, difficulty="hard")


def _expr(text: str):
    text = text.replace("÷", "/")
    text = re.sub(r"(\d)n", r"\1*n", text)
    return sympify(text, locals={"n": _n})


def rerun(text: str) -> list[int]:
    m = re.match(
        r"An algorithm starts with n = (\d+) and repeats this step: if n is (even|a multiple of 3), replace n by "
        r"(.+?); otherwise replace n by (.+?)\. It stops as soon as n (=|<) (\d+)\.",
        text,
    )
    start, test, first, other, op, bound = m.groups()
    start, bound = int(start), int(bound)
    divisor = 2 if test == "even" else 3
    first, other = _expr(first), _expr(other)
    n, out = start, [start]
    while not (n == bound if op == "=" else n < bound):
        expr = first if n % divisor == 0 else other
        n = int(expr.subs(_n, n))
        out.append(n)
        assert len(out) < 40
    return out


def test_no_text_defects_and_three_shapes():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []
    assert len({prompt_shape(_q(s)) for s in range(60)}) >= 3


def test_exam_adds_shapes_over_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_q(s)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_key_matches_rerun_over_300_seeds():
    bad = [(s, _q(s).text) for s in range(300) if rerun(_q(s).text) != _q(s).answer]
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["list"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)


def test_every_run_uses_both_branches_and_ends_at_the_stop():
    for seed in range(100):
        q = _q(seed)
        values = q.answer
        assert 5 <= len(values) <= 10
        assert len(set(values)) == len(values)


def test_deterministic():
    assert _q(3).to_json() == _q(3).to_json()
