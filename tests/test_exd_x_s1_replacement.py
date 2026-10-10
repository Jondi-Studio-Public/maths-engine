"""Exam source for `f10-compound-outcomes-replacement`: two draws WITHOUT replacement, four ordered outcomes."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape


import itertools

TOPIC = "exd_x_s1_replacement"
_HEAD = re.compile(
    r"A bag contains (\d+) (\w+) balls and (\d+) (\w+) balls, (\d+) in total\. .*not returned to the bag\. "
    r"List the probability of each of these four outcomes, in this order: (.+)\.$"
)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(text):
    m = _HEAD.match(text)
    R, ca, B, cb, T, order = m.groups()
    R, B = int(R), int(B)
    assert R + B == int(T)
    colour = [ca] * R + [cb] * B
    pairs = list(itertools.permutations(range(R + B), 2))
    out = []
    for label in order.split(", "):
        first, second = label.split("-then-")
        n = sum(1 for i, j in pairs if colour[i] == first and colour[j] == second)
        out.append(Fraction(n, len(pairs)))
    return out


def _parse(answer_text):
    return [Fraction(x.strip()) for x in answer_text.split(",")]


def test_shapes_over_60_seeds_and_new_over_hard():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("p11d_compound_outcomes_replacement", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_q(s)) == []


def test_independent_oracle_agrees_over_300_seeds():
    bad = []
    for s in range(300):
        q = _q(s)
        if _oracle(q.text) != _parse(q.answer_text) or _oracle(q.text) != list(q.answer):
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_grader_accepts_key_and_rejects_every_distractor():
    g = GRADERS["ordered_list"]
    bad = []
    for s in range(150):
        q = _q(s)
        if not g(q, q.answer_text).correct:
            bad.append(("key", s))
        assert len(q.distractors) == 3
        for d in q.distractors:
            if g(q, d).correct:
                bad.append(("distractor", s, d))
    assert bad == []
