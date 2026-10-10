"""Exam source for `f10-probability-complements-compounds`: "at least one" probabilities via the complement."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape


import itertools

TOPIC = "exd_x_s1_complement"
SKILL = "f10-probability-complements-compounds"
_SAMPLING = re.compile(
    r"A box holds (\d+) \w+, of which (\d+) are \w+\. (Two|Three) of the \w+ are chosen at random without replacement\."
)
_PARTS = re.compile(r"(\w+) (\d+(?:/\d+)?)(?:,|\.)")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(text):
    m = _SAMPLING.match(text)
    if m:
        n, d, k = int(m.group(1)), int(m.group(2)), {"Two": 2, "Three": 3}[m.group(3)]
        hits = sum(1 for s in itertools.combinations(range(n), k) if any(i < d for i in s))
        total = sum(1 for _ in itertools.combinations(range(n), k))
        return Fraction(hits, total)
    body = text.split("are: ", 1)[1].split(". Find", 1)[0] + "."
    probs = [Fraction(v) for _, v in _PARTS.findall(body)]
    assert len(probs) >= 2, text
    none = Fraction(1)
    for p in probs:
        none *= 1 - p
    return 1 - none


def test_shapes_over_60_seeds_and_new_over_hard():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("probability_complements_compounds", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_q(s)) == []


def test_independent_oracle_agrees_over_300_seeds():
    bad = []
    for s in range(300):
        q = _q(s)
        if _oracle(q.text) != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_grader_accepts_key_and_rejects_every_distractor():
    g = GRADERS["fraction"]
    bad = []
    for s in range(150):
        q = _q(s)
        if not g(q, q.answer_text).correct:
            bad.append(("key", s))
        for d in q.distractors:
            if g(q, d).correct:
                bad.append(("distractor", s, d))
        assert len(q.distractors) >= 2
    assert bad == []
