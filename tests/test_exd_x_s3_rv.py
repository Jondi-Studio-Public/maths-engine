"""Tests the exam source of `non_uniform_discrete_random_variable` (`exd_x_s3_rv`): the key is re-derived from the
rendered prompt alone, with exact fractions."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s3_rv"
HARD = "non_uniform_discrete_random_variable"


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_make(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD, seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam source adds no prompt shape beyond the hard tier"


def test_no_text_defects_and_deterministic():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _make(seed).to_json()


def _weight(label: str, x: int) -> int:
    if label == "kx":
        return x
    if label == "kx^2":
        return x * x
    if label == "k(x^2 + 1)":
        return x * x + 1
    m = re.fullmatch(r"k\(x \+ (\d+)\)", label)
    if m:
        return x + int(m.group(1))
    m = re.fullmatch(r"k\((\d+) - x\)", label)
    return int(m.group(1)) - x


def _oracle(text: str) -> Fraction:
    values_part = re.search(r"takes the values (.*?), with", text).group(1)
    xs = [int(v) for v in re.findall(r"\d+", values_part)]
    label = re.search(r"P\(X = x\) = (.*?) for each", text).group(1)
    ws = [_weight(label, x) for x in xs]
    total = sum(ws)
    mean = Fraction(sum(w * x for w, x in zip(ws, xs)), total)
    return Fraction(sum(w * (x - mean) ** 2 for w, x in zip(ws, xs)), total)


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        want = _oracle(q.text)
        if Fraction(q.answer["p"], q.answer["q"]) != want or Fraction(q.answer_text) != want:
            bad.append((seed, q.text, q.answer_text, want))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_100_seeds():
    grader = GRADERS["fraction"]
    for seed in range(100):
        q = _make(seed)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert Fraction(d) != Fraction(q.answer_text), (seed, d)
            assert not grader(q, d).correct, (seed, d, q.answer_text)
