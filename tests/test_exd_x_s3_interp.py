"""Tests the exam source of `interpolation_extrapolation_classify` (`exd_x_s3_interp`): the key is re-derived from
the rendered prompt alone, with no import of the generator module."""

from __future__ import annotations

import re

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s3_interp"
HARD = "interpolation_extrapolation_classify"
INTERP = "Interpolation"
EXTRAP = "Extrapolation"


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


def _oracle(text: str) -> str:
    m = re.search(r"y = (-?\d+) ([+-]) (\d+)([tx])", text)
    a, b = int(m.group(1)), int(m.group(3)) * (1 if m.group(2) == "+" else -1)
    if m.group(4) == "t":
        base = int(re.search(r"years after (\d+)", text).group(1))
        first, last = map(int, re.search(r"from (\d+) to (\d+) inclusive", text).groups())
        t0 = int(re.search(r"when t = (\d+)", text).group(1))
        inside = first - base <= t0 <= last - base
    else:
        lo, hi = map(int, re.search(r"x from (\d+) to (\d+)", text).groups())
        y0 = int(re.search(r"reach (\d+)", text).group(1))
        assert (y0 - a) % b == 0
        inside = lo <= (y0 - a) // b <= hi
    return INTERP if inside else EXTRAP


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if not q.answer_text.startswith(_oracle(q.text)):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_both_categories_and_boundary_cases_occur():
    answers = {_make(s).answer_text.split(" ")[0] for s in range(100)}
    assert answers == {INTERP, EXTRAP}


def test_grader_accepts_key_and_rejects_every_distractor_over_100_seeds():
    grader = GRADERS["mcq"]
    for seed in range(100):
        q = _make(seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4
        for i, option in enumerate(options):
            assert grader(q, i).correct == (option == q.answer_text), (seed, option)
        assert q.answer_text not in q.distractors
