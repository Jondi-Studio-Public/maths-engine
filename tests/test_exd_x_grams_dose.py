"""Tests for the `exd_x_grams_dose` exam source (tablets per dose from a mg/kg/day rate)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_grams_dose"
HARD_TOPIC = "grams_milligrams_convert"
HARD_PARAMS = {"difficulty": "hard"}
_GRADER = GRADERS["number"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_shapes_vary_and_are_new_over_hard():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD_TOPIC, seed=s, **HARD_PARAMS)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q.text)
        if expected != q.answer:
            bad.append((seed, q.text, expected, q.answer))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert len(q.distractors) >= 3, (seed, q.distractors)
        if not _GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed))
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d))
    assert bad == [], bad[:3]


_RE = re.compile(r"mass of (\d+) kg is prescribed (\d+) mg .*? over (\d+) doses\. Each tablet contains ([\d.]+) g")


def _oracle(text):
    mass, rate, doses, grams = _RE.search(text).groups()
    per_dose = Fraction(int(mass) * int(rate), int(doses))
    tablets = per_dose / (Fraction(grams) * 1000)
    assert tablets.denominator == 1
    return int(tablets)
