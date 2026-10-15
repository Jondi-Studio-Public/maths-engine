"""Tests for the `exd_x_journey_cost` exam source (total wages plus fuel for a two-stage run)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_journey_cost"
HARD_TOPIC = "journey_time_or_cost"
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


_RE = re.compile(
    r"paid \$(\d+) for every hour.*?travels (\d+) km on a highway at an average speed of (\d+) km/h, using (\d+) litres "
    r"of fuel per 100 km, and then (\d+) km through town at an average speed of (\d+) km/h, using (\d+) litres per "
    r"100 km. Fuel costs \$(\d+\.\d\d) per litre"
)


def _oracle(text):
    wage, d1, v1, k1, d2, v2, k2, price = _RE.search(text).groups()
    hours = Fraction(int(d1), int(v1)) + Fraction(int(d2), int(v2))
    litres = Fraction(int(d1) * int(k1) + int(d2) * int(k2), 100)
    total = int(wage) * hours + Fraction(price) * litres
    assert total.denominator == 1
    return int(total)
