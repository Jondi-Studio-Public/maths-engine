"""Tests for the `exd_x_meridian_voyage` exam source (hours to sail between two latitudes)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_meridian_voyage"
HARD_TOPIC = "earth_meridian_distance"
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


_LAT = r"(\d+)\u00b0(\d\d)\u2032 ([NS])"
_RE = re.compile(rf"from latitude {_LAT} to latitude {_LAT} at a steady speed of (\d+) knots")


def _signed(deg, minutes, hemi):
    return (1 if hemi == "N" else -1) * (int(deg) * 60 + int(minutes))


def _oracle(text):
    m = _RE.search(text).groups()
    a, b, speed = _signed(*m[0:3]), _signed(*m[3:6]), int(m[6])
    hours = Fraction(abs(a - b), speed)
    assert hours.denominator == 1
    return int(hours)
