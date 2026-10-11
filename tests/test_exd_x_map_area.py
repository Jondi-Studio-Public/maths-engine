"""Tests for the `exd_x_map_area` exam source (real area in hectares from a map area)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_map_area"
HARD_TOPIC = "map_scale_distance"
HARD_PARAMS = {"mode": "ratio_reverse"}
_GRADER = GRADERS["number"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_shapes_vary_and_are_new_over_hard():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD_TOPIC, seed=s, **HARD_PARAMS)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_deterministic():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


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


_SCALE = re.compile(r"scale of 1 : (\d+)")
_AREA = re.compile(r"has an area of ([\d.]+) cm")
_SIDES = re.compile(r"measuring ([\d.]+) cm by ([\d.]+) cm")


def _oracle(text):
    n = int(_SCALE.search(text).group(1))
    if _AREA.search(text):
        map_cm2 = Fraction(_AREA.search(text).group(1))
    else:
        a, b = _SIDES.search(text).groups()
        map_cm2 = Fraction(a) * Fraction(b)
    hectares = map_cm2 * n * n / 10**8
    assert hectares.denominator == 1
    return int(hectares)


def test_both_prompt_forms_occur():
    texts = [_exam(s).text for s in range(60)]
    assert any("measuring" in t for t in texts) and any("has an area of" in t for t in texts)


def test_indefinite_article_matches_the_map_name():
    seen = set()
    for seed in range(120):
        text = make("exd_x_map_area", seed=seed, difficulty="exam").text
        assert " a online map" not in text, text
        seen.add("an online map" in text)
    assert seen == {True, False}
