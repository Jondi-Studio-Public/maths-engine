"""Exam-tier checks for the parallel/perpendicular y-intercept source.

(a) the exam source shows several prompt shapes, some absent from the hard tier; (b) the key is re-derived
from the RENDERED PROMPT with independent code over 300 seeds; (c) the skill's grader accepts the key and
rejects every distractor over 150 seeds.
"""

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
TOPIC = "exd_x_a4_intercept_parallel_perp"
GRADER = GRADERS["number"]
HARD = ("linear_from_table", {"kind": "intercept_context", "hi": 15})


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(60):
        q = _q(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, **HARD[1])) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_deterministic():
    for seed in (0, 1, 7, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_grader_accepts_key_and_rejects_every_distractor():
    bad = []
    for seed in range(150):
        q = _q(seed)
        assert len(q.distractors) >= 2, (seed, q.distractors)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        pts = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"([ABC])\((-?\d+), (-?\d+)\)", q.text)}
        slope = Fraction(pts["B"][1] - pts["A"][1], pts["B"][0] - pts["A"][0])
        if "perpendicular" in q.text:
            slope = -1 / slope
        want = pts["C"][1] - slope * pts["C"][0]
        if want != int(q.answer_text) or want.denominator != 1:
            bad.append((seed, q.text, q.answer_text, want))
    assert bad == [], bad


def test_both_relationships_occur():
    words = {"perpendicular" if "perpendicular" in _q(s).text else "parallel" for s in range(60)}
    assert words == {"perpendicular", "parallel"}
