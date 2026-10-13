"""Exam-tier checks for the median-length / midpoint-distance source.

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
TOPIC = "exd_x_a4_median_length"
GRADER = GRADERS["expression"]
HARD = ("distance_two_points", {"kind": "perimeter", "coord_hi": 16, "leg_hi": 24, "allow_negative": True})


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


def _coords(text):
    return {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"([A-D])\s*[(=]\s*\(?(-?\d+), (-?\d+)\)", text)}


def _mid(p, q):
    return (Fraction(p[0] + q[0], 2), Fraction(p[1] + q[1], 2))


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        pts = _coords(q.text)
        if "median" in q.text:
            m = _mid(pts["A"], pts["B"])
            target = (Fraction(pts["C"][0]), Fraction(pts["C"][1]))
        else:
            m = _mid(pts["A"], pts["B"])
            target = _mid(pts["C"], pts["D"])
        d2 = (target[0] - m[0]) ** 2 + (target[1] - m[1]) ** 2
        assert d2.denominator == 1
        want = sympy.sqrt(int(d2))
        got = sympy.sympify(q.answer_text)
        if sympy.simplify(got - want) != 0:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad


def test_working_has_no_repeated_step():
    from mathgen import make as _make

    for seed in range(100):
        for line in _make("exd_x_a4_median_length", seed=seed, difficulty="exam").working:
            sides = [x.strip().rstrip(".") for x in line.split(" = ")]
            assert len(set(sides)) == len(sides), line
