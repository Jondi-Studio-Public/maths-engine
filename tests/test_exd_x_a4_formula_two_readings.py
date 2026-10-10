"""Exam-tier checks for the two-readings linear formula source.

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
TOPIC = "exd_x_a4_formula_two_readings"
GRADER = GRADERS["expression"]
HARD = ("develop_a_linear_formula_from_a_word_description", {"variant": "back_solved_intercept"})


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


def _readings(text):
    body = text.split("Write a formula")[0]
    nums = [int(x) for x in re.findall(r"\d+", body)]
    if "A further" in body:
        v1, t1, gap, v2 = nums
        return (t1, v1), (t1 + gap, v2)
    v1, t1, v2, t2 = nums
    return (t1, v1), (t2, v2)


def test_key_matches_independent_derivation_over_300_seeds():
    t = sympy.Symbol("t")
    bad = []
    for seed in range(300):
        q = _q(seed)
        (t1, v1), (t2, v2) = _readings(q.text)
        slope = Fraction(v2 - v1, t2 - t1)
        start = v1 - slope * t1
        rhs = q.answer_text.split(" = ")[1]
        expr = parse_expr(rhs, transformations=TRANSFORMS, local_dict={"t": t})
        want = sympy.Rational(slope.numerator, slope.denominator) * t + sympy.Rational(
            start.numerator, start.denominator
        )
        if sympy.simplify(expr - want) != 0:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad
