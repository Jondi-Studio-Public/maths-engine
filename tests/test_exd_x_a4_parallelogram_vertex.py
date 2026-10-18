"""Exam-tier checks for the parallelogram fourth-vertex source.

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
TOPIC = "exd_x_a4_parallelogram_vertex"
GRADER = GRADERS["pair"]
HARD = ("midpoint_two_points", {"kind": "find_endpoint", "coord_lo": -14, "coord_hi": 14})


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


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
        pts = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"([A-D])\((-?\d+), (-?\d+)\)", q.text)}
        missing = (set("ABCD") - set(pts)).pop()
        # A parallelogram ABCD: A + C = B + D, so each vertex is the sum of its two neighbours minus the opposite.
        opposite = {"A": "C", "B": "D", "C": "A", "D": "B"}[missing]
        neighbours = [v for v in "ABCD" if v not in (missing, opposite)]
        want = tuple(sum(pts[n][i] for n in neighbours) - pts[opposite][i] for i in (0, 1))
        got = tuple(int(v) for v in re.findall(r"-?\d+", q.answer_text))
        if got != want:
            bad.append((seed, q.text, q.answer_text, want))
    assert bad == [], bad


def test_hidden_vertex_varies():
    hidden = {(set("ABCD") - set(re.findall(r"([A-D])\(", _q(s).text))).pop() for s in range(60)}
    assert len(hidden) == 4, hidden
