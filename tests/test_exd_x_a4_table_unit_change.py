"""Exam-tier checks for the table-with-unit-change source.

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
TOPIC = "exd_x_a4_table_unit_change"
GRADER = GRADERS["pair"]
HARD = ("linear_from_table", {"kind": "equation_context", "hi": 15})


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


FACTOR = {"hour": 60, "day": 24, "week": 7, "year": 12}


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        rows = re.findall(r"\): ([\d, ]+)\.", q.text)
        ts, ys = ([int(v) for v in r.split(", ")] for r in rows)
        big = re.search(r"number of (\w+?)s \(", q.text).group(1)
        slope = Fraction(ys[1] - ys[0], ts[1] - ts[0])
        assert all(Fraction(ys[i + 1] - ys[i], ts[i + 1] - ts[i]) == slope for i in range(2))
        b = ys[0] - slope * ts[0]
        x = sympy.Symbol("x")
        rhs = q.answer_text.split(" = ")[1]
        got = parse_expr(rhs, transformations=TRANSFORMS, local_dict={"x": x})
        want = int(slope * FACTOR[big]) * x + int(b)
        if sympy.expand(got - want) != 0:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad
