"""Tests for the exam source of `senior-examine-the-relationship-between-the-angle-of-inclination`
(topic `exd_x_f5_incl`): the angle between two lines."""

from __future__ import annotations

import re

import pytest
from sympy import Eq, N, Symbol, atan, diff, pi, solve
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f5_incl"
_GRADER = GRADERS["expression"]
X, Y = Symbol("x"), Symbol("y")
_TR = standard_transformations + (implicit_multiplication_application,)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("angle_of_inclination_gradient", seed=seed, variant="from_points")


_LINE = re.compile(r"Line \d(?: is |: )(.*?)\.(?= Line| Find)")
_POINTS = re.compile(r"the line through \((-?\d+), (-?\d+)\) and \((-?\d+), (-?\d+)\)")


def _gradient(desc):
    m = _POINTS.fullmatch(desc)
    if m:
        x1, y1, x2, y2 = map(int, m.groups())
        return N(parse_expr(str(y2 - y1)) / parse_expr(str(x2 - x1)), 30)
    lhs, rhs = desc.split(" = ")
    eq = parse_expr(lhs, transformations=_TR) - parse_expr(rhs, transformations=_TR)
    y_of_x = solve(Eq(eq, 0), Y)[0]
    return N(diff(y_of_x, X), 30)


def _independent(text):
    descs = _LINE.findall(text)
    assert len(descs) == 2, text
    angles = [N(atan(_gradient(d)) * 180 / pi, 30) % 180 for d in descs]
    gap = abs(angles[0] - angles[1])
    acute = min(gap, 180 - gap)
    want = re.search(r"Find the (acute|obtuse) angle", text).group(1)
    value = acute if want == "acute" else 180 - acute
    assert abs(value - round(float(value))) < 1e-9
    return round(float(value))


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        if _independent(q.text) != int(q.answer_text):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    bad = []
    for seed in range(120):
        q = _exam(seed)
        if not _GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) == 3
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]


def test_prompt_asks_for_the_bare_number_of_degrees():
    """The expression grader rejects a degree sign or word, so the prompt must say to leave them off."""
    for seed in range(60):
        q = _exam(seed)
        assert q.text.endswith("Give the number of degrees only."), q.text
        assert _GRADER(q, q.answer_text).correct
