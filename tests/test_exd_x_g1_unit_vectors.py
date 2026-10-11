"""Tests for the exam tier of `senior-define-and-use-unit-vectors-and-the-perpendicular`.

Unit vector perpendicular to AB, and unit vector in the direction of a p + b q.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g1_unit_vectors"
GRADER = GRADERS["fraction"]
_SEEDS = range(320)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make(
        "define_and_use_unit_vectors_and_the_perpendicular", seed=seed, difficulty="hard", variant="perpendicular"
    )


_NUM = r"-?\d+"
_TRIPLE = re.compile(r"\((-?\d+), (-?\d+), (-?\d+)\)")
_PAIR = re.compile(r"\((-?\d+), (-?\d+)\)")


def _triples(text):
    return [tuple(int(x) for x in m.groups()) for m in _TRIPLE.finditer(text)]


def _pairs(text):
    return [tuple(int(x) for x in m.groups()) for m in _PAIR.finditer(text)]


def _terms(expr):
    """Coefficients of the named vectors in an expression such as '3p - q'."""
    out = {}
    for sign, digits, name in re.findall(r"([+-]?)\s*(\d*)([a-z])", expr):
        out[name] = (-1 if sign == "-" else 1) * (int(digits) if digits else 1)
    return out


def _kind(text):
    return "points" if "A(" in text or "A is the point" in text else "combination"


def _expected(text):
    axis = re.search(r"coefficient of (i|j)", text).group(1)
    pts = _pairs(text)
    if _kind(text) == "points":
        (ax, ay), (bx, by) = pts
        x, y = bx - ax, by - ay
        if "anticlockwise" in text:
            wx, wy = -y, x
        else:
            wx, wy = y, -x
    else:
        (px, py), (qx, qy) = pts
        expr = re.search(r"(?:direction of|The vector) ([^.]*?)(?: has a unit|\.)", text).group(1)
        t = _terms(expr)
        wx, wy = t["p"] * px + t["q"] * qx, t["p"] * py + t["q"] * qy
    m2 = wx * wx + wy * wy
    m = int(round(m2**0.5))
    assert m * m == m2, text
    return Fraction(wx if axis == "i" else wy, m)


def _agrees(q, expected):
    return Fraction(q.answer["p"], q.answer["q"]) == expected


def test_no_text_defects_over_60_seeds():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes, "the exam tier adds no prompt shape beyond the hard tier"


def test_deterministic_for_a_seed():
    for seed in (0, 1, 2, 17, 999):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_only_the_exam_preset_exists():
    with pytest.raises(KeyError):
        make(TOPIC, seed=0, difficulty="easy")


def test_key_matches_independent_derivation_from_the_rendered_prompt():
    bad = []
    for seed in _SEEDS:
        q = _exam(seed)
        expected = _expected(q.text)
        if not _agrees(q, expected):
            bad.append((seed, q.text, q.answer_text, expected))
    assert bad == [], bad[:5]


def test_both_question_kinds_occur():
    kinds = {_kind(_exam(s).text) for s in _SEEDS}
    assert len(kinds) >= 2, kinds


def test_grader_accepts_key_and_rejects_every_distractor():
    bad = []
    for seed in range(160):
        q = _exam(seed)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) >= 2, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.distractors)
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_rejects_deliberately_wrong_values():
    for seed in range(40):
        q = _exam(seed)
        for wrong in ["0", "", "1", "-1", "2/3", "1/2"]:
            if wrong == q.answer_text:
                continue
            assert not GRADER(q, wrong).correct, (seed, wrong, q.answer_text)
