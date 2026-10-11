"""Tests for the exam tier of `vector-addition`.

Parallelogram vertex and ratio-division questions; keys are re-derived from the rendered prompt.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g1_vector_addition"
GRADER = GRADERS["list"]
_SEEDS = range(320)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("vectors", seed=seed, difficulty="hard", kinds=("subtract",), hi=15)


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
    return "parallelogram" if "parallelogram" in text else "ratio"


def _n_vector(text):
    """The stated vector n is the last triple written in every prompt."""
    return _triples(text)[-1]


def _expected_vector(text):
    pts = _triples(text)[:-1]
    if _kind(text) == "parallelogram":
        a, b, c = pts
        return [Fraction(a[i] + c[i] - b[i]) for i in range(3)]
    a, b = pts
    m, k = (int(x) for x in re.search(r"(\d+) : (\d+)(?!.*\d : \d)", text).groups())
    return [Fraction(k * a[i] + m * b[i], m + k) for i in range(3)]


def _expected(text):
    n = _n_vector(text)
    return [sum(x * y for x, y in zip(_expected_vector(text), n))]


def _agrees(q, expected):
    return [Fraction(x) for x in q.answer] == expected


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
        for wrong in [
            "(0, 0, 0)",
            "",
            "(1, 2)",
            "(%s)" % ", ".join(str(x + 1) for x in q.answer),
            str(q.answer[0] + 1),
        ]:
            if wrong == q.answer_text:
                continue
            assert not GRADER(q, wrong).correct, (seed, wrong, q.answer_text)


def test_key_is_a_wide_scalar_not_readable_from_the_prompt():
    from collections import Counter

    answers = Counter()
    for seed in range(300):
        q = _exam(seed)
        assert len(q.answer) == 1 and q.answer_text == str(q.answer[0]), (seed, q.answer)
        shown = {abs(int(t)) for t in re.findall(r"-?\d+", q.text)}
        assert abs(q.answer[0]) not in shown, (seed, q.text, q.answer_text)
        answers[q.answer_text] += 1
    assert len(answers) >= 100, len(answers)
    assert max(answers.values()) / 300 <= 0.05, answers.most_common(3)


def test_no_component_or_reordering_of_the_position_vector_is_graded_correct():
    """The `list` grader ignores order; the key is a dot product, so the position vector itself, in any
    order, or any one of its coordinates, must be wrong."""
    import itertools

    bad = []
    for seed in range(150):
        q = _exam(seed)
        vec = [int(x) for x in _expected_vector(q.text)]
        for perm in itertools.permutations(vec):
            if GRADER(q, "(" + ", ".join(str(x) for x in perm) + ")").correct:
                bad.append((seed, "whole vector", perm))
        for x in vec:
            if x != q.answer[0] and GRADER(q, str(x)).correct:
                bad.append((seed, "coordinate", x))
    assert bad == [], bad[:5]
