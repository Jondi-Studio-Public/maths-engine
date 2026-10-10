"""Least-squares-line normal-equation summary-statistics hard tier."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

TOPIC = "least_squares_line"
PATTERN = re.compile(
    r"A data set of (\d+) points has sum x = (-?\d+), sum y = (-?\d+), sum xy = (-?\d+), sum x\^2 = (-?\d+)\."
)


def _summary(seed):
    return make(TOPIC, seed=seed, difficulty="hard", kind="summary")


def test_summary_key_matches_recomputation_from_rendered_text():
    for seed in range(60):
        question = _summary(seed)
        match = PATTERN.search(question.text)
        assert match, question.text
        n, s_x, s_y, s_xy, s_xx = (int(v) for v in match.groups())
        assert "Find the least-squares line y = mx + b." in question.text
        assert "(" not in question.text  # no data points shown
        m = Fraction(n * s_xy - s_x * s_y, n * s_xx - s_x**2)
        b = (s_y - m * s_x) / n
        assert m.denominator == 1 and b.denominator == 1
        assert question.answer == {"m": int(m), "b": int(b)}
        assert question.answer_text == f"m = {int(m)}, b = {int(b)}"


def test_summary_x_sets_are_not_always_zero_to_three():
    sets = set()
    for seed in range(60):
        n, s_x, *_ = (int(v) for v in PATTERN.search(_summary(seed).text).groups())
        sets.add((n, s_x))
    assert len(sets) > 3
    assert {4, 5} <= {n for n, _ in sets}


def test_default_path_unchanged_and_summary_text_differs():
    for seed in range(20):
        default = make(TOPIC, seed=seed, difficulty="hard")
        explicit = make(TOPIC, seed=seed, difficulty="hard", kind=None)
        assert default.text == explicit.text
        assert default.answer_text == explicit.answer_text
        assert default.text.startswith("For the data points (0, ")
        assert default.text != _summary(seed).text


def test_summary_distractors_clear_the_real_pair_grader():
    grader = GRADERS["pair"]
    for seed in range(200):
        question = _summary(seed)
        assert grader(question, question.answer_text).correct, seed
        assert len(question.distractors) == 3
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)
            m, b = (int(v) for v in re.findall(r"-?\d+", distractor))
            assert abs(m - question.answer["m"]) >= 1 or abs(b - question.answer["b"]) >= 1


def test_summary_prompt_shape_differs_from_easy_and_medium():
    shape = prompt_shape(_summary(1))
    for difficulty in ("easy", "medium"):
        assert prompt_shape(make(TOPIC, seed=1, difficulty=difficulty)) != shape
