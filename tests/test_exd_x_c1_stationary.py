"""Tests for the exam-tier source `exd_x_c1_stationary` (stationary points of (x - p)^m (x - q)^n)."""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape
from tests.exd_x_c1_support import function_in_prompt, shapes, x

TOPIC = "exd_x_c1_stationary"
_VALUE_RE = re.compile(r"-?\d+")


def _values(text: str) -> list[int]:
    return sorted(int(v) for v in _VALUE_RE.findall(text))


def test_prompt_variety_and_new_shape_over_hard_tier():
    exam = shapes(TOPIC, difficulty="exam")
    hard = shapes("differentiation", difficulty="hard", kinds=("stationary_points",))
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_key_matches_independent_solution_of_rendered_prompt_over_300_seeds():
    bad = []
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="exam")
        f = function_in_prompt(q.text)
        roots = sorted(sympy.solve(sympy.diff(f, x), x))
        assert all(r.is_integer for r in roots), (seed, roots)
        if [int(r) for r in roots] != q.answer or _values(q.answer_text) != q.answer:
            bad.append((seed, q.text, roots, q.answer, q.answer_text))
    assert bad == [], bad[:3]


def test_three_distinct_stationary_points_every_seed():
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert len(set(q.answer)) == 3, (seed, q.answer)


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    bad = []
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="exam")
        if not GRADERS["list"](q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) >= 2, (seed, q.distractors)
        for d in q.distractors:
            if GRADERS["list"](q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]


def test_text_is_clean_and_generation_is_deterministic():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == make(TOPIC, seed=seed, difficulty="exam").to_json()
        assert prompt_shape(q)
