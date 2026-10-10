"""Tests for the exam source of senior-use-the-approximate-confidence-interval-left-widehat-p.

The key is re-derived from the rendered prompt alone: centre and half-width of the stated interval, scaled by the
ratio of the z values, rounded half-up to one decimal place with exact fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_s5_ci_rescale  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


SKILLS = [("senior-use-the-approximate-confidence-interval-left-widehat-p", "exd_x_s5_ci_rescale")]
HARD = {"senior-use-the-approximate-confidence-interval-left-widehat-p": ("p8c_confidence_interval_proportion", "hard")}
ANSWER_KIND = {"senior-use-the-approximate-confidence-interval-left-widehat-p": "pair"}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    return next(s for s in skills if s.id == skill_id).sources


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_skill_declares_exam_source_last(skill_id, topic):
    src = _skill_sources(skill_id)[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (topic, "exam", ["exam"], {"tier": "exam"})


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_no_text_defects(skill_id, topic):
    for seed in range(60):
        q = _exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_prompt_variety_and_new_shapes_over_hard_tier(skill_id, topic):
    hard_topic, hard_difficulty = HARD[skill_id]
    exam = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard = {prompt_shape(make(hard_topic, seed=s, difficulty=hard_difficulty)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_determinism(skill_id, topic):
    for seed in (0, 1, 7, 99):
        assert _exam(topic, seed).to_json() == _exam(topic, seed).to_json()


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_grader_accepts_key_and_rejects_every_distractor(skill_id, topic):
    grader = GRADERS[ANSWER_KIND[skill_id]]
    for seed in range(150):
        q = _exam(topic, seed)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3, (seed, q.distractors)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d, q.answer_text)


T = "exd_x_s5_ci_rescale"
Z = {90: Fraction(1645, 1000), 95: Fraction(196, 100), 99: Fraction(2576, 1000)}


def _half_up_tenth(x):
    return Fraction(math.floor(x * 10 + Fraction(1, 2)), 10)


def _oracle(text):
    levels = [int(v) for v in re.findall(r"(?<![\d.])(\d+)%", text)]
    bounds = [Fraction(v) for v in re.findall(r"(\d+\.\d)%", text)]
    old, new = levels[0], levels[-1]
    lo, hi = bounds[0], bounds[1]
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    new_half = half * Z[new] / Z[old]
    return (
        f"lower = {float(_half_up_tenth(centre - new_half)):.1f}, "
        f"upper = {float(_half_up_tenth(centre + new_half)):.1f}"
    )


def test_key_matches_independent_derivation_over_400_seeds():
    bad = []
    for s in range(400):
        q = _exam(T, s)
        if _oracle(q.text) != q.answer_text:
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_wrong_bounds_are_rejected():
    grader = GRADERS["pair"]
    q = _exam(T, 3)
    lo, hi = q.answer["lower"], q.answer["upper"]
    for wrong in (
        f"lower = {lo + 0.2:.1f}, upper = {hi:.1f}",
        f"lower = {lo:.1f}, upper = {hi - 0.3:.1f}",
        f"lower = {hi:.1f}, upper = {lo:.1f}",
    ):
        assert not grader(q, wrong).correct, wrong
