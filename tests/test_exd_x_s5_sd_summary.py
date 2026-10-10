"""Tests for the exam source of standard-deviation-from-data.

The key is re-derived from the rendered prompt: either from n, the sum and the sum of squares, or by recovering
the missing value from the stated mean, then applying the definition with exact fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_s5_sd_summary  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


SKILLS = [("standard-deviation-from-data", "exd_x_s5_sd_summary")]
HARD = {"standard-deviation-from-data": ("standard_deviation_from_data", "compare_sets")}
ANSWER_KIND = {"standard-deviation-from-data": "number"}


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


T = "exd_x_s5_sd_summary"


def _oracle(text):
    body = text.replace("2 decimal places", "")
    nums = [Fraction(v) for v in re.findall(r"\d+(?:\.\d+)?", body)]
    if "mean" in body:
        n, mean, listed = nums[0], nums[1], nums[2:-1]
        assert nums[-1] == n and len(listed) == n - 1
        values = listed + [n * mean - sum(listed)]
        assert values[-1] > 0
        mu = sum(values) / n
        var = sum((v - mu) ** 2 for v in values) / (n - 1)
    else:
        n, s, q = nums
        var = (q - s * s / n) / (n - 1)
    return math.sqrt(float(var))


def test_key_matches_independent_derivation_over_400_seeds():
    bad = []
    for s in range(400):
        q = _exam(T, s)
        if f"{_oracle(q.text):.2f}" != q.answer_text:
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_both_forms_occur():
    texts = [_exam(T, s).text for s in range(60)]
    assert any("mean" in t for t in texts) and any("mean" not in t for t in texts)


def test_wrong_values_rejected():
    grader = GRADERS["number"]
    q = _exam(T, 1)
    v = float(q.answer_text)
    for wrong in (v + 0.01, v - 0.01, v * v, v * math.sqrt(8 / 9)):
        assert not grader(q, f"{wrong:.2f}").correct, wrong
