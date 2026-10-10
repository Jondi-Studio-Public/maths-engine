"""Tests for the exam source of senior-use-the-notation-p-a-vert-b.

The key is re-derived from the rendered prompt: the intersection is rebuilt from whichever of the union, the
complement or the difference is stated, using exact fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_s5_condprob  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


SKILLS = [("senior-use-the-notation-p-a-vert-b", "exd_x_s5_condprob")]
HARD = {"senior-use-the-notation-p-a-vert-b": ("conditional_probability_notation", "hard")}
ANSWER_KIND = {"senior-use-the-notation-p-a-vert-b": "fraction"}


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


T = "exd_x_s5_condprob"


def _oracle(text):
    a, b, third = (Fraction(v) for v in re.findall(r"\b\d\.\d\d\b", text))
    if "neither" in text:
        inter = a + b - (1 - third)
    elif "without" in text or "does not" in text:
        inter = a - third
    else:
        inter = a + b - third
    return inter / b


def test_key_matches_independent_derivation_over_400_seeds():
    bad = []
    for s in range(400):
        q = _exam(T, s)
        want = _oracle(q.text)
        if (q.answer["p"], q.answer["q"]) != (want.numerator, want.denominator) or q.answer_text != (
            f"{want.numerator}/{want.denominator}"
        ):
            bad.append((s, q.text, q.answer_text, want))
    assert bad == []


def test_all_three_given_facts_occur():
    texts = [_exam(T, s).text for s in range(120)]
    assert any("neither" in t for t in texts)
    assert any("without" in t or "does not" in t for t in texts)
    assert any("neither" not in t and "without" not in t and "does not" not in t for t in texts)


def test_wrong_fractions_rejected():
    grader = GRADERS["fraction"]
    q = _exam(T, 2)
    p, qq = q.answer["p"], q.answer["q"]
    for wrong in (f"{qq}/{p}", f"{p + 1}/{qq}", f"{p}/{qq + 1}"):
        assert not grader(q, wrong).correct, wrong
