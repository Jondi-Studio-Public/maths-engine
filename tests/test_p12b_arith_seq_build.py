"""Arithmetic-sequence discrete-linear models --
senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear (Methods, topic p12b_arith_seq_methods)
and senior-use-arithmetic-sequences-to-model-and-analyse-practical (General, topic p12b_arith_seq_general).
See the generator module docstring for the construction and boundary this pair exists to keep.
"""

from __future__ import annotations

import collections
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

METHODS_TOPIC = "p12b_arith_seq_methods"
METHODS_SKILL = "senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear"
GENERAL_TOPIC = "p12b_arith_seq_general"
GENERAL_SKILL = "senior-use-arithmetic-sequences-to-model-and-analyse-practical"

TOPICS = (METHODS_TOPIC, GENERAL_TOPIC)

# The boundary this pair exists to hold against the neighbouring
# simple-interest skill (never asks for "interest" itself).
_FORBIDDEN_RE = re.compile(r"\binterest\b", re.IGNORECASE)


# --- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Boundary: never asks for the interest amount itself -------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_never_mentions_interest(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        blobs = [q.text, q.latex, q.answer_text, q.answer_latex, *q.working, *q.distractors]
        for blob in blobs:
            assert not _FORBIDDEN_RE.search(str(blob)), (
                f"topic={topic} seed={seed} tier={tier}: forbidden 'interest' in {blob!r}"
            )


# --- Distractor hygiene ------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_a_plain_integer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert d.lstrip("-").isdigit(), (
                f"topic={topic} seed={seed} tier={tier}: distractor {d!r} not a plain integer"
            )


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed} tier={tier}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed} tier={tier}: distractor {d!r} accepted"
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed} tier={tier}: deliberately wrong {wrong!r} accepted"


# --- Answer kind --------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_bare_integer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"
        assert "," not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a comma"


# --- Distinct rendered questions ----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Independent-check tripwire ------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Skill loading -------------------------------------------------------


def test_methods_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == METHODS_SKILL), None)
    assert skill is not None, f"{METHODS_SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == METHODS_TOPIC
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


def test_general_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == GENERAL_SKILL), None)
    assert skill is not None, f"{GENERAL_SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == GENERAL_TOPIC
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


# Answer-concentration sanity --------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_distribution_is_not_grossly_concentrated(topic, tier):
    seeds = range(400)
    answers = collections.Counter()
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        answers[q.answer_text] += 1
    total = sum(answers.values())
    top_text, top_n = answers.most_common(1)[0]
    share = top_n / total
    assert share < 0.25, f"{topic}/{tier}: top answer {top_text!r} is {share:.0%} of {total} draws"
