"""Tests for the n-th roots of unity (ACMSM087) and n-th roots of a general
complex number (ACMSM088) skills."""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import grade_number, grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7c_nth_roots_build import (
    _check_root,
    _unity_check_angle,
    _unity_negative_real_count,
    _verify_unity_count,
)

UNITY_TOPIC = "nth_roots_of_unity"
COMPLEX_TOPIC = "nth_roots_of_complex_number"
UNITY_TIERS = ["easy", "medium", "hard"]
COMPLEX_TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


# --- determinism -------------------------------------------------------


@pytest.mark.parametrize("difficulty", UNITY_TIERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unity_determinism(difficulty, seed):
    q1 = make(UNITY_TOPIC, seed=seed, difficulty=difficulty)
    q2 = make(UNITY_TOPIC, seed=seed, difficulty=difficulty)
    assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", COMPLEX_TIERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_complex_determinism(difficulty, seed):
    q1 = make(COMPLEX_TOPIC, seed=seed, difficulty=difficulty)
    q2 = make(COMPLEX_TOPIC, seed=seed, difficulty=difficulty)
    assert q1.to_json() == q2.to_json()


# --- distractor hygiene + real grader, both topics ----------------------


@pytest.mark.parametrize("difficulty", UNITY_TIERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unity_distractors_and_grader(difficulty, seed):
    q = make(UNITY_TOPIC, seed=seed, difficulty=difficulty)
    assert q.distractors, "no distractors"
    assert len(q.distractors) == len(set(q.distractors))
    assert q.answer_text not in q.distractors
    assert isinstance(q.answer, int)

    result = grade_number(q, q.answer_text)
    assert result.correct, f"grader rejected canonical: {q.answer_text}"
    for d in q.distractors:
        result = grade_number(q, d)
        assert not result.correct, f"grader accepted distractor: {d}"
    # a deliberately wrong answer, distinct from both canonical and every distractor
    wrong = q.answer + 1000
    while str(wrong) in q.distractors or wrong == q.answer:
        wrong += 1000
    assert not grade_number(q, str(wrong)).correct


@pytest.mark.parametrize("difficulty", COMPLEX_TIERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_complex_distractors_and_grader(difficulty, seed):
    q = make(COMPLEX_TOPIC, seed=seed, difficulty=difficulty)
    assert q.distractors, "no distractors"
    assert len(q.distractors) == len(set(q.distractors))
    assert q.answer_text not in q.distractors
    assert isinstance(q.answer, dict) and set(q.answer) == {"r", "theta"}
    for v in q.answer.values():
        assert float(v) == int(v), f"answer component {v} is not an exact integer"

    result = grade_pair(q, q.answer_text)
    assert result.correct, f"grader rejected canonical: {q.answer_text}"
    for d in q.distractors:
        result = grade_pair(q, d)
        assert not result.correct, f"grader accepted distractor: {d}"
    # a deliberately wrong answer
    wrong_text = f"r = {q.answer['r'] + 1000}, theta = {q.answer['theta']}"
    assert not grade_pair(q, wrong_text).correct


def test_complex_grader_rejects_swapped_order():
    """grade_pair is positional: a swapped (theta, r) must be rejected even
    though both values appear somewhere in the text."""
    q = make(COMPLEX_TOPIC, seed=0, difficulty="easy")
    r, theta = q.answer["r"], q.answer["theta"]
    if r == theta:
        q = make(COMPLEX_TOPIC, seed=1, difficulty="easy")
        r, theta = q.answer["r"], q.answer["theta"]
    assert r != theta, "need a seed where r != theta to prove order matters"
    swapped = f"r = {theta}, theta = {r}"
    assert not grade_pair(q, swapped).correct


# --- independent check fires on a corrupted answer, exercised directly --


def test_unity_angle_check_fires():
    _unity_check_angle(6, 2, 120)  # correct: 360*2/6 = 120
    with pytest.raises(VerificationError):
        _unity_check_angle(6, 2, 121)


def test_unity_count_check_fires():
    _verify_unity_count(6, 3)  # hand-checked: n=6 -> 3
    with pytest.raises(VerificationError):
        _verify_unity_count(6, 4)


def test_complex_root_check_fires():
    _check_root(4, 90, 2, 45, 2)  # w = 4*cis(90), sqrt is 2*cis(45)
    with pytest.raises(VerificationError):
        _check_root(4, 90, 2, 50, 2)


# --- hand-checked hard-tier count formula --------------------------------


def test_unity_hard_count_hand_checked_cases():
    assert _unity_negative_real_count(4) == 1
    assert _unity_negative_real_count(6) == 3


# --- skill 1 never touches w != 1; skill 2 never emits w = 1 -------------


def test_unity_never_mentions_a_general_w():
    """ACMSM087 has no w at all: no distractor or working line implies a
    general (non-unity) complex number is involved."""
    for difficulty in UNITY_TIERS:
        for seed in range(60):
            q = make(UNITY_TOPIC, seed=seed, difficulty=difficulty)
            fields = [q.text, *q.working, *q.distractors]
            for f in fields:
                assert "w =" not in f and " w " not in f, f"unexpected w framing: {f!r}"


def test_complex_never_emits_w_equal_to_one():
    """No generated w in ACMSM088 ever equals 1 (r=1 and theta=0 simultaneously),
    across a full 60-seed sweep, every tier."""
    from mathgen.generators.p7c_nth_roots_build import _COMPLEX_BASES

    assert all(b >= 2 for b in _COMPLEX_BASES)
    for difficulty in COMPLEX_TIERS:
        for seed in range(60):
            q = make(COMPLEX_TOPIC, seed=seed, difficulty=difficulty)
            # r_w is never surfaced directly in `answer`, but every base is >= 2
            # and n >= 2, so r_w = base**n >= 4 always -- w can never be 1.
            assert q.answer["r"] >= 2, "root modulus below the base floor"


# distinct question texts per tier ------------------------------------ ACMSM087's
# per-tier n sets cap the true question-text pool: easy (n in {3,4,5,6}, k in [0,n)) has exactly 18
# possible (n,k) pairs; hard has exactly 12. Both are documented pinned pool sizes, not defects.
# Medium's k is sampled widely outside [0,n) (the k=n+2/2n-1/-1 cases are "e.g." examples, not an
# exhaustive list), so its pool is large.
UNITY_POOL = {"easy": 15, "medium": 20, "hard": 10}
COMPLEX_POOL = {"easy": 20, "medium": 20, "hard": 20}


@pytest.mark.parametrize("difficulty", UNITY_TIERS)
def test_unity_distinct_questions_per_tier(difficulty):
    texts = {make(UNITY_TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(texts) >= UNITY_POOL[difficulty], f"{difficulty}: only {len(texts)} distinct texts over 60 seeds"


@pytest.mark.parametrize("difficulty", COMPLEX_TIERS)
def test_complex_distinct_questions_per_tier(difficulty):
    texts = {make(COMPLEX_TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(texts) >= COMPLEX_POOL[difficulty], f"{difficulty}: only {len(texts)} distinct texts over 60 seeds"


# --- answer/distractor shape parity ---------------------------------------


def test_unity_options_formatted_alike():
    for difficulty in UNITY_TIERS:
        for seed in range(60):
            q = make(UNITY_TOPIC, seed=seed, difficulty=difficulty)
            has_degree = q.answer_text.endswith("°")
            for d in q.distractors:
                assert d.endswith("°") == has_degree, f"{q.answer_text!r} vs {d!r}"


def test_complex_options_formatted_alike():
    pattern = re.compile(r"^r = -?\d+, theta = -?\d+$")
    for difficulty in COMPLEX_TIERS:
        for seed in range(60):
            q = make(COMPLEX_TOPIC, seed=seed, difficulty=difficulty)
            assert pattern.match(q.answer_text), q.answer_text
            for d in q.distractors:
                assert pattern.match(d), d


# --- skills load through the API ------------------------------------------


def test_skills_load_through_api():
    _domains, skills = load_skills()

    unity = next((s for s in skills if s.id == "senior-determine-and-examine-the-n-th-roots-of"), None)
    assert unity is not None
    assert unity.answer == "number"
    assert unity.prerequisites == ["convert-metric-length"]
    assert unity.placement_eligible is False
    assert unity.domain == "measurement"

    complex_skill = next((s for s in skills if s.id == "senior-determine-and-examine-the-n-th-roots-of-acmsm088"), None)
    assert complex_skill is not None
    assert complex_skill.answer == "pair"
    assert complex_skill.prerequisites == []
    assert complex_skill.placement_eligible is False
    assert complex_skill.domain == "senior-mathematics"


def test_skill_sources_cover_all_stages():
    _domains, skills = load_skills()
    for skill_id in (
        "senior-determine-and-examine-the-n-th-roots-of",
        "senior-determine-and-examine-the-n-th-roots-of-acmsm088",
    ):
        skill = next(s for s in skills if s.id == skill_id)
        stages_covered = set()
        for source in skill.sources:
            stages_covered.update(source.stages)
        # stages 0-6 plus "maintenance"
        assert {0, 1, 2, 3, 4, 5, 6, "maintenance"} <= stages_covered
