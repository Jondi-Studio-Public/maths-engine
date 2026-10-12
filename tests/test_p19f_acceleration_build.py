"""Acceleration as the second derivative of position with respect to time.

Skill: senior-recognise-acceleration-as-the-second-derivative-of-position
(topic ``p19f_acceleration_second_derivative``, answer: number).
"""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_acceleration_second_derivative"
SKILL = "senior-recognise-acceleration-as-the-second-derivative-of-position"

_grade_number = GRADERS["number"]


# --- Determinism ---------------------------------------------------------


# --- Distractor hygiene ----------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_number(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_number(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # answer_text is an exact integer (no decimal point) -> exact match required.
        wrong = str(int(q.answer_text) + 7)
        result = _grade_number(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


# --- Answer kind: always a plain integer ------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_a_plain_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert re.fullmatch(r"-?\d+", q.answer_text), (
            f"{tier} seed={seed}: answer_text not a plain integer: {q.answer_text!r}"
        )
        assert isinstance(q.answer, int), f"{tier} seed={seed}: answer not an int: {q.answer!r}"


# --- Answer and distractors formatted alike (all plain integers) -----------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+", d), f"{tier} seed={seed}: distractor {d!r} not a plain integer"


# --- No stray "1 seconds" ----------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_singular_count_plural_defect(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "1 seconds" not in q.text, f"{tier} seed={seed}: '1 seconds' defect: {q.text!r}"


# --- Never bare "differentiate" or a symbolic second-derivative request ----
# (the `second-derivative` boundary: this skill always evaluates numerically
# at a stated time in a motion context, never asks for f''(x) symbolically.)


@pytest.mark.parametrize("tier", TIERS)
def test_never_bare_differentiate_or_symbolic_request(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "differentiate" not in q.text.lower(), f"{tier} seed={seed}: bare 'differentiate' in text: {q.text!r}"
        assert "s(t) = " in q.text and "metres" in q.text, (
            f"{tier} seed={seed}: missing explicit motion/position context: {q.text!r}"
        )


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["second-derivative", "senior-interpret-the-derivative-as-the-instantaneous-rate-of"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


# Realistic magnitude -----------------


# Medium added at integration: the rework bounded hard and exam only, which
# left medium the loudest tier (a toy car at 338 m/s^2; 136/200 over 50).
@pytest.mark.parametrize("tier", ("medium", "hard", "exam"))
def test_acceleration_magnitude_is_realistic_200_seeds(tier):
    import statistics

    vals = [abs(make(TOPIC, seed=seed, difficulty=tier).answer) for seed in range(200)]
    assert max(vals) <= 50, f"{tier}: max |a| = {max(vals)}, expected <= 50"
    over_1000 = sum(1 for v in vals if v > 1000)
    assert over_1000 == 0, f"{tier}: {over_1000}/200 seeds have |a| > 1000"
    print(f"{tier}: median |a| = {statistics.median(vals)}, max = {max(vals)}")


# --- Top-key concentration (15% bar, >=200 seeds/tier) ---------


@pytest.mark.parametrize("tier", TIERS)
def test_top_key_under_15_percent_over_200_seeds(tier):
    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer_text for seed in range(200))
    top_key, top_n = counts.most_common(1)[0]
    assert top_n / 200 < 0.15, f"{tier}: top answer {top_key!r} is {top_n}/200 = {top_n / 200:.3f}, >= 15%"
