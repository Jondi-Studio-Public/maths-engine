"""Velocity/acceleration to displacement, motion in a line.

Skill 1: senior-determine-displacement-given-velocity-in-linear-motion-problems
(ACMMM123, topic p10a_velocity_to_displacement_line, answer: expression).
Skill 2: senior-determine-positions-given-acceleration-and-initial-values-of
(ACMMM135, topic p10a_acceleration_to_position_line, answer: expression).
See the generator module docstring for the pinned-design misconception-scope
fix (sign-error-family misconceptions collapse to the true answer whenever
the stated IC instant is 0, and skill 1's misread-worded-sign collapses
whenever s0 >= 0).
"""

import re

import pytest
from sympy import symbols, sympify

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC_1 = "p10a_velocity_to_displacement_line"
SKILL_1 = "senior-determine-displacement-given-velocity-in-linear-motion-problems"
TOPIC_2 = "p10a_acceleration_to_position_line"
SKILL_2 = "senior-determine-positions-given-acceleration-and-initial-values-of"

_grade_expression = GRADERS["expression"]


# --- Determinism -------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = _grade_expression(q, q.answer_text)
        assert result.correct, f"{topic}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_expression(q, d)
            assert not result.correct, f"{topic}/{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        wrong = q.answer_text + " + 1"
        result = _grade_expression(q, wrong)
        assert not result.correct, f"{topic}/{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent("t + 1", "t + 2", "should fail")


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Answer/distractor shape: always a polynomial-in-t expression ----------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_are_all_t_expressions(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert "t" in q.answer_text, f"{topic}/{tier} seed={seed}: answer_text has no t: {q.answer_text!r}"
        for d in q.distractors:
            assert "t" in d, f"{topic}/{tier} seed={seed}: distractor has no t: {d!r}"


# --- Never a negative/implied-negative time in the question text ----------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_never_states_a_negative_time(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert "t ≥ 0" in q.text or "t >= 0" in q.text, (
            f"{topic}/{tier} seed={seed}: question does not state t >= 0: {q.text!r}"
        )


# --- Skill 1 hard tier: worded IC sign convention --------------------------


_HARD_WORDED_RE = re.compile(r"the particle is (?P<mag>\d+) m to the (?P<side>left|right) of the origin")


def test_skill1_hard_worded_sign_matches_left_negative_right_positive():
    """left of the origin <=> s0 < 0; right of the origin <=> s0 > 0. Parses
    the worded phrase back to a signed s0 and checks it against S(a),
    evaluated from `answer_text` -- not merely that the phrase occurs."""
    saw_left = saw_right = False
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="hard")
        m = _HARD_WORDED_RE.search(q.text)
        assert m is not None, f"seed={seed}: hard tier did not render the worded IC form: {q.text!r}"
        mag = int(m.group("mag"))
        side = m.group("side")
        saw_left = saw_left or side == "left"
        saw_right = saw_right or side == "right"
        s0 = -mag if side == "left" else mag

        a_match = re.search(r"At time t = (-?\d+)", q.text)
        a = int(a_match.group(1))
        s_expr = sympify(q.answer_text.replace("^", "**"), locals={"t": symbols("t")})
        s_at_a = s_expr.subs(symbols("t"), a)
        assert s_at_a == s0, f"seed={seed}: worded side={side!r} implies s0={s0}, but S({a}) = {s_at_a}"
    assert saw_left, "no hard-tier seed in 60 rendered 'left of the origin'"
    assert saw_right, "no hard-tier seed in 60 rendered 'right of the origin'"


# --- Skill loading -----------------------------------------------------


def test_skill_1_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_1), None)
    assert skill is not None, f"{SKILL_1} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["integrate-polynomial", "definite-integral"]
    assert len(skill.sources) == 4  # easy, medium, hard and exam
    for source in skill.sources[:3]:  # the exam source is covered by this skill's tests/test_exd_x_* file
        assert source.topic == TOPIC_1
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}


def test_skill_2_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_2), None)
    assert skill is not None, f"{SKILL_2} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == [
        "senior-recognise-acceleration-as-the-second-derivative-of-position",
        "senior-determine-displacement-given-velocity-in-linear-motion-problems",
    ]
    assert len(skill.sources) == 4  # easy, medium, hard and exam
    for source in skill.sources[:3]:  # the exam source is covered by this skill's tests/test_exd_x_* file
        assert source.topic == TOPIC_2
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
