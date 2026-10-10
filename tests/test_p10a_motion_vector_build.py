"""Vector calculus, motion in a line and a plane.

Skill 1: senior-determine-equations-of-motion-of-a-particle-travelling
(ACMSM114, topic p10a_vector_motion_line_accel, answer: number).
Skill 2: senior-apply-vector-calculus-to-motion-in-a-plane (ACMSM115, topic
p10a_projectile_motion_vector, answer: number). See the generator module
docstring for the i/j-notation grading-defect workaround (both skills ask
for exactly one named axis's component, never a full vector expression) and
Skill 2's projectile-only scope.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.text_quality import find_text_defects

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC_1 = "p10a_vector_motion_line_accel"
SKILL_1 = "senior-determine-equations-of-motion-of-a-particle-travelling"
TOPIC_2 = "p10a_projectile_motion_vector"
SKILL_2 = "senior-apply-vector-calculus-to-motion-in-a-plane"

TOPICS = (TOPIC_1, TOPIC_2)


# --- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene -------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed}: distractor {d!r} accepted"
        # Every answer here is a bare integer (no decimal point), so
        # grade_number requires an EXACT match -- a +1 offset is always
        # outside tolerance regardless of the answer's own magnitude.
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_bare_integer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int), f"seed={seed}: answer {q.answer!r} is not a plain int"
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"


# --- Formatted alike (every option is a bare, possibly-signed integer) -----


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_options_all_bare_integer_shaped(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        for opt in [q.answer_text, *q.distractors]:
            assert opt.lstrip("-").isdigit(), f"seed={seed}: {opt!r} not shaped like a bare integer"


# --- Distinct rendered questions --------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- No text-quality flags (mechanical prose defects) -----------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_no_text_quality_flags(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert not find_text_defects(q), f"seed={seed}: {find_text_defects(q)} on {q.text!r}"


# --- Vectors genuinely visible: d/v(0)/r(0)/a(t) stated in i/j form ---------


@pytest.mark.parametrize("tier", TIERS)
def test_skill_1_states_direction_vector_and_never_asks_for_a_vector(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        assert "i" in q.text and "j" in q.text, f"seed={seed}: no i/j notation: {q.text!r}"
        assert "vector d = " in q.text
        # answer: number -- never a full vector expression typed by the learner
        assert isinstance(q.answer, int)


@pytest.mark.parametrize("tier", TIERS)
def test_skill_2_states_velocity_and_acceleration_vectors(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        assert "v(0) = " in q.text
        assert "a(t) = " in q.text
        assert isinstance(q.answer, int)


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Misconceptions are exercisable where the skill claims they are ----------


def test_skill_1_forgot_position_offset_never_offered_at_easy():
    """forgot-position-offset collides with the answer at easy (r0 = (0,0)),
    so the generator must never even attempt to offer it there -- confirmed
    by the candidate-count discipline (easy has 2 candidates, medium/hard
    have 3+)."""
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="easy")
        assert len(q.distractors) <= 2, f"seed={seed}: easy tier offering >2 distractors"


def test_skill_1_power_rule_no_divide_only_at_hard():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            assert len(q.distractors) <= 3, f"seed={seed}/{tier}: unexpectedly >=4 distractors"


def test_skill_2_forgot_position_offset_only_at_hard_position_questions():
    """forgot-position-offset collides with the answer at medium (r0 =
    (0,0)) and never applies to a velocity question (easy) -- confirmed
    indirectly via the candidate-count discipline for medium/easy."""
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            assert len(q.distractors) <= 3, f"seed={seed}/{tier}: unexpectedly >=4 distractors"


def test_skill_2_sign_error_only_on_y_axis_questions():
    """sign-error-on-gravity only applies to a question touching the
    y-axis: an i-axis question must never carry a candidate count that
    could only arise from it being offered."""
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            if "j-component" not in q.text:
                # i-axis: at most wrong-axis + dropped-initial (+ forgot-offset
                # at hard/position) = at most 3, never the 4th (sign-error).
                assert len(q.distractors) <= 3, (
                    f"seed={seed}/{tier}: i-axis question has >3 distractors: {q.distractors}"
                )


# --- Plausibility guard (Skill 2): RED then GREEN ---------------------------


def test_plausibility_guard_is_not_vacuous_on_a_widened_draw():
    """Prove the [-60, 60] velocity / [-100, 100] position bound is a real
    constraint, not a vacuous one, by widening T to 20 and vy0 to 50 (well
    outside the actual pools) and confirming the bound FAILS there."""
    g = 10
    violations = 0
    for t in range(1, 21):
        for vy0 in range(1, 51):
            vy_t = vy0 - g * t
            y_t = 9 + vy0 * t - Fraction(g, 2) * t**2  # y0 = 9, the real pool's own max magnitude
            if not (-60 <= vy_t <= 60) or not (-100 <= y_t <= 100):
                violations += 1
    assert violations > 0, "widened draw did not violate the bound -- the guard below would be vacuous"


@pytest.mark.parametrize("tier", TIERS)
def test_plausibility_guard_holds_on_the_actual_pools(tier):
    """Every velocity component the actual pools can produce stays within
    [-60, 60] and every position component within [-100, 100] -- computed
    directly from the specified pools (vx0, vy0, x0, y0, T), not merely
    sampled via `make()`, so every combination the generator can draw is
    covered, not just 60 seeds' worth."""
    g = 10
    vx0_pool = list(range(-9, 0)) + list(range(1, 10))
    vy0_pool = list(range(1, 10))
    offset_pool = list(range(-9, 0)) + list(range(1, 10))
    t_pool = (1, 2, 3)
    offsets = offset_pool if tier == "hard" else [0]
    for t in t_pool:
        for vx0 in vx0_pool:
            for vy0 in vy0_pool:
                vx_t = vx0
                vy_t = vy0 - g * t
                assert -60 <= vx_t <= 60
                assert -60 <= vy_t <= 60
        for x0 in offsets:
            for y0 in offsets:
                for vx0 in (vx0_pool[0], vx0_pool[-1]):
                    for vy0 in (vy0_pool[0], vy0_pool[-1]):
                        for t in t_pool:
                            x_t = x0 + vx0 * t
                            y_t = y0 + vy0 * t - Fraction(g, 2) * t**2
                            assert -100 <= x_t <= 100, f"x({t})={x_t} out of bounds"
                            assert -100 <= y_t <= 100, f"y({t})={y_t} out of bounds"


@pytest.mark.parametrize("tier", TIERS)
def test_plausibility_guard_holds_through_the_generator_itself(tier):
    """The pool enumeration above proves the bound holds for every value the
    POOLS can produce; this sweeps `make()` itself (sweeping
    every seed/tier) so a future drift between the generator's
    actual draws and this test's transcribed pools cannot go unnoticed."""
    bound = 60 if tier == "easy" else 100
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        assert -bound <= q.answer <= bound, f"seed={seed}/{tier}: answer {q.answer} outside [-{bound}, {bound}]"


# --- Skill loading -----------------------------------------------------


def test_skill_1_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_1), None)
    assert skill is not None, f"{SKILL_1} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["vector-addition"]
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC_1
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}


def test_skill_2_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_2), None)
    assert skill is not None, f"{SKILL_2} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["vector-addition"]
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC_2
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}
