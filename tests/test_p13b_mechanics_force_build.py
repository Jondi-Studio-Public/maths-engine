"""Momentum/force/resultant-force/ action-and-reaction; constant and non-constant force.

Skill A: senior-examine-momentum-force-resultant-force-action-and-reaction
(topic ``p13b_momentum_force_reaction``, answer: number).
Skill B: senior-consider-constant-and-non-constant-force
(topic ``p13b_constant_nonconstant_force``, answer: number).
"""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC_A = "p13b_momentum_force_reaction"
SKILL_A = "senior-examine-momentum-force-resultant-force-action-and-reaction"
TOPIC_B = "p13b_constant_nonconstant_force"
SKILL_B = "senior-consider-constant-and-non-constant-force"

_grade_number = GRADERS["number"]


# --- Determinism -------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1, f"seed={seed}: no distractors"


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = _grade_number(q, q.answer_text)
        assert result.correct, f"{topic}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_number(q, d)
            assert not result.correct, f"{topic}/{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        # answer_text is a plain integer (no decimal point) here, so
        # grade_number requires an EXACT match; +1 is always wrong.
        wrong = str(int(q.answer_text) + 1)
        result = _grade_number(q, wrong)
        assert not result.correct, f"{topic}/{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Answer kind: always a plain integer -----------------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_are_plain_integers(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert re.fullmatch(r"-?\d+", q.answer_text), (
            f"{topic}/{tier} seed={seed}: answer_text not a plain integer: {q.answer_text!r}"
        )
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+", d), f"{topic}/{tier} seed={seed}: distractor not a plain integer: {d!r}"


# Answer-concentration measure, 200 seeds/tier: no single answer above 25% of draws on any tier.


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("tier", TIERS)
def test_answer_not_concentrated_at_200_seeds(topic, tier):
    from collections import Counter

    answers = [make(topic, seed=seed, difficulty=tier).answer_text for seed in range(200)]
    counts = Counter(answers)
    top_value, top_count = counts.most_common(1)[0]
    share = top_count / len(answers)
    assert share <= 0.25, f"{topic}/{tier}: answer {top_value!r} is {share:.2%} of 200 draws (> 25%)"


# --- Skill A: hard tier's friction opposes the correct (net-applied) sign --


_HARD_A_RE = re.compile(
    r"(\d+) N (?:to the right|pushing right).*?"
    r"(\d+) N (?:to the left|pushing left).*?"
    r"magnitude (\d+) N"
)


def test_skill_a_hard_friction_smaller_than_net_applied():
    """Friction is always strictly smaller in magnitude than the net of the
    two applied forces, so its opposing direction is never ambiguous and the
    resulting acceleration is never zero."""
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty="hard")
        m = _HARD_A_RE.search(q.text)
        assert m is not None, f"seed={seed}: could not parse hard text: {q.text!r}"
        f1, f2, friction = (int(g) for g in m.groups())
        net_applied = abs(f1 - f2)
        assert friction < net_applied, f"seed={seed}: friction {friction} not smaller than net applied {net_applied}"
        assert int(q.answer_text) != 0, f"seed={seed}: hard acceleration is zero"
        checked += 1
    assert checked == len(SEEDS)


# --- Skill A: medium tier never has F1 == F2 (would collapse the resultant
# and the sign-flip distractor together) ------------------------------------


_MEDIUM_A_RE = re.compile(
    r"(\d+) N (?:to the right|acting right|pushing right).*?(\d+) N (?:to the left|acting left|pushing left)"
)


def test_skill_a_medium_forces_never_equal():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty="medium")
        m = _MEDIUM_A_RE.search(q.text)
        assert m is not None, f"seed={seed}: could not parse medium text: {q.text!r}"
        f1, f2 = int(m.group(1)), int(m.group(2))
        assert f1 != f2, f"seed={seed}: F1 == F2 == {f1}"
        checked += 1
    assert checked == len(SEEDS)


# --- Skill B: boundary against solve-ode-separable-poly: never a bare
# "y(x) = ... + C" shape ------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_skill_b_never_asks_for_y_of_x_with_free_constant(tier):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=tier)
        assert "y(x)" not in q.text, f"{tier} seed={seed}: y(x) phrasing leaked in: {q.text!r}"
        assert " + C" not in q.text and q.text.strip()[-1:] != "C", (
            f"{tier} seed={seed}: bare constant C leaked in: {q.text!r}"
        )
        assert re.fullmatch(r"-?\d+", q.answer_text), f"{tier} seed={seed}: answer not a bare number: {q.answer_text!r}"


# --- Skill loading -----------------------------------------------------


def test_skill_a_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_A), None)
    assert skill is not None, f"{SKILL_A} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["solve-ode-exponential"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC_A
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


def _pool_check(topic, tier, min_distinct=20, max_share=0.15, seeds=40):
    from collections import Counter

    answers = [make(topic, seed=seed, difficulty=tier).answer_text for seed in range(seeds)]
    counts = Counter(answers)
    distinct = len(counts)
    top_value, top_count = counts.most_common(1)[0]
    share = top_count / len(answers)
    assert distinct >= min_distinct, (
        f"{topic}/{tier}: only {distinct} distinct answers over {seeds} seeds (need >= {min_distinct})"
    )
    assert share <= max_share, (
        f"{topic}/{tier}: answer {top_value!r} is {share:.2%} of {seeds} draws (> {max_share:.0%})"
    )


def test_pool_momentum_hard_not_concentrated():
    _pool_check(TOPIC_A, "hard")


def test_pool_momentum_exam_not_concentrated():
    _pool_check(TOPIC_A, "exam")


def test_pool_constant_nonconstant_easy_not_concentrated():
    _pool_check(TOPIC_B, "easy")


def test_pool_constant_nonconstant_medium_not_concentrated():
    _pool_check(TOPIC_B, "medium")


def test_pool_constant_nonconstant_exam_not_concentrated():
    _pool_check(TOPIC_B, "exam")


def test_skill_b_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_B), None)
    assert skill is not None, f"{SKILL_B} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["solve-ode-exponential"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC_B
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
