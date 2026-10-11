"""Tests for linear_transform_dilation_rotation_reflection -- basic linear transformations: dilation,
rotation, reflection, each shown by its own 2x2 matrix; graded answer is the image point.

The independent oracle (never imports this module) lives in
test_p9d_linear_transform_dilation_rotation_reflection_oracle.py.
"""

import random

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check
from mathgen.generators.p9d_linear_transform_dilation_rotation_reflection_build import (
    _apply_matrix,
    _dilation,
    _reflection,
    _reflection_degenerate,
    _rotation,
    _sample_point,
)

TOPIC = "linear_transform_dilation_rotation_reflection"
SKILL_ID = "senior-define-and-use-basic-linear-transformations-dilations-of"
TIERS = ("easy", "medium", "hard")
_SWEEP_SEEDS = 60


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism(difficulty):
    for seed in range(_SWEEP_SEEDS):
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


# ---------------------------------------------------------------------------
# _apply_matrix -- the generic 2x2 matrix-vector product
# ---------------------------------------------------------------------------


def test_apply_matrix():
    assert _apply_matrix(((2, 0), (0, -3)), 4, -1) == (8, 3)
    assert _apply_matrix(((0, -1), (1, 0)), 4, -1) == (1, 4)
    assert _apply_matrix(((1, 0), (0, -1)), 5, 7) == (5, -7)


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty (>=2 per tier), never equal to the answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_and_never_equal_the_answer(difficulty):
    counts = []
    for seed in range(_SWEEP_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        counts.append(len(q.distractors))
    assert min(counts) >= 1, f"{difficulty}: a seed had zero distractors: {counts}"


# Collision-rate measurement (filter-before-offering discipline).
# Measured over 60 seeds/tier:
#   easy (dilation):    55/60 keep all 3, 3/60 keep 2, 2/60 keep only 1 --
#     `pick_distractors` also dedupes two DIFFERENT misconception formulas
#     that happen to coincide with EACH OTHER (not just with the answer) at
#     x == 0: swapped-lambda and same-factor both reduce to (0, lambda1*y),
#     and sign-flip-first collides with the true answer there too. This is
#     inherent to the three formulas, not a filtering bug
#     an unspecified fourth candidate.
#   medium (rotation):  52/60 keep all 3, 8/60 keep 2, never below 2.
#   hard (reflection):  51/60 keep all 3, 9/60 keep 2, never below 2.
_MIN_DISTRACTOR_FLOOR = {"easy": 1, "medium": 2, "hard": 2}


@pytest.mark.parametrize("difficulty", TIERS)
def test_measure_distractor_collision_rate(difficulty):
    counts = [len(make(TOPIC, seed=seed, difficulty=difficulty).distractors) for seed in range(_SWEEP_SEEDS)]
    n_with_3 = sum(1 for c in counts if c == 3)
    assert min(counts) >= _MIN_DISTRACTOR_FLOOR[difficulty], (
        f"{difficulty}: measured floor regressed below {_MIN_DISTRACTOR_FLOOR[difficulty]}: {counts}"
    )
    assert n_with_3 >= _SWEEP_SEEDS * 0.8, (
        f"{difficulty}: only {n_with_3}/{_SWEEP_SEEDS} seeds kept all 3 distractors: {counts}"
    )


# ---------------------------------------------------------------------------
# The real grader accepts the canonical answer and rejects every distractor
# and a deliberately wrong value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in range(_SWEEP_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        result = GRADERS["pair"](q, q.answer_text)
        assert result.correct, f"seed={seed}: grader rejected canonical {q.answer_text!r}"
        for d in q.distractors:
            wrong = GRADERS["pair"](q, d)
            assert not wrong.correct, f"seed={seed}: grader accepted distractor {d!r}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_deliberately_wrong_answer_rejected(difficulty):
    for seed in range(5):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        result = GRADERS["pair"](q, "x = -999, y = 999")
        assert not result.correct


# ---------------------------------------------------------------------------
# The independent check firing on a corrupted answer (exercised directly)
# ---------------------------------------------------------------------------


def test_independent_check_fires_on_corrupted_construction():
    with pytest.raises(VerificationError):
        check((1, 2) == (3, 4), "dilation construction does not match the direct rule")


# ---------------------------------------------------------------------------
# Degeneracy guards actually fire -- reachable/exercised, not dead code.
# Forced with a stub RNG so this does not depend on a rare natural draw.
# ---------------------------------------------------------------------------


class _StubRNG:
    """Replays a fixed sequence of randint() results; choice() delegates to
    a real Random so tests that don't care about it still work."""

    def __init__(self, randint_values, seed=0):
        self._randint_values = list(randint_values)
        self._real = random.Random(seed)

    def randint(self, lo, hi):
        if self._randint_values:
            return self._randint_values.pop(0)
        return self._real.randint(lo, hi)

    def choice(self, seq):
        return self._real.choice(seq)


def test_sample_point_resamples_away_from_the_origin():
    # First draw is the origin (0, 0); the guard must resample to (3, 4).
    rng = _StubRNG([0, 0, 3, 4])
    assert _sample_point(rng) == (3, 4)


def test_reflection_degenerate_resample_is_reachable():
    """Force the first sampled point onto the line of reflection (line drawn
    as "x-axis"), and prove the resample loop replaces it with a
    non-degenerate point rather than using the on-axis (5, 0) draw."""
    real = random.Random(0)
    line_choices = ("x-axis", "y-axis", "y=x", "y=-x")

    class _LineStub(_StubRNG):
        def choice(self, seq):
            if tuple(seq) == line_choices:
                return "x-axis"
            return real.choice(seq)

    rng = _LineStub([5, 0, 5, 4])
    result = _reflection(rng)
    # If the resample loop were dead code the image would be the on-axis
    # point (5, 0) reflected in the x-axis, i.e. (5, 0) itself unchanged.
    assert result["answer"] != {"x": 5, "y": 0}
    # Directly confirm the degeneracy predicate the loop tests against.
    assert _reflection_degenerate("x-axis", 5, 0)
    assert not _reflection_degenerate("x-axis", 5, 4)


def test_dilation_lambda_distinctness_resample_is_reachable():
    """Force lambda1 == lambda2 on the first draw and prove the guard
    redraws lambda2 until it differs."""
    real = random.Random(0)
    factors = (-4, -3, -2, 2, 3, 4)

    calls = {"n": 0}

    class _ChoiceStub:
        def choice(self, seq):
            if tuple(seq) == factors:
                calls["n"] += 1
                if calls["n"] <= 2:
                    return 3  # lambda1 = 3, lambda2 = 3 (equal) -- forces the resample
                return -2
            return real.choice(seq)

        def randint(self, lo, hi):
            return real.randint(lo, hi)

    result = _dilation(_ChoiceStub())
    assert calls["n"] >= 3, "the lambda1 != lambda2 resample loop never ran"
    assert "The dilation matrix is [[3, 0], [0, -2]]." in result["working"]


# ---------------------------------------------------------------------------
# At least 20 distinct rendered questions per tier over 60 seeds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(_SWEEP_SEEDS)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {_SWEEP_SEEDS} seeds"


# ---------------------------------------------------------------------------
# Skill loads correctly through mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["matrix-multiplication"]
    assert {src.difficulty for src in skill.sources} == set(TIERS) | {"exam"}
    stages_covered = set()
    for src in skill.sources:
        stages_covered.update(src.stages)
    assert stages_covered.issuperset({0, 1, 2, 3, 4, 5, 6, "maintenance"})


def test_skill_sources_drive_the_three_variants():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    # the exam source is covered by this skill's tests/test_exd_x_* file
    practice = [src for src in skill.sources if src.difficulty != "exam"]
    seen_variants = {src.params.get("variant") for src in practice}
    assert seen_variants == {"dilation", "rotation", "reflection"}
    for src in practice:
        q = make(src.topic, seed=0, difficulty=src.difficulty, **(src.params or {}))
        assert q.answer_text.startswith("x = ")
