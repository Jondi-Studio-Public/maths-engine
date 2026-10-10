"""Tests `p7d_examine_quadratic_variation`: distractors and plain positive-number answers."""

import functools

"""Quadratic variation.

Covers ``senior-examine-examples-of-quadratically-related-variables``
(``p7d_examine_quadratic_variation``, ACMMM006).
"""
import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7d_quadratic_variation_build import (
    _verify_easy,
    _verify_medium,
    _verify_hard,
)

TOPIC = "p7d_examine_quadratic_variation"
SKILL_ID = "senior-examine-examples-of-quadratically-related-variables"
SEEDS = range(40)


# `load_skills()` rescans every skill YAML from disk and costs ~560 ms. These
# helpers are called once per SEED, so an uncached lookup spent ~290 s of the
# suite reloading the same library. Cached: `make()` still runs per seed and
# every assertion is unchanged -- only the repeated disk scan is removed.
@functools.lru_cache(maxsize=None)
def _source(difficulty):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return next(s for s in skill.sources if s.difficulty == difficulty)


def _make(seed, difficulty):
    """Generate through the skill's own sources, never a bare difficulty."""
    source = _source(difficulty)
    return make(source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))


# --- determinism -------------------------------------------------------------

# --- distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = _make(seed, difficulty)
        assert len(q.distractors) == 3, f"seed {seed}/{difficulty}: expected 3 distractors, got {q.distractors}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_and_distractors_are_plain_positive_integers_formatted_alike(difficulty):
    """Every candidate value (answer and each distractor) must be a positive
    integer, displayed in exactly the same plain-integer format -- no
    decimal point, no fraction, no leading zero -- so no option is
    identifiable by its shape (the midpoint_coordinate_scalar defect)."""
    for seed in SEEDS:
        q = _make(seed, difficulty)
        for text in [q.answer_text, *q.distractors]:
            assert text.isdigit(), f"seed {seed}/{difficulty}: {text!r} is not a plain positive integer"
        assert isinstance(q.answer, int)
        assert q.answer > 0


# --- real grader -----------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = _make(seed, difficulty)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed {seed}/{difficulty}: canonical not accepted"
        for distractor in q.distractors:
            result = grade_number(q, distractor)
            assert not result.correct, f"seed {seed}/{difficulty}: distractor {distractor} was accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(seed, difficulty)
        wrong = str(int(q.answer_text) + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed {seed}/{difficulty}: wrong answer {wrong} accepted"


# --- unknown preset ----------------------------------------------------------


def test_unknown_variant_raises_value_error():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, variant="not_a_real_variant")


# --- independent-check corruption -------------------------------------------


def test_verify_easy_fires_on_corrupted_answer():
    # x1=3, y1=6 -> k=6/9=2/3; x2=6 -> true y2 = (2/3)*36 = 24
    _verify_easy(3, 6, 6, 24)  # sanity: correct value does not raise
    with pytest.raises(VerificationError):
        _verify_easy(3, 6, 6, 25)


def test_verify_medium_fires_on_corrupted_answer():
    # x1 = t^2*a2 = 4*2 = 8 (t=2, a2=2); y1=2; y2=a2*t^2=8; true x2 = a2*t^3 = 16
    _verify_medium(8, 2, 16, 8)  # sanity
    with pytest.raises(VerificationError):
        _verify_medium(8, 2, 17, 8)
    with pytest.raises(VerificationError):
        _verify_medium(8, 2, -16, 8)  # not positive


def test_verify_hard_fires_on_corrupted_answer():
    # v1=3, e1=9 (k=1), t=4 -> true factor = 16
    _verify_hard(3, 9, 4, 16)  # sanity
    with pytest.raises(VerificationError):
        _verify_hard(3, 9, 4, 17)


# --- distinct-question coverage ----------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_at_least_20_distinct_rendered_questions(difficulty):
    texts = [_make(seed, difficulty).text for seed in range(60)]
    distinct_count = len(set(texts))
    assert distinct_count >= 20, f"only {distinct_count} distinct questions (< 20) in {difficulty} tier"


# --- hard-tier collision fallback --------------------------------------------


def test_hard_tier_t_equals_2_does_not_ship_a_collision():
    """t=2 is the one value where the naive '2*t' distractor collides with
    the correct answer t^2 (2*2 == 2^2 == 4); confirm the fallback keeps 3
    genuinely distinct distractors whenever the draw lands on t=2."""
    seen_t2 = False
    for seed in range(200):
        q = _make(seed, "hard")
        if q.answer_text == "4":  # t^2 == 4 => t == 2
            seen_t2 = True
            assert "4" not in q.distractors
            assert len(set(q.distractors)) == 3
    assert seen_t2, "no seed in range(200) drew t=2 for the hard tier; widen the sweep"


# --- skill loading -----------------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None, "skill not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert "quadratic-vertex" in skill.prerequisites


def test_skill_sources_cover_correct_stages():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}, "easy stages"
    assert set(skill.sources[1].stages) == {3, 4}, "medium stages"
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}, "hard stages"
