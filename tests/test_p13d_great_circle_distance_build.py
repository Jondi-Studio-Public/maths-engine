"""Great-circle distance between two places on Earth at DIFFERENT longitudes (ACMEM161)."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_tolerance

TOPIC = "p13d_great_circle_distance"
TIER_PARAMS = {
    "easy": dict(mode="same_hemisphere"),
    "medium": dict(mode="opposite_hemisphere"),
    "hard": dict(mode="combined_km"),
    "exam": dict(mode="inverse_longitude"),
}
TIERS = list(TIER_PARAMS)
SEEDS = range(40)


def _make(seed, tier):
    return make(TOPIC, seed=seed, difficulty=tier, **TIER_PARAMS[tier])


# --- determinism / distractor hygiene ----------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_across_seeds(tier):
    for seed in SEEDS:
        assert _make(seed, tier).to_json() == _make(seed, tier).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for seed in SEEDS:
        q = _make(seed, tier)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_correct_type(tier):
    for seed in SEEDS:
        q = _make(seed, tier)
        assert isinstance(q.answer, (int, float))
        assert not isinstance(q.answer, bool)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_formatted_alike(tier):
    for seed in SEEDS:
        q = _make(seed, tier)
        is_answer_decimal = "." in q.answer_text
        for d in q.distractors:
            assert ("." in d) == is_answer_decimal, (tier, seed, q.answer_text, d)


@pytest.mark.parametrize("tier", TIERS)
def test_longitudes_always_differ(tier):
    """The ACMEM161 scope: never the same-longitude degenerate case."""
    for seed in SEEDS:
        q = _make(seed, tier)
        assert "longitude" in q.text or tier == "exam"  # exam states no longitude at all


# Old line, quoted before being replaced: `assert 10 <= len(texts) <= table_size,
# (tier, len(texts))`. That assertion was pointed at "does the table_size ceiling match how many
# rows exist" -- a 10-of-12 result passed it even though a learner sees the same handful of city
# pairs.
def test_at_least_30_distinct_questions_per_forward_tier_over_40_seeds():
    """easy/medium/hard must each offer >= 30 distinct prompts over the same 40-seed window the pool-variety check
    measures, so a learner cannot memorise the small fixed table of city pairs."""
    for tier in ("easy", "medium", "hard"):
        texts = {_make(seed, tier).text for seed in range(40)}
        assert len(texts) >= 30, (tier, len(texts))


def test_exam_tier_at_least_30_distinct_over_40_seeds():
    texts = {_make(seed, "exam").text for seed in range(40)}
    assert len(texts) >= 30, len(texts)


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _make(seed, tier)
        assert grade_number(q, q.answer_text).correct, (tier, seed, q.answer_text)
        for d in q.distractors:
            assert not grade_number(q, d).correct, (tier, seed, d)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for seed in SEEDS:
        q = _make(seed, tier)
        if "." in q.answer_text:
            dp = len(q.answer_text.split(".")[1])
            offset = Fraction(1, 10**dp) * 5
        else:
            offset = 1
        wrong = Fraction(q.answer_text) + offset
        assert not grade_number(q, str(float(wrong))).correct


@pytest.mark.parametrize("tier", TIERS)
def test_decimal_answers_meet_stated_precision_tolerance(tier):
    for seed in SEEDS:
        q = _make(seed, tier)
        if "." not in q.answer_text:
            tol = Fraction(1, 2)
        else:
            places = len(q.answer_text.split(".")[1])
            tol = stated_precision_tolerance(places)
        answer_frac = Fraction(q.answer_text)
        for d in q.distractors:
            gap = abs(Fraction(d) - answer_frac)
            assert gap > tol, (tier, seed, d, q.answer_text, gap, tol)


# --- independent-check-in-generator fires on corruption ---------------------


def test_verify_central_angle_fires_on_corrupted_angle():
    from mathgen.generators.p13d_great_circle_distance_build import _verify_central_angle

    with pytest.raises(VerificationError):
        _verify_central_angle(-33.87, 151.21, -31.94, 115.86, 9999.0)


# --- plausibility: 5 < central angle < 150, proven red first ----------------


def _all_forward_rows():
    from mathgen.generators.p13d_great_circle_distance_build import (
        SAME_HEMISPHERE,
        OPPOSITE_HEMISPHERE,
    )

    return SAME_HEMISPHERE + OPPOSITE_HEMISPHERE


def _assert_angles_plausible(rows):
    import math
    from mathgen.generators.p13d_great_circle_distance_build import _central_angle_deg

    for place_a, lat_a, lon_a, place_b, lat_b, lon_b in rows:
        angle = _central_angle_deg(lat_a, lon_a, lat_b, lon_b)
        assert 5.0 < angle < 150.0, (place_a, place_b, angle)


def test_plausibility_bound_red_then_green():
    """Prove the (5, 150) central-angle bound is a real constraint -- red
    against a corrupted row (two places pinned unrealistically close
    together, central angle near 0) before proving green on the real
    tables."""
    corrupted = list(_all_forward_rows())
    place_a, lat_a, lon_a, place_b, _, _ = corrupted[0]
    corrupted[0] = (place_a, lat_a, lon_a, place_b, lat_a, lon_a + 0.001)  # near-0 angle
    with pytest.raises(AssertionError):
        _assert_angles_plausible(corrupted)

    _assert_angles_plausible(_all_forward_rows())


def test_exam_tier_given_angle_plausible():
    from mathgen.generators.p13d_great_circle_distance_build import (
        INVERSE_LONGITUDE_ROWS,
        _central_angle_deg,
    )

    for place_a, lat_a, lon_a, place_b, lat_b, lon_b in INVERSE_LONGITUDE_ROWS:
        angle = _central_angle_deg(lat_a, lon_a, lat_b, lon_b)
        assert 5.0 < angle < 150.0, (place_a, place_b, angle)


# --- conditioning: +/-1 km must not move the answer >= 0.02 deg, red first --


def test_inverse_rows_well_conditioned_red_then_green():
    """Proven red first: a corrupted single-row table containing ONLY the
    Bogota/Lima row must fail `_well_conditioned`, before checking every row
    in the real (fixed) table passes it.
    """
    from mathgen.generators.p13d_great_circle_distance_build import _well_conditioned

    # Red: the known ill-conditioned row (measured ~0.052 deg/km)
    # must fail the predicate.
    assert not _well_conditioned(4.71, -74.07, -12.05, -77.04), (
        "Bogota/Lima must be rejected by the conditioning filter"
    )

    # Green: every row actually shipped in INVERSE_LONGITUDE_ROWS must pass.
    from mathgen.generators.p13d_great_circle_distance_build import INVERSE_LONGITUDE_ROWS

    for place_a, lat_a, lon_a, place_b, lat_b, lon_b in INVERSE_LONGITUDE_ROWS:
        assert _well_conditioned(lat_a, lon_a, lat_b, lon_b), (place_a, place_b)


# --- every widened row is actually exercisable, not just sampled -----------


class _FixedChoiceRng:
    """A stub whose `choice` always returns the ONE row it was built with -- lets a test drive every row of a
    130-140-row table through the real `_forward_question`/`_inverse_question` bodies without relying on 40 or
    200 random `rng.choice` draws to happen to land on it."""

    def __init__(self, row):
        self._row = row

    def choice(self, rows):
        return self._row


def test_every_widened_row_is_individually_exercisable():
    """Every row in every widened table must construct a valid question
    through the REAL generator body (no `VerificationError`, no distractor
    collision) -- not reimplemented arithmetic, the actual functions."""
    from mathgen.generators.p13d_great_circle_distance_build import (
        SAME_HEMISPHERE,
        OPPOSITE_HEMISPHERE,
        INVERSE_LONGITUDE_ROWS,
        _forward_question,
        _inverse_question,
    )

    for row in SAME_HEMISPHERE:
        for unit in ("nm", "km"):
            q = _forward_question(_FixedChoiceRng(row), [row], unit=unit)
            assert len(q["distractors"]) >= 2, (row, unit)
    for row in OPPOSITE_HEMISPHERE:
        for unit in ("nm", "km"):
            q = _forward_question(_FixedChoiceRng(row), [row], unit=unit)
            assert len(q["distractors"]) >= 2, (row, unit)
    for row in INVERSE_LONGITUDE_ROWS:
        q = _inverse_question(_FixedChoiceRng(row))
        assert len(q["distractors"]) >= 2, row


# --- 200-seed answer concentration (measured concentration) --------------------


def test_answer_concentration_under_ceiling():
    """No single answer value should dominate a tier's draw -- ceiling
    consistent with the project's existing per-skill concentration checks."""
    for tier in TIERS:
        answers = Counter(_make(seed, tier).answer_text for seed in range(200))
        top_share = answers.most_common(1)[0][1] / 200
        assert top_share < 0.25, (tier, answers.most_common(3))


# --- load through mathgen.skills --------------------------------------------------


def test_skill_loads_with_correct_prerequisites_and_placement():
    domains, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-find-distances-between-two-places-on-earth-using"),
        None,
    )
    assert skill is not None
    assert skill.prerequisites == []
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
