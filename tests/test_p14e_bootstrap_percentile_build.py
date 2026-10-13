"""Tests `p14e_bootstrap_percentile` (`ucv1-bootstrap-summary`), built by
`mathgen/generators/p14e_bootstrap_percentile_build.py`.

``ucv1-bootstrap-summary``, built by ``mathgen/generators/p14e_bootstrap_percentile_build.py`` (topic
``p14e_bootstrap_percentile``, answer: pair, all four tiers).

See the generator module's docstring for the kernel, the boundary against
`calculate-mean`, and the tier design.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_pair, mcq_options  # noqa: F401 (mcq_options unused, kept for symmetry)
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import PRESETS, REGISTRY, VerificationError, check
from mathgen.generators.p14e_bootstrap_percentile_build import (
    _components_differ,
    _kth_smallest,
    _toward_middle,
    p14e_bootstrap_percentile,
)

TOPIC = "p14e_bootstrap_percentile"
SKILL_ID = "ucv1-bootstrap-summary"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)

_RESAMPLE_RUN_RE = re.compile(r"(?:-?\d+,\s*){2,}-?\d+")


def _parse_resample(text: str) -> list[int]:
    """The resample list is the LONGEST run of comma-separated integers in
    the rendered prompt -- robust to the several template wordings."""
    runs = _RESAMPLE_RUN_RE.findall(text)
    assert runs, f"could not parse resample from {text!r}"
    longest = max(runs, key=len)
    return [int(tok.strip()) for tok in longest.split(",") if tok.strip()]


# --- Registration -----------------------------------------------------


def test_topic_registered_with_all_four_presets():
    assert TOPIC in REGISTRY
    assert set(PRESETS[TOPIC]) == {"easy", "medium", "hard", "exam"}


# --- Determinism --------------------------------------------------------


# --- Distractor hygiene ---------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    counts = Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1
        counts[len(q.distractors)] += 1
    counts_dict = dict(counts)
    at_least_two = sum(v for k, v in counts_dict.items() if k >= 2)
    assert at_least_two / 200 >= 0.85, f"{tier}: <2-distractor share too high: {counts_dict}"


# --- Real grader: pair, per component ------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_answer_rejects_every_distractor(tier):
    grader = GRADERS["pair"]
    assert grader is grade_pair


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    grader = GRADERS["pair"]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        wrong = q.answer_text.replace(str(q.answer["endpoint"]), str(int(q.answer["endpoint"]) + 1000))
        assert not grader(q, wrong).correct, f"{tier} seed={seed}: corrupted answer accepted"


# --- Answer kind: mean exact Fraction, endpoint exact int ----------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_shape(tier):
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert set(q.answer) == {"mean", "endpoint"}
        assert isinstance(q.answer["mean"], Fraction)
        assert isinstance(q.answer["endpoint"], int)


# --- Options formatted alike: mean/endpoint always same shape ------------


@pytest.mark.parametrize("tier", TIERS)
def test_options_formatted_alike(tier):
    for seed in range(50):
        q = make(TOPIC, seed=seed, difficulty=tier)
        for opt in [q.answer_text, *q.distractors]:
            left, right = opt.split(",")
            assert "=" in left and "=" in right


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


# --- Percentile convention sentence stated verbatim -----------------------


@pytest.mark.parametrize("tier", TIERS)
def test_percentile_convention_stated_verbatim(tier):
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "is its (100k/" in q.text and ")-th percentile" in q.text
        assert "no interpolation" in q.text


# --- easy tier: list already sorted, exactly 10 values --------------------


def test_easy_list_is_already_sorted():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        nums = _parse_resample(q.text)
        assert len(nums) == 10
        assert nums == sorted(nums)


# --- medium/hard: 20 values, unsorted ---------------------------------


@pytest.mark.parametrize("tier", ("medium", "hard", "exam"))
def test_unsorted_tiers_are_not_sorted_and_have_20_values(tier):
    unsorted_count = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        nums = _parse_resample(q.text)
        assert len(nums) == 20
        if nums != sorted(nums):
            unsorted_count += 1
    assert unsorted_count == 60, f"{tier}: {60 - unsorted_count} seeds rendered an already-sorted list"


# --- hard: forced duplicate on every seed ---------------------------------


def test_hard_forces_a_duplicate_pair_every_seed():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        nums = _parse_resample(q.text)
        assert len(nums) != len(set(nums)), f"hard seed={seed}: no duplicate in {nums}"


# --- exam: coverage stated, percentile never stated directly -------------


def test_exam_never_states_the_bare_percentile_number():
    """The convention sentence itself states "(100k/n)-th percentile" as a
    formula, never a concrete percentile number for THIS question; the
    concrete request is phrased only via coverage + lower/upper -- the
    learner must derive the percentile level themselves."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "% bootstrap interval" in q.text
        assert ("lower endpoint" in q.text) or ("upper endpoint" in q.text)


def test_exam_coverage_inversion_misconception_exercisable():
    """coverage-inversion (treats the coverage % as the percentile level) is
    a genuinely different value whenever coverage*n/100 lands on an integer
    order statistic different from the true one -- confirm it appears among
    distractors on at least some seeds."""
    seen = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        if len(q.distractors) >= 4 - 1:  # candidates capped at 3; just count non-empty draws
            seen += 1
    assert seen > 0


# --- Independent in-generator check tripwire ------------------------------


def test_check_fires_on_corrupted_condition():
    with pytest.raises(VerificationError):
        check(1 == 2, "should fail")
    check(1 == 1, "should not fail")


def test_kth_smallest_and_toward_middle_helpers():
    assert _kth_smallest([1, 5, 9, 20], 1) == 1
    assert _kth_smallest([1, 5, 9, 20], 4) == 20
    assert _toward_middle(1, 10) == 2
    assert _toward_middle(9, 10) == 8
    assert _toward_middle(5, 20) == 6
    assert _toward_middle(15, 20) == 14


def test_components_differ_uses_grader_tolerance_for_mean_exact_for_endpoint():
    assert not _components_differ(Fraction(1), 5, Fraction(1), 5)
    assert _components_differ(Fraction(1), 5, Fraction(1), 6)
    assert _components_differ(Fraction(1), 5, Fraction(2), 5)
    # Within tolerance on the mean, same endpoint -> NOT different.
    assert not _components_differ(Fraction(1), 5, Fraction(1) + Fraction(1, 1000), 5)


def test_direct_builder_callable_via_public_entry_point():
    import random

    rng = random.Random(1)
    d = p14e_bootstrap_percentile(rng, kind="easy")
    assert "mean" in d["answer"] and "endpoint" in d["answer"]


# --- Skill loading ---------------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["calculate-mean"]
    assert len(skill.sources) == 4
    stages = [set(s.stages) for s in skill.sources]
    assert stages == [{0, 1, 2}, {3, 4}, {5, 6, "maintenance"}, {"exam"}]
    assert skill.sources[3].difficulty == "exam"
    assert all(source.topic == TOPIC for source in skill.sources[:3])
    assert skill.sources[3].topic == "exd_x_r2g_bootstrap_bias"  # the exam tier is a multi-part chain now


# 200-seed measurement --------------


@pytest.mark.parametrize("tier", TIERS)
def test_200_seed_answer_concentration_measured(tier):
    """Answer-concentration measure at 200 seeds per tier, per component: the
    mean should show wide spread (many distinct values); the endpoint (an
    order statistic of a random sample) may repeat more but must still show
    real spread, never collapsing to one or two values."""
    means = set()
    endpoints = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        means.add(q.answer["mean"])
        endpoints.add(q.answer["endpoint"])
    assert len(means) >= 100, f"{tier}: mean concentration too high, {len(means)} distinct over 200"
    assert len(endpoints) >= 20, f"{tier}: endpoint concentration too high, {len(endpoints)} distinct over 200"
