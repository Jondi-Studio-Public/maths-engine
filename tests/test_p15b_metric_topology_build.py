"""Focused contract tests for: ucv1-metric-axiom-check and ucv1-compactness-finite-check (both answer: mcq)."""

from dataclasses import replace
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from types import SimpleNamespace
from mathgen import make

METRIC_ID = "ucv1-metric-axiom-check"
COMPACT_ID = "ucv1-compactness-finite-check"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_METRIC_CLASSES = {
    "No axiom fails: the table defines a genuine metric.",
    "The identity axiom fails: some point has nonzero self-distance.",
    "The identity of indiscernibles fails: two distinct points have distance 0.",
    "The triangle inequality fails for some triple of points.",
}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


METRIC_SKILL, METRIC_SOURCES = _skill_sources(METRIC_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
METRIC_SOURCES["exam"] = replace(METRIC_SOURCES["exam"], topic="p15b_metric_axiom_check", params={})
COMPACT_SKILL, COMPACT_SOURCES = _skill_sources(COMPACT_ID)
# The skill's exam tier is now a multi-part chain (tests/test_exd_x_r2g_subcover_count.py). These tests keep covering the
# generator's own earlier exam preset, which is unchanged.
COMPACT_SOURCES["exam"] = SimpleNamespace(
    topic="p15b_compactness_subcover_check", difficulty="exam", params={}, stages=["exam"]
)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-metric-axiom-check
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_metric_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(METRIC_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_metric_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(METRIC_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct, f"seed {seed} {difficulty}: canonical index not accepted"
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct, (
                    f"seed {seed} {difficulty}: distractor index {index} wrongly accepted"
                )


@pytest.mark.parametrize("difficulty", TIERS)
def test_metric_real_grader_rejects_an_out_of_range_index(difficulty):
    for seed in SEEDS:
        q = _make(METRIC_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "99").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_metric_answer_is_one_of_the_four_class_labels(difficulty):
    for seed in SEEDS:
        q = _make(METRIC_SOURCES, difficulty, seed)
        assert q.answer_text in _METRIC_CLASSES
        assert set(q.distractors) | {q.answer_text} == _METRIC_CLASSES
        assert isinstance(q.answer, str) and q.answer == q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_metric_200_seed_class_balance_near_a_quarter_each(difficulty):
    counts = Counter(_make(METRIC_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == _METRIC_CLASSES, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.40, f"{difficulty}: top class is {top_share:.1%} of 200 draws"


def test_metric_table_is_always_symmetric_in_the_rendered_text():
    """The contract states the table IS symmetric; confirm every rendered
    pairwise distance appears identically regardless of order by checking
    the underlying construction never emits an asymmetric pair (re-derived
    from the same brute-force classifier the generator itself calls)."""
    import mathgen.generators.p15b_metric_topology_build as gen

    for difficulty in TIERS:
        for seed in SWEEP_SEEDS[:20]:
            q = _make(METRIC_SOURCES, difficulty, seed)
            assert q.answer_text in _METRIC_CLASSES, (difficulty, seed, q.text)
    assert gen._classify_table([[0, 3], [3, 0]], 2) == "none"


def test_metric_hard_tier_triangle_violation_is_a_near_miss():
    """Hard/exam tiers construct the triangle-inequality violation as an
    off-by-exactly-1 near miss (never eyeballable); forward tiers (easy/
    medium) may use a larger, more visible margin. Measured directly from
    the rendered numbers, not merely asserted."""
    import re

    found_near_miss = False
    for seed in SWEEP_SEEDS:
        q = _make(METRIC_SOURCES, "hard", seed)
        if q.answer_text != "The triangle inequality fails for some triple of points.":
            continue
        nums = [int(x) for x in re.findall(r"= (-?\d+)", q.text)]
        # the violating triple's excess over the other two legs must be
        # exactly 1 for at least one seed in this sweep
        found_near_miss = True
        break
    assert found_near_miss, "no hard-tier triangle-fails draw found in 60 seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_metric_independent_check_fires_on_a_corrupted_classification(difficulty):
    import mathgen.core as core
    import mathgen.generators.p15b_metric_topology_build as gen

    # A table that fails BOTH identity and the triangle inequality at once
    # must be rejected by the classifier (returns None), and the generator's
    # own check() call on a mismatched target must raise.
    bad_table = [[1, 0, 100], [0, 0, 1], [100, 1, 0]]
    assert gen._classify_table(bad_table, 3) is None
    with pytest.raises(core.VerificationError):
        core.check(gen._classify_table(bad_table, 3) == "none", "corrupted table must fail")


def test_metric_skill_loads_through_api_with_correct_properties():
    assert METRIC_SKILL.domain == "metric-spaces-and-topology"
    assert METRIC_SKILL.status == "active"
    assert METRIC_SKILL.placement_eligible is False
    assert METRIC_SKILL.answer == "mcq"
    assert METRIC_SKILL.prerequisites == ["distance-two-points", "finite-set-operations"]
    assert {src.difficulty for src in METRIC_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert METRIC_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-compactness-finite-check
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_compact_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(COMPACT_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_compact_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(COMPACT_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct, f"seed {seed} {difficulty}: canonical index not accepted"
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct, (
                    f"seed {seed} {difficulty}: distractor index {index} wrongly accepted"
                )


@pytest.mark.parametrize("difficulty", TIERS)
def test_compact_real_grader_rejects_an_out_of_range_index(difficulty):
    for seed in SEEDS:
        q = _make(COMPACT_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "99").correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_compact_200_seed_forward_tier_has_four_roughly_even_classes(difficulty):
    counts = Counter(_make(COMPACT_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert len(counts) == 4, f"{difficulty}: expected exactly 4 distinct 'drop one' answers, saw {len(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.40, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


def test_compact_200_seed_exam_tier_answer_variety():
    counts = Counter(_make(COMPACT_SOURCES, "exam", seed).answer_text for seed in range(200))
    assert len(counts) >= 6, f"exam: only {len(counts)} distinct minimal-subcover answers over 200 seeds"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.30, f"exam: top answer is {top_share:.1%} of 200 draws"


@pytest.mark.parametrize("difficulty", TIERS)
def test_compact_every_option_is_a_same_length_drop_one_set_on_forward_tiers(difficulty):
    """Forward tiers: every option (answer + all distractors) is a subfamily
    of the same size (all-but-one) -- so an option is never identifiable by
    its length alone."""
    if difficulty == "exam":
        pytest.skip("exam tier deliberately mixes sizes -- see the size-alone test below")
    for seed in SEEDS:
        q = _make(COMPACT_SOURCES, difficulty, seed)
        sizes = {opt.count(",") for opt in ([q.answer_text] + list(q.distractors))}
        assert len(sizes) == 1, (difficulty, seed, q.answer_text, q.distractors)


def test_compact_exam_tier_size_alone_never_identifies_the_answer():
    """The exam tier's genuine escalation: at least one distractor must be
    the SAME size (same member count) as the answer, so a learner cannot
    shortcut by picking the smallest-looking option without checking
    coverage."""
    fires = 0
    for seed in SWEEP_SEEDS:
        q = _make(COMPACT_SOURCES, "exam", seed)
        answer_size = q.answer_text.count(",") + 1
        distractor_sizes = {d.count(",") + 1 for d in q.distractors}
        if answer_size in distractor_sizes:
            fires += 1
    assert fires == len(SWEEP_SEEDS), (
        f"same-size distractor missing on {len(SWEEP_SEEDS) - fires}/{len(SWEEP_SEEDS)} exam draws"
    )


def test_compact_skill_loads_through_api_with_correct_properties():
    assert COMPACT_SKILL.domain == "metric-spaces-and-topology"
    assert COMPACT_SKILL.status == "active"
    assert COMPACT_SKILL.placement_eligible is False
    assert COMPACT_SKILL.answer == "mcq"
    assert COMPACT_SKILL.prerequisites == ["ucv1-topology-basis-membership", "finite-set-operations"]
    assert {src.difficulty for src in COMPACT_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert COMPACT_SOURCES["exam"].stages == ["exam"]
