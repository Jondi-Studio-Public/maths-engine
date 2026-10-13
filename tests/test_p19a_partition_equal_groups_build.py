"""Tests for `mathgen/generators/p19a_partition_equal_groups_build.py`: `p19a_partition_equal_groups` (skill
`f10-partition-collections-equal-groups`, AC9M1N03).

This file holds the generator-specific proofs: the in-generator partition check firing
on a corrupted value, skill load, the real-grader battery, distractor/answer
properties, and the neighbour-separation property.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19a_partition_equal_groups_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19a_partition_equal_groups_build as gen

TOPIC = "p19a_partition_equal_groups"
TIERS = ("easy", "medium", "hard")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Shared / by-hand prompt-shape and construction checks
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in [make(TOPIC, seed=s, difficulty=tier) for s in range(30)]:
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(tier):
    shapes = {prompt_shape(q) for q in _all(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all(tier):
        assert len(q.distractors) >= 1, f"{tier} seed {q.seed}: no distractors"
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor == answer: {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_on_85pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"{tier}: only {at_least_two}/200 have >= 2 distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_top_answer_under_15pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top answer {counts.most_common(1)} share {top_share}"


# ============================================================================
# Answer-space contract: a single whole number, no remainders ever, groups
# always the skip-counting number {2, 5, 10} at easy / {2, 5} at medium/hard.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_a_whole_number_string(tier):
    for q in _all(tier, seeds=range(60)):
        assert q.answer_text.isdigit(), f"{tier} seed {q.seed}: {q.answer_text!r} not a bare whole number"
        assert isinstance(q.answer, int)


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_a_whole_number_string_never_negative(tier):
    for q in _all(tier, seeds=range(60)):
        for d in q.distractors:
            assert d.isdigit(), f"{tier} seed {q.seed}: distractor {d!r} not a bare whole number"


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    gen._check_partition(12, 3, 4, "ok")  # 3 * 4 == 12, no exception
    with pytest.raises(VerificationError):
        gen._check_partition(12, 3, 5, "corrupted group size")
    with pytest.raises(VerificationError):
        gen._check_partition(13, 3, 4, "corrupted total")


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(tier):
    for q in _all(tier, seeds=range(30)):
        assert grade_number(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_number(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(15)):
        wrong = str(int(q.answer_text) + 1000)
        assert not grade_number(q, wrong).correct


# ============================================================================
# Medium/hard place-value properties
# ============================================================================


def test_medium_never_needs_unbundling():
    for q in _all("medium", seeds=range(60)):
        assert "one bundle" not in q.text.lower() and "unbundled" not in q.text.lower(), (
            f"medium seed {q.seed}: unexpectedly mentions unbundling: {q.text!r}"
        )


_BUNDLES_LOOSE_TEST_RE = re.compile(r"(\d+)\s+bundles of ten.*?(\d+)\s+loose\s+[a-z]+", re.DOTALL)
_GROUPS_TEST_RE = re.compile(r"into\s+(\d+)\s+(?:equal\s+)?groups")


def test_hard_never_announces_the_divisibility_outcome():
    banned = (
        "does not share evenly",
        "cannot be shared whole",
        "one bundle unbundled",
        "unbundled to share out",
        "so it is unbundled",
    )
    for q in _all("hard", seeds=range(60)):
        lowered = q.text.lower()
        for phrase_ in banned:
            assert phrase_ not in lowered, f"hard seed {q.seed}: announces the outcome: {q.text!r}"


def test_hard_always_offers_the_neutral_open_a_bundle_hint():
    for q in _all("hard", seeds=range(60)):
        lowered = q.text.lower()
        assert "open a bundle" in lowered or "opening a bundle" in lowered, (
            f"hard seed {q.seed}: missing the neutral hint: {q.text!r}"
        )


def test_hard_mixes_evenly_dividing_and_leftover_draws():
    """About one third of hard draws should divide evenly (bundles % groups
    == 0, no regrouping actually needed) and about two thirds should leave
    exactly one bundle over -- computed from the parsed bundles/groups, since
    the prompt itself never states which case applies."""
    seeds = _all("hard", seeds=range(200))
    even = 0
    for q in seeds:
        bl = _BUNDLES_LOOSE_TEST_RE.search(q.text)
        bundles = int(bl.group(1))
        groups = int(_GROUPS_TEST_RE.search(q.text).group(1))
        if bundles % groups == 0:
            even += 1
    share = even / len(seeds)
    assert 0.20 <= share <= 0.45, f"hard evenly-dividing share {share} ({even}/{len(seeds)})"


@pytest.mark.parametrize("tier", ("medium", "hard"))
def test_groups_value_genuinely_varies(tier):
    """Confirm at least 3 distinct groups values appear over 200 seeds."""
    seeds = _all(tier, seeds=range(200))
    groups_seen = set()
    for q in seeds:
        groups_seen.add(int(_GROUPS_TEST_RE.search(q.text).group(1)))
    assert len(groups_seen) >= 3, f"{tier}: only {groups_seen} groups values seen over 200 seeds"


# ============================================================================
# Neighbour separation: never use the arrays skill's fixed "shared equally
# into N equal groups. How many ... in each group?" stem verbatim, and groups
# is always a skip-counting number.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_never_uses_the_bare_arrays_stem(tier):
    for q in _all(tier, seeds=range(60)):
        lowered = q.text.lower()
        assert "count" in lowered or "bundle" in lowered, (
            f"{tier} seed {q.seed}: method language missing from {q.text!r}"
        )


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-partition-collections-equal-groups")
        assert skill.domain == "early-foundations"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["equal-groups", "skip-count-2-5-10"]
        stages_covered = set()
        for source in skill.sources[:3]:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1  # the pr1 exam-tier chain source, appended after hard
