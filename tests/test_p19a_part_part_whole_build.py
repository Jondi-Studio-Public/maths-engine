"""Tests for `mathgen/generators/p19a_part_part_whole_build.py`: `p19a_part_part_whole` (skill
`f10-part-part-whole-collections-to-10`, AC9MFN04).

This file holds the
generator-specific proofs: the answer-space/concentration targets
(top key < 15%, whole == 10 < 30%, whole == 5 reported), the
`number` grader battery, the in-generator checks firing on a corrupted value,
skill load, and the misconception-specific properties.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19a_part_part_whole_oracle.py`.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19a_part_part_whole_build as gen

TOPIC = "p19a_part_part_whole"
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


def test_default_params_also_have_at_least_two_distractors_on_85pct():
    seeds = [make(TOPIC, seed=s) for s in range(200)]
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"default: only {at_least_two}/200 have >= 2 distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_integer_1_to_9(tier):
    for q in _all(tier, seeds=range(60)):
        assert isinstance(q.answer, int)
        assert 1 <= q.answer <= 9, f"{tier} seed {q.seed}: key {q.answer} out of 1-9"
        assert q.answer_text == str(q.answer)


# ============================================================================
# Construct-to-target contract: top key < 15%, whole == 10 < 30%,
# whole == 5 reported (not over-represented). Measured directly on the
# generator's own draw function -- deterministic, same rng seeding scheme
# `mathgen.core.make` uses ("{topic}:{difficulty or ''}:{seed}").
# ============================================================================


def _draws(tier, n=200):
    out = []
    for s in range(n):
        rng = random.Random(f"{TOPIC}:{tier}:{s}")
        out.append(gen._draw_whole_part(rng))
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_top_key_under_15pct_of_200_draws(tier):
    draws = _draws(tier)
    counts = Counter(key for _whole, _part, key in draws)
    top_share = counts.most_common(1)[0][1] / len(draws)
    assert top_share < 0.15, f"{tier}: top key {counts.most_common(1)} share {top_share}"


@pytest.mark.parametrize("tier", TIERS)
def test_whole_equals_10_under_30pct_of_200_draws(tier):
    draws = _draws(tier)
    whole10 = sum(1 for whole, _part, _key in draws if whole == 10)
    share = whole10 / len(draws)
    assert share < 0.30, f"{tier}: whole==10 share {share}"


@pytest.mark.parametrize("tier", TIERS)
def test_whole_equals_5_not_over_represented(tier):
    draws = _draws(tier)
    whole5 = sum(1 for whole, _part, _key in draws if whole == 5)
    share = whole5 / len(draws)
    # report + guard: whole==5 must not dominate the way whole==10 could;
    # <30% is the same ceiling applied to the other named total.
    assert share < 0.30, f"{tier}: whole==5 share {share}"


# ============================================================================
# Independent in-generator checks
# ============================================================================


def test_draw_whole_part_always_satisfies_range_and_addition_check():
    for seed in range(60):
        rng = random.Random(f"{TOPIC}:easy:{seed}")
        whole, part, key = gen._draw_whole_part(rng)
        assert 1 <= part < whole <= 10
        assert part + key == whole


def test_check_fires_on_a_corrupted_range():
    from mathgen.core import check

    with pytest.raises(VerificationError):
        check(1 <= 11 < 5 <= 10, "p19a_part_part_whole: whole/part out of range")


def test_check_equivalent_fires_on_a_corrupted_key():
    from mathgen.core import check_equivalent

    check_equivalent(4, 10 - 6, "ok")  # 10 - 6 == 4, no exception
    with pytest.raises(VerificationError):
        check_equivalent(5, 10 - 6, "corrupted key")


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
        wrong = str(q.answer + 1000)
        assert not grade_number(q, wrong).correct


# ============================================================================
# Properties: never reuses the bond stem, whole ranges over 2-10
# (not fixed at 10), hard tier's two shapes both appear.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_never_uses_the_bond_stem(tier):
    for q in _all(tier, seeds=range(60)):
        assert "and what make" not in q.text.lower(), f"{tier} seed {q.seed}: {q.text!r} uses the bond stem"


@pytest.mark.parametrize("tier", TIERS)
def test_whole_ranges_over_2_to_10_not_fixed(tier):
    wholes = set()
    for whole, _part, _key in _draws(tier, n=60):
        wholes.add(whole)
    assert len(wholes) >= 5, f"{tier}: only {len(wholes)} distinct wholes seen in 60 draws: {wholes}"
    assert wholes != {10}, f"{tier}: whole is fixed at 10"


def test_hard_tier_both_shapes_appear():
    combine_seen = False
    labelled_seen = False
    for q in _all("hard", seeds=range(60)):
        if "Whole:" in q.text:
            labelled_seen = True
        elif "make" in q.text:
            combine_seen = True
    assert combine_seen, "hard: combining-language shape never seen in 60 seeds"
    assert labelled_seen, "hard: labelled Whole:/One part: shape never seen in 60 seeds"


def test_hard_tier_labelled_shape_varies_whole_position():
    whole_first_seen = False
    whole_last_seen = False
    for q in _all("hard", seeds=range(60)):
        if q.text.startswith("Whole:"):
            whole_first_seen = True
        elif "Whole:" in q.text:
            whole_last_seen = True
    assert whole_first_seen, "hard: 'Whole:' first line order never seen in 60 seeds"
    assert whole_last_seen, "hard: 'Whole:' last line order never seen in 60 seeds"


# Every raw template in every tier must carry an explicit two-part-closure marker -- enumerated here
# over the module's own template tuples, not sampled rendered text, so a future template addition
# that forgets closure fails immediately rather than only showing up in a large seed sweep.
# ============================================================================

_CLOSURE_MARKERS = ("the rest", "either", "not {colora}", "make {whole}", "made up of", "whole:")


def _all_templates():
    return (
        list(gen._EASY)
        + list(gen._MEDIUM)
        + list(gen._HARD_COMBINE)
        + list(gen._HARD_LABELLED_WHOLE_FIRST)
        + list(gen._HARD_LABELLED_WHOLE_LAST)
    )


def test_every_template_states_the_two_part_closure():
    missing = []
    for template in _all_templates():
        lowered = template.lower()
        if not any(marker in lowered for marker in _CLOSURE_MARKERS):
            missing.append(template)
    assert missing == [], f"templates missing a two-part-closure marker: {missing}"


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_at_least_90pct_of_200_draws_state_the_two_part_closure(tier):
    """A rendered prompt states the closure if it contains 'the rest'/'the other'/'not <colour>', is a hard-tier
    'make'/'made up of'/'Whole:' shape, or explicitly says "either... or..."."""
    closure_substrings = ("the rest", "the other", "either", "not ")
    no_closure = []
    for q in _all(tier, seeds=range(200)):
        lowered = q.text.lower()
        if not any(s in lowered for s in closure_substrings):
            no_closure.append(q.text)
    share_with_closure = 1 - len(no_closure) / 200
    assert share_with_closure >= 0.90, (
        f"{tier}: only {share_with_closure:.2%} of 200 draws state the two-part closure; "
        f"examples missing it: {no_closure[:3]}"
    )


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-part-part-whole-collections-to-10")
        assert skill.domain == "early-foundations"
        assert skill.learner_band == "early"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["f10-subitise-name-to-5", "number-bonds-to-10"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
        assert "exam" not in stages_covered
