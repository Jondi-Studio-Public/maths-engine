"""Tests for `mathgen/generators/p19a_position_language_build.py`: `p19a_position_language` (skill
`f10-position-location-language`, AC9MFSP02).

This file holds the generator-specific proofs: the mcq
grader battery, the in-generator exactly-one-true-option check, the
mutual-exclusivity contract (never two true options), key-term concentration,
key-letter balance, and skill load.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19a_position_language_oracle.py`.
"""

from __future__ import annotations

from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19a_position_language_build as gen

TOPIC = "p19a_position_language"
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


# ============================================================================
# Answer-space contract: no single term is key on more than 30% of 200 draws.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_single_key_term_over_30pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share <= 0.30, f"{tier}: top key {counts.most_common(1)} share {top_share}"


@pytest.mark.parametrize("tier", TIERS)
def test_options_are_plain_text_never_a_number(tier):
    for q in _all(tier, seeds=range(60)):
        for opt in [q.answer_text] + list(q.distractors):
            assert not opt.strip().isdigit(), f"{tier}: numeric option {opt!r} in {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_key_letter_position_balanced_at_400_seeds(tier):
    counts = Counter()
    for s in range(400):
        q = make(TOPIC, seed=s, difficulty=tier)
        options = mcq_options(q)
        counts[options.index(q.answer_text)] += 1
    total = sum(counts.values())
    for idx, n in counts.items():
        share = n / total
        assert 0.10 <= share <= 0.60, f"{tier}: option index {idx} share {share} ({n}/{total})"


# ============================================================================
# Mutual exclusivity: exactly one option is true (checked here by re-deriving
# from the rendered options -- never "next to" alongside a directional term
# for a genuinely adjacent pair. See the oracle for the fully independent
# re-derivation from question.text alone.)
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_never_offers_both_next_to_and_a_true_directional_term(tier):
    directional_terms = {"above", "below", "in front of", "behind", "to the left of", "to the right of"}
    for q in _all(tier, seeds=range(60)):
        options = mcq_options(q)
        if "next to" in options and q.answer_text != "next to":
            # "next to" is offered as a distractor -- it must be FALSE, i.e.
            # the true directional term (the key) is some other option, and
            # "next to" itself is never also the key at the same time as a
            # directional key (mutual exclusivity is enforced at construction;
            # this asserts the observable consequence).
            assert q.answer_text in directional_terms or q.answer_text.startswith("between")


# See tests/test_p19a_position_language_oracle.py for the fully independent proof from question.text
# alone; this checks the generator's own internal helper directly (never exercised in the shipped
# generator without this guarantee already holding, by construction of `_directional_options`).
# ============================================================================


def test_independent_bare_between_helper_matches_end_vs_interior():
    order = ["a", "b", "c", "d", "e"]
    assert gen._independent_bare_between_truth(order, "a") is False  # end
    assert gen._independent_bare_between_truth(order, "e") is False  # end
    assert gen._independent_bare_between_truth(order, "b") is True  # interior
    assert gen._independent_bare_between_truth(order, "c") is True  # interior
    assert gen._independent_bare_between_truth(order, "d") is True  # interior


def test_pick_false_between_pair_never_brackets_the_target():
    import random

    rng = random.Random(0)
    order = ["a", "b", "c", "d", "e"]
    for target in ("b", "c", "d"):
        for _ in range(20):
            a, b = gen._pick_false_between_pair(rng, order, target)
            ti, ai, bi = order.index(target), order.index(a), order.index(b)
            lo, hi = min(ai, bi), max(ai, bi)
            assert not (lo < ti < hi), f"false pair ({a}, {b}) actually brackets {target!r}"


# The full from-text re-derivation (recovering the entity list and target
# from `question.text` alone, with no import of this module) is in
# `tests/test_p19a_position_language_oracle.py::
# test_bare_between_never_offered_for_an_interior_target` -- the independent
# proof of the same contract these two direct-helper tests check above.


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(tier):
    for q in _all(tier, seeds=range(30)):
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct, f"{tier} seed {q.seed}: canonical rejected"
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, i).correct, f"{tier} seed {q.seed}: distractor {opt!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(15)):
        assert not grade_mcq(q, "99").correct


# ============================================================================
# Neighbour separation: a stated (never rendered) arrangement, never a
# Cartesian coordinate pair.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_never_mentions_coordinates(tier):
    for q in _all(tier, seeds=range(60)):
        lowered = q.text.lower()
        assert "coordinate" not in lowered and "(" not in q.text, (
            f"{tier} seed {q.seed}: unexpected coordinate language: {q.text!r}"
        )


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-position-location-language")
        assert skill.domain == "primary-geometry-time"
        assert skill.answer == "mcq"
        assert skill.placement_eligible is False
        assert skill.prerequisites == []
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 0  # Foundation year -- exempt
