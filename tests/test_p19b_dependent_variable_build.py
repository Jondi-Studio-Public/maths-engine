"""Tests the dependent-and-independent-variable identification skill (`senior-identify-the-dependent-and-
independent-variable`)."""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p19b_dependent_variable_build import (
    PAIRS,
    _verify_no_answer_leak,
)

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_distractors_distinct(tier):
    for seed in SEEDS:
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        repeat = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        assert q.to_json() == repeat.to_json()
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("tier", TIERS)
def test_answer_in_mcq_options_and_grader_accepts_rejects(tier):
    for seed in SEEDS:
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        options = mcq_options(q)
        assert q.answer_text in options
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for distractor in q.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(q, str(distractor_index)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_dependent_word_appears_exactly_once_independent_never(tier):
    for seed in SEEDS:
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        assert q.text.count("dependent") == 1
        assert "independent" not in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_distractor_count_per_tier(tier):
    expected = {"easy": 2, "medium": 2, "hard": 3, "exam": 2}
    for seed in SEEDS:
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        assert len(q.distractors) == expected[tier], (tier, seed, q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_rendered_questions_over_60_seeds(tier):
    texts = {make("p19b_dependent_variable", seed=s, difficulty=tier, tier=tier).text for s in SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct texts over 60 seeds at tier {tier}"


@pytest.mark.parametrize("tier", TIERS)
def test_named_first_balance_40_60_over_200_seeds(tier):
    """Within the stimulus, the dependent variable is named before the independent
    one on 40-60% of 200 draws per tier (checked by finding whichever of dv/iv text
    appears earlier in the rendered stimulus, up to the fixed question sentence)."""
    dv_first = 0
    total = 0
    for seed in range(200):
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        text = q.text
        dv = q.answer_text
        # iv is the first distractor by construction (the pair's iv is always index 0)
        iv = q.distractors[0]
        dv_pos = text.find(dv)
        iv_pos = text.find(iv)
        assert dv_pos != -1 and iv_pos != -1
        total += 1
        if dv_pos < iv_pos:
            dv_first += 1
    share = dv_first / total
    assert 0.40 <= share <= 0.60, f"tier {tier}: dv named first {share:.2f} of 200 draws"


def test_in_generator_check_fires_on_corrupted_value():
    """Deliberately corrupt a rendered text (drop the dv's own text) and confirm
    _verify_no_answer_leak raises."""
    q = make("p19b_dependent_variable", seed=0, difficulty="easy", tier="easy")
    dv = q.answer_text
    iv = q.distractors[0]
    corrupted_text = q.text.replace(dv, "")
    with pytest.raises(VerificationError):
        _verify_no_answer_leak(corrupted_text, dv, iv, tuple(q.distractors[1:]))
    # sanity: the uncorrupted text passes
    _verify_no_answer_leak(q.text, dv, iv, tuple(q.distractors[1:]))


def test_answer_space_widened_pool_has_fifteen_scenarios_with_distinct_fields():
    assert len(PAIRS) == 15
    for pair in PAIRS:
        values = (pair["iv"], pair["dv"], pair["controlled"], pair["other"], pair["exam_trap"])
        assert len(set(values)) == 5, f"scenario fields must all differ: {pair}"
        assert pair["trap_style"] in ("condition", "selection")


@pytest.mark.parametrize("tier", TIERS)
def test_no_depend_root_in_stimulus_over_200_seeds(tier):
    """Defect 1: the root 'depend' (depend/depends/depended/ depending/dependent) must never appear in
    the stimulus -- only the fixed question sentence may say 'dependent', and exactly once."""
    for seed in range(200):
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        stimulus = q.text.rsplit(" Which quantity", 1)[0]
        assert "depend" not in stimulus.lower(), (tier, seed, q.text)


_FOR_EACH_UNIT_RE = re.compile(r"for (?:each|every) ([a-z]+)")
_CONTEXT_UNIT_WHITELIST = {
    "plant": {"plant", "seedling"},
    "seedling": {"seedling", "plant"},
    "student": {"student"},
    "participant": {"participant"},
    "patient": {"patient"},
    "person": {"person"},
    "musician": {"musician"},
    "store": {"store", "week"},
    "week": {"week", "store"},
}


@pytest.mark.parametrize("tier", TIERS)
def test_per_case_unit_word_matches_context_over_200_seeds(tier):
    """Defect 2: the per-case unit word in 'for each/every ___' must fit the study's own context (never
    'participant' applied to a plant study)."""
    for seed in range(200):
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        for unit in _FOR_EACH_UNIT_RE.findall(q.text):
            if unit == "case":
                continue
            allowed = _CONTEXT_UNIT_WHITELIST.get(unit)
            assert allowed, (tier, seed, unit, q.text)
            assert any(word in q.text for word in allowed), (tier, seed, unit, q.text)


_INCOHERENT_HELD_CONSTANT_RE = re.compile(r"(measured|taken|score|speed|rate|severity)[^.,;]*\bwas kept the same\b")


@pytest.mark.parametrize("tier", TIERS)
def test_no_incoherent_held_constant_measurement_over_200_seeds(tier):
    """Defect 3: a baseline MEASUREMENT can never be described as 'kept the same' -- only a genuinely
    controllable condition can."""
    for seed in range(200):
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        assert not _INCOHERENT_HELD_CONSTANT_RE.search(q.text), (tier, seed, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_sentence_case_over_200_seeds(tier):
    """Defect 5: every sentence in question.text starts with a capital letter."""
    for seed in range(200):
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        for sentence in (s.strip() for s in q.text.split(". ") if s.strip()):
            assert sentence[0].isupper(), (tier, seed, sentence, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_other_and_extra_quantities_are_genuine_per_case_or_fixed_over_60_seeds(tier):
    """Defect 4: every 'other named quantity' offered as a distractor must be a genuine per-case
    variable or a stated fixed condition, never a whole-study aggregate (e.g. 'the number of X in the
    trial/class')."""
    banned_substrings = ("number of patients in the trial", "number of students in the class")
    for seed in SEEDS:
        q = make("p19b_dependent_variable", seed=seed, difficulty=tier, tier=tier)
        for banned in banned_substrings:
            assert banned not in q.text, (tier, seed, q.text)


def test_skill_loads_with_expected_metadata():
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}
    skill = skills_dict.get("senior-identify-the-dependent-and-independent-variable")
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.prerequisites == ["classify-data", "variable-concept"]
    assert skill.placement_eligible is False
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    exam_sources = [src for src in skill.sources if src.difficulty == "exam"]
    assert len(exam_sources) == 1
    assert exam_sources[0].stages == ["exam"]
