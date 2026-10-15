"""The scenario states plain facts, never a category's definitional phrase; the definitions are
stated once; the MCQ options are the three short labels."""

from __future__ import annotations

from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p19b_noncausal_explanation_build import (
    BANNED_PHRASES,
    CATEGORIES,
    COINCIDENCE_PAIRS,
    COMMON_CAUSE_TRIPLES,
    CONFOUNDING_TRIPLES,
    _IRRELEVANT_EXTRAS_GENERIC,
    _IRRELEVANT_EXTRAS_SUBJECTS,
    _PEOPLE_SUBJECTS,
    LABEL_COINCIDENCE,
    LABEL_COMMON_CAUSE,
    LABEL_CONFOUNDING,
    OPTIONS,
    _DEFINITIONS_MARKER,
    _fact,
    _verify_scenario,
)

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")


def _scenario_part(text: str) -> str:
    return text.split(_DEFINITIONS_MARKER)[0]


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_distractors_distinct(tier):
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        repeat = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        assert q.to_json() == repeat.to_json()
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) == 2


@pytest.mark.parametrize("tier", TIERS)
def test_answer_always_one_of_three_fixed_labels(tier):
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        assert q.answer_text in OPTIONS
        assert set(q.distractors) | {q.answer_text} == set(OPTIONS)


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_answer_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, correct_index).correct
        for distractor in q.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(q, distractor_index).correct


@pytest.mark.parametrize("tier", TIERS)
def test_all_three_definitions_stated_once_and_options_are_short_labels(tier):
    """Options are the short labels; the definitions appear once,
    after the scenario, never repeated as option text."""
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        assert _DEFINITIONS_MARKER in q.text
        assert q.text.count(_DEFINITIONS_MARKER) == 1
        assert set(q.distractors) | {q.answer_text} == {LABEL_COINCIDENCE, LABEL_COMMON_CAUSE, LABEL_CONFOUNDING}


@pytest.mark.parametrize("tier", TIERS)
def test_scenario_text_never_states_a_categorys_own_definition(tier):
    """Ban list of definitional phrases, checked only against the
    scenario portion of the text (before the definitions block, which legitimately
    states all three)."""
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        scenario = _scenario_part(q.text).lower()
        for phrase in BANNED_PHRASES:
            assert phrase not in scenario, f"tier {tier} seed {seed}: banned phrase {phrase!r} in scenario {scenario!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_every_scenario_states_the_observed_xy_association(tier):
    """Every scenario, whichever category, opens by stating the
    observed association between X and Y itself (not just an X-Z or a bare fact)."""
    xy_clauses = {t[3] for t in CONFOUNDING_TRIPLES}
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        scenario = _scenario_part(q.text)
        if q.answer_text == LABEL_CONFOUNDING:
            assert any(clause in scenario for clause in xy_clauses), (
                f"tier {tier} seed {seed}: confounding scenario never states the X/Y association: {scenario!r}"
            )
        else:
            # coincidence and common-cause openings name X and Y directly in the
            # same lead sentence ("X and Y have tended to rise and fall together" /
            # "both X and Y increase ...").
            first_sentence = scenario.split(".")[0]
            assert " and " in first_sentence, (
                f"tier {tier} seed {seed}: opening does not pair X and Y: {first_sentence!r}"
            )


_ALL_NAMED_VARIABLES = set()
for _x, _y in COINCIDENCE_PAIRS:
    _ALL_NAMED_VARIABLES.update((_x, _y))
for _x, _y, _z, _w, _subj in COMMON_CAUSE_TRIPLES:
    _ALL_NAMED_VARIABLES.update((_x, _y, _z))
for _x, _y, _z, _xy, _xz, _subj in CONFOUNDING_TRIPLES:
    _ALL_NAMED_VARIABLES.update((_x, _y, _z))


@pytest.mark.parametrize("tier", TIERS)
def test_hard_tier_irrelevant_extra_is_never_the_linking_variable(tier):
    """The hard-tier irrelevant extra must be a plausible study-meta
    quantity that fits the scenario's own frame, and never one of the scenario's own
    named variables (of ANY pool entry, not just the one drawn)."""
    if tier != "hard":
        pytest.skip("only applies to the hard tier")
    subjects_only = set(_IRRELEVANT_EXTRAS_SUBJECTS) - set(_IRRELEVANT_EXTRAS_GENERIC)
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        scenario = _scenario_part(q.text)
        scenario_lower = scenario.lower()
        matched = [extra for extra in _IRRELEVANT_EXTRAS_SUBJECTS if extra in scenario_lower]
        assert matched, f"seed {seed}: no irrelevant extra found in {scenario!r}"
        assert matched[0] not in _ALL_NAMED_VARIABLES
        # A subjects-only extra ("participants", "survey questions") must only appear in
        # a study framed around a named group of people, never a time-series study.
        if matched[0] in subjects_only:
            assert any(subj in scenario_lower for subj in _PEOPLE_SUBJECTS), (
                f"seed {seed}: subjects-only extra {matched[0]!r} used in a non-subjects scenario: {scenario!r}"
            )


@pytest.mark.parametrize("tier", TIERS)
def test_scenario_opens_with_a_capital_letter(tier):
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        first_char = q.text[0]
        assert first_char == first_char.upper() and first_char.isalpha()


@pytest.mark.parametrize("tier", TIERS)
def test_every_sentence_in_the_scenario_starts_with_a_capital_letter(tier):
    """Applied to every sentence, not just the first: a mid-scenario
    sentence (e.g. the hard-tier irrelevant-extra sentence) must also start with a
    capital letter."""
    for seed in SEEDS:
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        scenario = _scenario_part(q.text).strip()
        sentences = [sent.strip() for sent in scenario.split(". ") if sent.strip()]
        for sent in sentences:
            first_char = sent[0]
            assert first_char == first_char.upper() and first_char.isalpha(), (
                f"tier {tier} seed {seed}: sentence does not start with a capital: {sent!r}"
            )


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_rendered_questions_over_60_seeds(tier):
    texts = {make("p19b_noncausal_explanation", seed=s, difficulty=tier, tier=tier).text for s in SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct texts over 60 seeds at tier {tier}"


@pytest.mark.parametrize("tier", TIERS)
def test_no_category_exceeds_45_percent_of_200_draws(tier):
    counts = Counter()
    for seed in range(200):
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier, tier=tier)
        counts[q.answer_text] += 1
    for option, count in counts.items():
        share = count / 200
        assert share <= 0.45, f"tier {tier}: option {option!r} is {share:.2f} of 200 draws"
    assert len(counts) == 3, f"tier {tier}: not all three categories drawn over 200 seeds: {counts}"


def test_in_generator_check_fires_on_corrupted_scenario():
    """A rendered scenario that never states the required marker phrase for the
    claimed category must be rejected by _verify_scenario."""
    _verify_scenario("Variable Z drives increases in both variable X and variable Y.", "common_cause")  # sanity
    corrupted_text = "This text does not state any mechanism at all."
    with pytest.raises(VerificationError):
        _verify_scenario(corrupted_text, "common_cause")


def test_in_generator_check_fires_on_banned_phrase():
    corrupted_text = "Variable Z independently affects both variable X and variable Y."
    with pytest.raises(VerificationError):
        _verify_scenario(corrupted_text, "common_cause")


def test_in_generator_check_fires_on_ambiguous_scenario():
    ambiguous_text = (
        "Variable Z drives increases in both variable X and variable Y. "
        "Variable Z is also known to affect variable Y directly, since reasons."
    )
    with pytest.raises(VerificationError):
        _verify_scenario(ambiguous_text, "common_cause")


def test_fact_sentences_are_category_unique():
    s = dict(
        X="variable X",
        Y="variable Y",
        Z="variable Z",
        when_clause="always",
        xy_clause="X relates to Y",
        xz_clause="X relates to Z",
    )
    facts = {cat: _fact(cat, s) for cat in CATEGORIES}
    assert len(set(facts.values())) == 3


def test_skill_loads_with_expected_metadata():
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}
    skill = skills_dict.get("senior-identify-possible-non-causal-explanations-for-an-association")
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.prerequisites == ["senior-distinguish-between-causality-and-correlation-through-examples"]
    assert skill.placement_eligible is False
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    exam_sources = [src for src in skill.sources if src.difficulty == "exam"]
    assert len(exam_sources) == 1
    assert exam_sources[0].stages == ["exam"]


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_confounding_scenario_states_the_direct_effect(tier):
    # Without "X could plausibly affect Y
    # directly", a confounding scenario ("Z affects Y, and Z goes with X") fits
    # Common cause as well, so the question had two defensible answers.
    seen = 0
    for seed in range(200):
        q = make("p19b_noncausal_explanation", seed=seed, difficulty=tier)
        if q.answer != "Confounding":
            continue
        seen += 1
        scenario = q.text.split(" The three possible explanations")[0]
        assert "could plausibly affect" in scenario, scenario
    assert seen >= 40
