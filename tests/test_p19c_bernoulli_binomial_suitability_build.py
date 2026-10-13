"""Tests for: Bernoulli + binomial model-suitability pair."""

from __future__ import annotations

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p19c_bernoulli_binomial_suitability_build import (
    OPTIONS_BERNOULLI,
    OPTIONS_BINOMIAL,
    OPT_BERNOULLI_CONTINUOUS,
    OPT_BERNOULLI_DEPENDENT,
    OPT_BERNOULLI_SUITED,
    OPT_BINOMIAL_NOT_CONST_P,
    OPT_BINOMIAL_NOT_FIXED,
    OPT_BINOMIAL_NOT_INDEP,
    OPT_BINOMIAL_NOT_TWO,
    OPT_BINOMIAL_SUITED,
    _CONTEXTS,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
TIERS = ["easy", "medium", "hard", "exam"]


# ---------------------------------------------------------------------------
# p19c_binomial_suitability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        q2 = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_distractors_distinct_and_not_answer(difficulty):
    # distractors = 4 options for every class, including NOT_INDEP, which used to offer all
    # 4 remaining options (key + 4 = 5) but now excludes NOT_CONST_P from its own pool so the
    # item never offers two defensible answers -- so the option COUNT must stay constant at
    # 4 rather than vary by class (5 elsewhere, 4 for NOT_INDEP), which would itself be a tell.
    for seed in SEEDS:
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == 3
        assert len(set(q.distractors)) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_options_are_recognised(difficulty):
    for seed in SEEDS:
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        assert q.answer_text in OPTIONS_BINOMIAL
        for d in q.distractors:
            assert d in OPTIONS_BINOMIAL


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_correct_answer_grades_correct(difficulty):
    for seed in SEEDS:
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        idx = options.index(q.answer_text)
        result = grade_mcq(q, str(idx))
        assert result.correct, f"seed {seed} ({difficulty}): canonical answer did not grade correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_every_distractor_grades_incorrect(difficulty):
    for seed in SEEDS:
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        for d in q.distractors:
            idx = options.index(d)
            result = grade_mcq(q, str(idx))
            assert not result.correct, f"seed {seed} ({difficulty}): distractor {d!r} graded correct"


def test_binomial_all_five_classes_occur_over_400_seeds():
    seen = set()
    for tier in TIERS:
        for seed in range(400):
            q = make("p19c_binomial_suitability", seed=seed, difficulty=tier)
            seen.add(q.answer_text)
    assert seen == set(OPTIONS_BINOMIAL), f"missing classes: {set(OPTIONS_BINOMIAL) - seen}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_top_key_share_under_40_percent(difficulty):
    counts = {}
    n = 200
    for seed in range(n):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        counts[q.answer_text] = counts.get(q.answer_text, 0) + 1
    top_share = max(counts.values()) / n
    assert top_share <= 0.40, f"{difficulty}: top class share {top_share:.2%} exceeds 40% ({counts})"


def test_binomial_corrupted_answer_fails_independent_check():
    from mathgen.core import VerificationError, check

    with pytest.raises(VerificationError):
        check("not a real option" in OPTIONS_BINOMIAL, "bad class")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_not_independent_never_asserts_fixed_probability_on_every_trial(difficulty):
    """Defect A: the 'not independent' class's own probability sentence must never claim a
    probability fixed "on every trial" in the same question as the independence-violation
    clause's "changes the probability" -- that is a self-contradiction. Also positively
    confirm the reworded first-draw-only phrasing is present whenever this class is drawn."""
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BINOMIAL_NOT_INDEP:
            lower = q.text.lower()
            assert not ("changes the probability" in lower and "on every trial" in lower), (
                f"seed {seed} ({difficulty}): contradictory clauses in {q.text!r}"
            )
            assert "at the start" in lower, (
                f"seed {seed} ({difficulty}): expected starting-share phrasing, got {q.text!r}"
            )


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_dependent_never_mixes_single_and_plural_trial_framing(difficulty):
    """Defect B: the 'dependent/unknown probability' class describes a SINGLE trial, so its
    scenario must never invoke multi-trial "independence"/"repeated" language ("trials",
    "independent", "repeated") which does not honestly apply to one trial."""
    for seed in range(200):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BERNOULLI_DEPENDENT:
            lower = q.text.lower()
            for banned in ("trials", "independent", "repeated"):
                assert banned not in lower, f"seed {seed} ({difficulty}): banned phrase {banned!r} in {q.text!r}"
            assert "one draw" in lower or "this draw" in lower or "single draw" in lower, (
                f"seed {seed} ({difficulty}): expected single-draw framing, got {q.text!r}"
            )


# ---------------------------------------------------------------------------
# p19c_bernoulli_suitability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        q2 = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_distractors_distinct_and_not_answer(difficulty):
    for seed in SEEDS:
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == 3
        assert len(set(q.distractors)) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_options_are_recognised(difficulty):
    for seed in SEEDS:
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        assert q.answer_text in OPTIONS_BERNOULLI
        for d in q.distractors:
            assert d in OPTIONS_BERNOULLI


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_correct_answer_grades_correct(difficulty):
    for seed in SEEDS:
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        idx = options.index(q.answer_text)
        result = grade_mcq(q, str(idx))
        assert result.correct, f"seed {seed} ({difficulty}): canonical answer did not grade correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_every_distractor_grades_incorrect(difficulty):
    for seed in SEEDS:
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        for d in q.distractors:
            idx = options.index(d)
            result = grade_mcq(q, str(idx))
            assert not result.correct, f"seed {seed} ({difficulty}): distractor {d!r} graded correct"


def test_bernoulli_all_four_classes_occur_over_400_seeds():
    seen = set()
    for tier in TIERS:
        for seed in range(400):
            q = make("p19c_bernoulli_suitability", seed=seed, difficulty=tier)
            seen.add(q.answer_text)
    assert seen == set(OPTIONS_BERNOULLI), f"missing classes: {set(OPTIONS_BERNOULLI) - seen}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_top_key_share_under_40_percent(difficulty):
    counts = {}
    n = 200
    for seed in range(n):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        counts[q.answer_text] = counts.get(q.answer_text, 0) + 1
    top_share = max(counts.values()) / n
    assert top_share <= 0.40, f"{difficulty}: top class share {top_share:.2%} exceeds 40% ({counts})"


# ---------------------------------------------------------------------------
# Cross-topic boundary: every scenario states its own trial count (binomial)
# or its own single-trial framing (bernoulli) explicitly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_suited_scenarios_always_state_a_trial_count(difficulty):
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BINOMIAL_SUITED:
            assert "trials, fixed in advance" in q.text or "trials are drawn" in q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_suited_scenarios_never_name_a_repeat_count(difficulty):
    for seed in range(200):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BERNOULLI_SUITED:
            assert "this is repeated" not in q.text.lower()
            assert "one " in q.text.lower()


# No rendered question echoes its own key-defining phrase, any tier.

# Banned substrings, keyed by the class whose own text must never contain them.
_BANNED_BINOMIAL = {
    OPT_BINOMIAL_NOT_FIXED: ["no trial count is fixed in advance"],
    OPT_BINOMIAL_NOT_INDEP: ["changes the probability"],
    OPT_BINOMIAL_NOT_TWO: ["not one of two discrete categories", "not just two"],
}
# These must never appear at all, in ANY class's text, at any tier (the deleted easy-tier
# restating block, and the deleted "negligible"/"changed noticeably" glosses).
_BANNED_ANYWHERE_BINOMIAL = [
    "fails the binomial condition that",
    "all four binomial conditions",
    "so the effect of not replacing each one is negligible",
]
_BANNED_BERNOULLI = {
    OPT_BERNOULLI_DEPENDENT: ["cannot be pinned down"],
    OPT_BERNOULLI_CONTINUOUS: ["not one of two discrete categories"],
}
_BANNED_ANYWHERE_BERNOULLI = [
    "this is a single trial with two outcomes and a fixed probability of success",
    "this describes a fixed number of repeated trials, not a single trial",
    "the outcome is not one of two discrete categories.",
    "an unstated number of items has already been removed",
    "so its composition has changed noticeably",
    "so its composition barely changes",
]


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_no_class_echoes_its_own_key_phrase(difficulty):
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        lower = q.text.lower()
        for banned in _BANNED_ANYWHERE_BINOMIAL:
            assert banned not in lower, f"seed {seed} ({difficulty}): banned {banned!r} in {q.text!r}"
        for banned in _BANNED_BINOMIAL.get(q.answer_text, []):
            assert banned not in lower, (
                f"seed {seed} ({difficulty}): class {q.answer_text!r} echoes {banned!r} in {q.text!r}"
            )


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_no_class_echoes_its_own_key_phrase(difficulty):
    for seed in range(200):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        lower = q.text.lower()
        for banned in _BANNED_ANYWHERE_BERNOULLI:
            assert banned not in lower, f"seed {seed} ({difficulty}): banned {banned!r} in {q.text!r}"
        for banned in _BANNED_BERNOULLI.get(q.answer_text, []):
            assert banned not in lower, (
                f"seed {seed} ({difficulty}): class {q.answer_text!r} echoes {banned!r} in {q.text!r}"
            )


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_not_indep_never_offers_not_const_p(difficulty):
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BINOMIAL_NOT_INDEP:
            assert OPT_BINOMIAL_NOT_CONST_P not in q.distractors, (
                f"seed {seed} ({difficulty}): NOT_INDEP offered NOT_CONST_P in {q.distractors!r}"
            )


# "novice workers"/"experienced workers" only fits staffed contexts. The phrase was still
# paired with "a basketball free-throw session" 9/137 times -- a free-throw shooter is a
# player, not a "worker" -- so that context is now unstaffed too.

_UNSTAFFED_SUBJECTS = ("a school raffle", "a plant nursery", "a basketball free-throw session")


@pytest.mark.parametrize("difficulty", TIERS)
def test_novice_experienced_wording_only_on_staffed_contexts(difficulty):
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text != OPT_BINOMIAL_NOT_CONST_P:
            continue
        if "novice" in q.text or "experienced" in q.text:
            for subject in _UNSTAFFED_SUBJECTS:
                assert subject not in q.text, (
                    f"seed {seed} ({difficulty}): novice/experienced wording paired with "
                    f"unstaffed subject {subject!r} in {q.text!r}"
                )


@pytest.mark.parametrize("difficulty", TIERS)
def test_novice_experienced_fix_narrows_pool_not_breaks_rendering(difficulty):
    neutral_pairs = ("Monday's shift", "Group A", "the morning session")
    found = False
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text != OPT_BINOMIAL_NOT_CONST_P:
            continue
        if not any(subject in q.text for subject in _UNSTAFFED_SUBJECTS):
            continue
        if any(pair in q.text for pair in neutral_pairs):
            found = True
            break
    assert found, (
        f"({difficulty}): no context-neutral group pair ever occurred for a raffle/nursery "
        "NOT_CONST_P draw over 200 seeds"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_binomial_every_item_has_exactly_four_options_regardless_of_class(difficulty):
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3
        assert len({q.answer_text, *q.distractors}) == 4


def test_binomial_hard_exam_suited_recodes_a_rating_not_mere_bigger_numbers():
    found = 0
    for difficulty in ("hard", "exam"):
        for seed in range(200):
            q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
            if q.answer_text == OPT_BINOMIAL_SUITED and "point rating comes in at" in q.text:
                found += 1
                assert "not the raw rating itself" in q.text
    assert found > 0, "hard/exam SUITED recoded-rating clause never occurred over 400 seeds"


def test_bernoulli_hard_exam_suited_recodes_a_measurement_not_mere_bigger_numbers():
    found = 0
    for difficulty in ("hard", "exam"):
        for seed in range(200):
            q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
            if q.answer_text == OPT_BERNOULLI_SUITED and "exceeds a fixed threshold" in q.text:
                found += 1
                assert "not the raw measurement itself" in q.text
    assert found > 0, "hard/exam SUITED recoded-measurement clause never occurred over 400 seeds"


_REMOVABLE_SUBJECTS = {c["subject"] for c in _CONTEXTS if c["removable"]}
_NON_REMOVABLE_SUBJECTS = {c["subject"] for c in _CONTEXTS if not c["removable"]}
_CONTINUOUS_MEASURES = {c["continuous_measure"] for c in _CONTEXTS if c["continuous_measure"]}


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_dependent_only_uses_removable_contexts(difficulty):
    for seed in range(200):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BERNOULLI_DEPENDENT:
            assert not any(subject in q.text for subject in _NON_REMOVABLE_SUBJECTS), (
                f"seed {seed} ({difficulty}): DEPENDENT used a non-removable context in {q.text!r}"
            )
            assert any(subject in q.text for subject in _REMOVABLE_SUBJECTS)


@pytest.mark.parametrize("difficulty", TIERS)
def test_bernoulli_continuous_uses_its_own_context_measure_word(difficulty):
    for seed in range(200):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_BERNOULLI_CONTINUOUS:
            assert "weight in grams" not in q.text.lower()
            assert any(measure in q.text for measure in _CONTINUOUS_MEASURES), (
                f"seed {seed} ({difficulty}): CONTINUOUS text names no known context measure: {q.text!r}"
            )


# Single defensible option -- oracle-derives the same class for the disambiguation-heavy exam
# scenarios (population-ratio trick, both topics).


def test_binomial_exam_population_trick_oracle_agrees():
    """Both the exam-tier genuinely-suited (large population) and genuinely-violated (small
    population) "without replacement" scenarios must independently re-derive to the SAME
    class the generator assigned, confirming exactly one defensible option even when the
    without-replacement wording is shared between SUITED and NOT_INDEP."""
    from tests.test_p19c_bernoulli_binomial_suitability_oracle import _derive_binomial_class

    seen_suited = seen_not_indep = 0
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty="exam")
        if q.answer_text in (OPT_BINOMIAL_SUITED, OPT_BINOMIAL_NOT_INDEP):
            assert _derive_binomial_class(q.text) == q.answer_text
            if q.answer_text == OPT_BINOMIAL_SUITED:
                seen_suited += 1
            else:
                seen_not_indep += 1
    assert seen_suited > 0 and seen_not_indep > 0


def test_bernoulli_exam_disguised_suited_case_oracle_agrees():
    """The exam-tier "already removed without replacement, large population" disguise of the
    SUITED case must independently re-derive to SUITED, distinguishing it from the DEPENDENT
    (small population) case that shares the same "already removed" language."""
    from tests.test_p19c_bernoulli_binomial_suitability_oracle import _derive_bernoulli_class

    seen_disguised_suited = 0
    for seed in range(200):
        q = make("p19c_bernoulli_suitability", seed=seed, difficulty="exam")
        if q.answer_text == OPT_BERNOULLI_SUITED and "large population" in q.text:
            assert _derive_bernoulli_class(q.text) == OPT_BERNOULLI_SUITED
            seen_disguised_suited += 1
    assert seen_disguised_suited > 0


# ---------------------------------------------------------------------------
# Skills loading and metadata
# ---------------------------------------------------------------------------


def test_binomial_skill_loads_correctly():
    _, skills_list = load_skills(ROOT / "skills")
    skills = {s.id: s for s in skills_list}
    skill = skills.get("senior-identify-contexts-suitable-for-modelling-by-binomial-random")
    assert skill is not None
    assert skill.domain == "statistics-probability"
    assert skill.answer == "mcq"
    assert skill.prerequisites == ["senior-identify-contexts-suitable-for-modelling-by-bernoulli-random"]
    assert skill.placement_eligible is False


def test_bernoulli_skill_loads_correctly():
    _, skills_list = load_skills(ROOT / "skills")
    skills = {s.id: s for s in skills_list}
    skill = skills.get("senior-identify-contexts-suitable-for-modelling-by-bernoulli-random")
    assert skill is not None
    assert skill.domain == "statistics-probability"
    assert skill.answer == "mcq"
    assert skill.prerequisites == [
        "senior-review-the-concepts-and-language-of-outcomes-sample",
        "simple-event-probability",
    ]
    assert skill.placement_eligible is False


# Second defensible answers and feasibility.

import re
from functools import lru_cache


@lru_cache(maxsize=None)
def _binomial_draws(tier):
    return tuple((seed, make("p19c_binomial_suitability", seed=seed, difficulty=tier)) for seed in range(200))


_ALL_DRAWS = [(t, s, q) for t in TIERS for s, q in _binomial_draws(t)]


def test_2061_small_population_shows_at_most_one_of_indep_and_const():
    small = 0
    both = []
    for t, s, q in _ALL_DRAWS:
        if "small population" not in q.text:
            continue
        small += 1
        opts = {q.answer_text, *q.distractors}
        if OPT_BINOMIAL_NOT_INDEP in opts and OPT_BINOMIAL_NOT_CONST_P in opts:
            both.append((t, s))
    assert small > 100, f"small-population scenario drawn only {small}/800 times"
    assert not both, f"{len(both)}/{small} draws show both dependence labels: {both[:5]}"


def test_2061_stated_start_share_is_feasible_or_a_count():
    checked = 0
    bad = []
    for t, s, q in _ALL_DRAWS:
        m_pop = re.search(r"population of (?:only )?(\d+)", q.text)
        if not m_pop:
            continue
        n_pop = int(m_pop.group(1))
        m_pct = re.search(r"success(?: on the first draw)? is (\d+)%", q.text)
        m_pct2 = re.search(r"draw is a success is (\d+)%", q.text)
        m_cnt = re.search(r"(\d+) of the (\d+) [a-z ]+ count as successes", q.text)
        if m_cnt:
            checked += 1
            if not (0 < int(m_cnt.group(1)) < n_pop and int(m_cnt.group(2)) == n_pop):
                bad.append((t, s))
            continue
        m = m_pct2 or m_pct
        if m:
            checked += 1
            x = int(m.group(1))
            if not any(abs(100 * k / n_pop - x) <= 0.5 for k in range(0, n_pop + 1)):
                bad.append((t, s, x, n_pop))
    assert checked > 150
    assert not bad, f"{len(bad)}/{checked} infeasible stated shares: {bad[:5]}"


def test_2061_no_first_draw_giveaway_and_no_announces_filler():
    for t, s, q in _ALL_DRAWS:
        lower = q.text.lower()
        assert "on the first draw" not in lower, (t, s, q.text)
        assert "announces the result" not in lower, (t, s, q.text)


def test_2061_inspector_only_in_production_contexts():
    seen = 0
    for t, s, q in _ALL_DRAWS:
        if "quality inspector" in q.text:
            seen += 1
            assert "factory quality-control line" in q.text, (t, s, q.text)
    assert seen > 0
    for t in TIERS:
        for seed in range(200):
            q = make("p19c_bernoulli_suitability", seed=seed, difficulty=t)
            if "quality inspector" in q.text:
                assert "factory quality-control line" in q.text, (t, seed, q.text)


def test_2061_supply_and_rating_wording_only_where_natural():
    for t, s, q in _ALL_DRAWS:
        if "basketball" in q.text:
            for bad in ("large supply", "-point rating", "rating scale"):
                assert bad not in q.text, (t, s, bad, q.text)
        if "school raffle" in q.text:
            for bad in ("-point rating", "rating scale", "time in seconds"):
                assert bad not in q.text, (t, s, bad, q.text)
        if "telemarketing" in q.text:
            assert "large supply" not in q.text, (t, s, q.text)
        if "without replacement" in q.text:
            assert "basketball" not in q.text, (t, s, q.text)


def test_2061_exam_large_population_states_the_approximation():
    seen = 0
    for s, q in _binomial_draws("exam"):
        m = re.search(r"large population of (\d+) \(much larger than the sample of (\d+)\)", q.text)
        if m:
            seen += 1
            assert int(m.group(1)) > 20 * int(m.group(2))
            assert "more than 20 times the sample" in q.text
            assert "treat the draws as independent" in q.text
    assert seen > 10


@pytest.mark.parametrize("difficulty", TIERS)
def test_2061_key_label_shares_between_15_and_40_percent(difficulty):
    counts = {}
    for s, q in _binomial_draws(difficulty):
        counts[q.answer_text] = counts.get(q.answer_text, 0) + 1
    for opt in OPTIONS_BINOMIAL:
        share = counts.get(opt, 0) / 200
        assert 0.15 <= share <= 0.40, f"{difficulty}: {opt!r} share {share:.1%} ({counts})"


# Wording cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_2063_binomial_never_repeats_a_sentence(difficulty):
    import re

    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        sentences = [x.strip().lower() for x in re.split(r"(?<=[.?])\s+", q.text) if x.strip()]
        assert len(sentences) == len(set(sentences)), q.text
        assert q.text.lower().count("until") <= 1, q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_2063_rating_scale_only_in_rated_contexts(difficulty):
    unrated = [dict(subject=x) for x in ("a basketball free-throw session", "a school raffle", "a plant nursery")]
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if "rating" in q.text:
            assert not any(c["subject"] in q.text for c in unrated), q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_2063_binomial_continuous_outcome_always_has_an_example(difficulty):
    for seed in range(200):
        q = make("p19c_binomial_suitability", seed=seed, difficulty=difficulty)
        if "continuous scale" in q.text:
            assert "(for example, " in q.text, q.text
