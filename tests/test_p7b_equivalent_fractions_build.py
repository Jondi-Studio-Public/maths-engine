"""Focused contract tests for: f10-equivalent-fractions-related-denominators."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7b_equivalent_fractions_build import (
    _denominator_word,
    _misconception_fractions,
    _verify_equivalent_fraction,
)

SEEDS = range(40)
SWEEP_SEEDS = range(60)
DIFFICULTIES = ["easy", "medium", "hard"]
TOPIC = "equivalent_fraction_related_denominator"

_FRACTION_TEXT_RE = re.compile(r"^\d+/\d+$")


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.to_json() == repeat.to_json()
        assert isinstance(q.answer, dict)
        assert {"p", "q"} <= set(q.answer)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_fraction(q, q.answer_text).correct, (
            f"seed {seed} {difficulty}: canonical {q.answer_text} not accepted"
        )
        for d in q.distractors:
            assert not grade_fraction(q, d).correct, f"seed {seed} {difficulty}: distractor {d} wrongly accepted"


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        correct = Fraction(q.answer["p"], q.answer["q"])
        wrong = correct + Fraction(1, 11)  # arbitrary offset, always a different value
        assert not grade_fraction(q, str(wrong)).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_equivalent_fraction(n=2, d=3, numerator=9, denominator=12)  # 9/12 != 2/3


def test_misconception_fractions_always_distinct_from_answer():
    """No distractor candidate is ever value-equal to the correct answer,
    for every (n, d, k) this generator can draw -- the specific failure
    mode this skill's grading discipline warns about twice."""
    from mathgen.generators.p7b_equivalent_fractions_build import (
        _DENOMS,
        _SCALE_FACTORS,
        _NAMED_PAIRS,
    )

    checked = 0
    pairs = [(d, k) for d in _DENOMS for k in _SCALE_FACTORS if k != d]
    pairs += list(_NAMED_PAIRS)
    for d, k in pairs:
        for n in range(1, d):
            D = d * k
            answer = Fraction(n * k, D)
            for candidate in _misconception_fractions(n, d, k, D):
                assert candidate != answer, f"n={n} d={d} k={k}: candidate {candidate} == answer {answer}"
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_a_positive_fraction_equivalent_to_a_proper_original(difficulty):
    """This skill's whole point is an UNREDUCED equivalent fraction at a
    related (larger) denominator -- p/q is deliberately not in lowest
    terms in general (e.g. 1/2 -> 3/6), so the answer kind this asserts is
    'a positive fraction equal in value to the original', not 'reduced'."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        p, qq = q.answer["p"], q.answer["q"]
        assert qq > 0 and p > 0
        assert q.answer_text == f"{p}/{qq}"


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_every_distractor_formatted_alike(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert _FRACTION_TEXT_RE.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _FRACTION_TEXT_RE.match(d), d


def test_easy_states_target_denominator_as_a_bare_number():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert "denominator" in q.text and "Write" in q.text
        assert re.search(r"denominator \d+\.$", q.text), q.text


def test_medium_never_states_target_denominator_as_a_digit():
    """The whole escalation is that the target denominator is named with a
    word, never shown as a number, in the medium tier."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert "Write" in q.text and " in " in q.text
        # After "in ", only the ordinal word should follow -- no digit.
        after_in = q.text.split(" in ", 1)[1]
        assert not re.search(r"\d", after_in), q.text


def test_hard_states_an_equation_with_the_new_fraction_missing():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "?/" in q.text and "=" in q.text


def test_never_asks_for_a_decimal_or_number_line_or_simplification():
    banned = ["decimal", "number line", "simplify", "simplest"]
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            lowered = q.text.lower()
            for word in banned:
                assert word not in lowered, f"seed {seed} {difficulty}: banned word {word!r} in {q.text}"


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in SWEEP_SEEDS}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SWEEP_SEEDS)} seeds"


def test_denominator_word_helper_matches_known_values():
    assert _denominator_word(6) == "sixths"
    assert _denominator_word(12) == "twelfths"
    assert _denominator_word(20) == "twentieths"
    assert _denominator_word(21) == "twenty-firsts"
    assert _denominator_word(24) == "twenty-fourths"
    assert _denominator_word(40) == "fortieths"


def test_skill_loads_through_api_with_correct_properties():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == "f10-equivalent-fractions-related-denominators"), None)
    assert skill is not None
    assert skill.domain == "primary-number"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "fraction"
    assert set(skill.prerequisites) == {"f10-unit-fractions-combine-whole", "compare-common-fractions"}
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert {0, 1, 2, 3, 4, 5, 6, "maintenance"} <= all_stages
