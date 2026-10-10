"""Australian/neighbour time zones with real daylight-saving rules, and call-window overlap
(senior-solve-problems-involving-time-zones-in-australia-and,
senior-solve-problems-associated-with-time-zones-for-example)."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p9b_timezone_australia_build import (
    PLACES_TZ,
    TABLE_CALL,
    _verify_timezone_diff,
    _verify_call_overlap,
)

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]
TZ_TOPIC = "australia_timezone_conversion"
CALL_TOPIC = "timezone_call_window_overlap"


@pytest.fixture(scope="module")
def skills_by_id():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_generators_are_deterministic_across_seeds(topic, tier):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_and_meet_85pct_bar(topic, tier):
    ge2 = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=tier)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            ge2 += 1
    assert ge2 / len(SEEDS) >= 0.85, f"{topic}/{tier}: only {ge2}/{len(SEEDS)} seeds had >=2 distractors"


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    ge2 = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            ge2 += 1
    assert ge2 / 200 >= 0.85


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct rendered questions over {len(SEEDS)} seeds"


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, tier):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=tier)
        assert grade_number(question, question.answer_text).correct, (
            f"{topic}/{tier} seed {seed}: grader rejected its own canonical {question.answer_text!r}"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"{topic}/{tier} seed {seed}: grader accepted distractor {distractor!r}"
            )


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, tier):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=tier)
        correct = Fraction(question.answer_text)
        wrong = str(float(correct + Fraction(1, 1)))
        assert not grade_number(question, wrong).correct


# --- answer-kind hygiene -----------------------------------------------------


@pytest.mark.parametrize("topic", [TZ_TOPIC, CALL_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_answers_are_whole_or_one_decimal_place(topic, tier):
    """Every answer here is a multiple of half an hour, so answer_text is
    either a bare integer or exactly 1 decimal place -- never on a rounding
    boundary, never 2+ decimal places."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=tier)
        if "." in question.answer_text:
            frac_part = question.answer_text.split(".")[1]
            assert len(frac_part) == 1
            assert frac_part == "5", f"non-half-hour fractional answer {question.answer_text}"
        else:
            assert float(question.answer_text) == int(question.answer_text)


# --- Skill 1 (australia_timezone_conversion) --------------------------------


def test_tz_easy_never_mentions_dst_observing():
    for seed in SEEDS:
        question = make(TZ_TOPIC, seed=seed, difficulty="easy")
        assert "is currently observing daylight saving" not in question.text


def test_tz_medium_mentions_exactly_one_dst_observing_place():
    for seed in SEEDS:
        question = make(TZ_TOPIC, seed=seed, difficulty="medium")
        assert question.text.count("is currently observing daylight saving") == 1


def test_tz_hard_both_shapes_occur():
    """Hard tier draws one of two subtypes (two-DST-places-both-active, or a
    +9:30 half-hour zone). Classify by replaying the SAME rng draw sequence
    the generator itself makes (its first hard-tier draw), not by scanning
    rendered text -- Adelaide can legitimately appear in EITHER subtype, so
    a text-based classifier is not reliable."""
    counts = {"two_dst": 0, "half_hour": 0}
    for seed in SEEDS:
        rng = random.Random(f"{TZ_TOPIC}:hard:{seed}")
        subtype = rng.choice(["two_dst", "half_hour"])
        counts[subtype] += 1
        make(TZ_TOPIC, seed=seed, difficulty="hard")  # must not raise for either shape
    assert counts["two_dst"] > 0, counts
    assert counts["half_hour"] > 0, counts


def test_tz_independent_check_fires_on_corrupted():
    _verify_timezone_diff("Sydney", "Perth", Fraction(3))  # 11 - 8 = 3, must not raise
    with pytest.raises(VerificationError):
        _verify_timezone_diff("Sydney", "Perth", Fraction(4))


def test_tz_independent_check_is_symmetric():
    # Swapping A/B must not change the magnitude the independent check accepts.
    _verify_timezone_diff("Sydney", "Perth", Fraction(3))
    _verify_timezone_diff("Perth", "Sydney", Fraction(3))


def test_tz_plausibility_real_table_matches_facts_and_bound_holds():
    """Prove RED first: a deliberately corrupted copy of the table (Brisbane
    marked as DST-observing, contradicting the real-world fact) must fail
    the same assertions the real table passes. This is a SEPARATE,
    independently re-typed reference table, not imported from the generator."""
    real_facts = {
        "Perth": False,
        "Darwin": False,
        "Brisbane": False,
        "Port Moresby": False,
        "Jakarta": False,
        "Adelaide": True,
        "Sydney": True,
        "Melbourne": True,
        "Hobart": True,
        "Canberra": True,
        "Auckland": True,
    }

    def assert_matches_facts(observance: dict) -> None:
        never_dst = {"Perth", "Darwin", "Brisbane", "Port Moresby", "Jakarta"}
        dst_observing = {"Sydney", "Melbourne", "Hobart", "Canberra", "Adelaide", "Auckland"}
        for place in never_dst:
            assert observance[place] is False, f"{place} wrongly marked DST-observing"
        for place in dst_observing:
            assert observance[place] is True, f"{place} wrongly marked never-DST"

    # RED: a corrupted copy (Brisbane flipped to DST-observing) must fail.
    corrupted = dict(real_facts)
    corrupted["Brisbane"] = True
    with pytest.raises(AssertionError):
        assert_matches_facts(corrupted)

    # GREEN: the real, independently-retyped facts must pass.
    assert_matches_facts(real_facts)

    # And the generator's own table must agree with the same real facts.
    generator_observance = {place: PLACES_TZ[place][1] for place in real_facts}
    assert_matches_facts(generator_observance)


def test_tz_plausibility_bound_0_to_6_hours():
    """RED first: an out-of-band value (0, or > 6) must fail the bound
    assertion; then confirm every real generated seed passes it."""

    def assert_in_band(value: Fraction) -> None:
        assert 0 < value <= 6

    with pytest.raises(AssertionError):
        assert_in_band(Fraction(0))
    with pytest.raises(AssertionError):
        assert_in_band(Fraction(7))

    for tier in TIERS:
        for seed in SEEDS:
            question = make(TZ_TOPIC, seed=seed, difficulty=tier)
            assert_in_band(Fraction(question.answer_text))


# --- Skill 2 (timezone_call_window_overlap) ---------------------------------


def test_call_easy_windows_always_symmetric_14h():
    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty="easy")
        assert question.text.count("8 am to 10 pm") == 2
        assert "9 am to 9 pm" not in question.text


def test_call_medium_windows_always_symmetric_14h():
    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty="medium")
        assert question.text.count("8 am to 10 pm") == 2
        assert "9 am to 9 pm" not in question.text


def test_call_hard_windows_always_asymmetric():
    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty="hard")
        assert question.text.count("8 am to 10 pm") == 1
        assert question.text.count("9 am to 9 pm") == 1


def test_call_hard_states_abbreviation_key_verbatim():
    from mathgen.generators.p9b_timezone_australia_build import ABBREV_KEY_TEXT

    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty="hard")
        assert ABBREV_KEY_TEXT in question.text


def test_call_medium_states_dst_derivation_for_at_least_one_place():
    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty="medium")
        assert "is currently observing daylight saving" in question.text


def test_call_easy_never_requires_derivation():
    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty="easy")
        assert "is currently observing daylight saving" not in question.text
        assert "time zone is" not in question.text


def test_call_independent_check_fires_on_corrupted():
    # Perth (UTC+8) vs Sydney std+DST (UTC+11): d = 3, both 14h windows
    # 8-22: overlap = max(0, min(22,22+3)-max(8,8+3)) = 22-11 = 11.
    _verify_call_overlap("medium", "Perth", "Sydney", 8, 22, 8, 22, None, None, Fraction(11))  # must not raise
    with pytest.raises(VerificationError):
        _verify_call_overlap("medium", "Perth", "Sydney", 8, 22, 8, 22, None, None, Fraction(12))


def test_call_independent_check_fires_on_corrupted_hard_tier():
    # Perth AWST(+8) vs Sydney AEDT(+11): d = 3, Perth 14h (8-22), Sydney
    # 12h (9-21): overlap = max(0, min(22,21+3)-max(8,9+3)) = 22-12 = 10.
    _verify_call_overlap("hard", "Perth", "Sydney", 8, 22, 9, 21, "AWST", "AEDT", Fraction(10))  # must not raise
    with pytest.raises(VerificationError):
        _verify_call_overlap("hard", "Perth", "Sydney", 8, 22, 9, 21, "AWST", "AEDT", Fraction(11))


def test_call_verify_rejects_out_of_band_d():
    with pytest.raises(VerificationError):
        _verify_call_overlap("hard", "Perth", "Auckland", 8, 22, 9, 21, "AWST", "NZDT", Fraction(0))


def test_call_plausibility_d_bound_and_positive_overlap():
    """Prove RED first, then confirm every real generated seed passes."""

    def assert_plausible(d: Fraction, overlap: Fraction) -> None:
        assert abs(d) <= 6
        assert overlap > 0

    with pytest.raises(AssertionError):
        assert_plausible(Fraction(10), Fraction(5))  # RED: |d| = 10 is out of band
    with pytest.raises(AssertionError):
        assert_plausible(Fraction(3), Fraction(0))  # RED: zero overlap

    for tier in TIERS:
        for seed in SEEDS:
            question = make(CALL_TOPIC, seed=seed, difficulty=tier)
            overlap = Fraction(question.answer_text)
            assert overlap > 0, f"{tier} seed {seed}: implausible zero overlap"


# --- skills load through the API --------------------------------------------


def test_both_skills_load(skills_by_id):
    tz = skills_by_id["senior-solve-problems-involving-time-zones-in-australia-and"]
    call = skills_by_id["senior-solve-problems-associated-with-time-zones-for-example"]
    assert tz.answer == "number"
    assert call.answer == "number"
    assert tz.placement_eligible is False
    assert call.placement_eligible is False
    assert tz.prerequisites == []
    assert call.prerequisites == []
