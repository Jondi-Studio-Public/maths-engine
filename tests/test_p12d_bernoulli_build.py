"""Tests `p12d_bernoulli_model` and `p12d_bernoulli_sequence`: using a Bernoulli random variable as a model, and
Bernoulli-sequence probabilities."""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

MODEL_TOPIC = "p12d_bernoulli_model"
MODEL_SKILL = "senior-use-a-bernoulli-random-variable-as-a-model"
SEQ_TOPIC = "p12d_bernoulli_sequence"
SEQ_EXAM_TOPIC = "exd_x_r2l_bernoulli_chain"
SEQ_SKILL = "senior-use-bernoulli-random-variables-and-associated-probabilities-to"

TOPICS = (MODEL_TOPIC, SEQ_TOPIC)


def _as_fraction(text: str) -> Fraction:
    if "/" in text:
        n, d = text.split("/")
        return Fraction(int(n), int(d))
    return Fraction(int(text))


# --- Determinism -------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_are_lowest_terms_fractions(topic, tier):
    frac_re = re.compile(r"^-?\d+(/\d+)?$")
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        for text in [q.answer_text, *q.distractors]:
            assert frac_re.match(text), f"seed={seed}: {text!r} not a plain fraction/integer"
            frac = _as_fraction(text)
            if "/" in text:
                n, d = text.split("/")
                assert Fraction(frac.numerator, frac.denominator) == Fraction(int(n), int(d))
                assert abs(int(n)) == frac.numerator and int(d) == frac.denominator, (
                    f"seed={seed}: {text!r} is not displayed in lowest terms"
                )


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = grade_fraction(q, q.answer_text)
        assert result.correct, f"seed={seed} tier={tier}: canonical not accepted"
        for d in q.distractors:
            result = grade_fraction(q, d)
            assert not result.correct, f"seed={seed} tier={tier}: distractor {d!r} accepted"
        correct_value = _as_fraction(q.answer_text)
        wrong_value = correct_value + Fraction(11, 13)  # odd offset, never coincides with a pool value
        wrong_text = f"{wrong_value.numerator}/{wrong_value.denominator}"
        result = grade_fraction(q, wrong_text)
        assert not result.correct, f"seed={seed} tier={tier}: deliberately wrong {wrong_text!r} accepted"


# --- 0 < p < 1 strictly, every seed, every tier (>= 200 draws swept) --------


@pytest.mark.parametrize("tier", TIERS)
def test_model_p_strictly_between_0_and_1(tier):
    """Every derived/recovered p is strictly between 0 and 1: for easy/medium
    parse the D/K spinner-style counts from the text; for hard/exam parse
    Var(X) and confirm both quadratic roots lie strictly in (0, 1)."""
    count = 0
    for seed in range(200):
        q = make(MODEL_TOPIC, seed=seed, difficulty=tier)
        if tier in ("easy", "medium"):
            # D is the total (sectors/tickets/members/marbles/cards/cars); K
            # is the count immediately preceding "of which"/"of whom"/"of
            # the cards" -- robust against the "numbered 1 to D" template's
            # extra embedded numbers, which a naive first-two-numbers scrape
            # would misparse.
            d_m = re.search(
                r"(\d+) (?:equal sectors|identical-looking raffle tickets|members|"
                r"marbles|cards is numbered 1 to \d+|parked cars)",
                q.text,
            )
            k_m = re.search(r"(\d+) of (?:which|whom|the cards)", q.text)
            assert d_m and k_m, f"tier={tier} seed={seed}: could not parse D/K from {q.text!r}"
            d, k = int(d_m.group(1)), int(k_m.group(1))
            p = Fraction(k, d)
            assert 0 < p < 1, f"tier={tier} seed={seed}: p={p} not strictly between 0 and 1"
        else:
            vm = re.search(r"Var\(X\) = (\d+)/(\d+)", q.text)
            v = Fraction(int(vm.group(1)), int(vm.group(2)))
            disc = 1 - 4 * v
            assert disc > 0, f"tier={tier} seed={seed}: discriminant not positive for Var(X)={v}"
        count += 1
    assert count == 200


@pytest.mark.parametrize("tier", TIERS)
def test_seq_p_strictly_between_0_and_1(tier):
    count = 0
    for seed in range(200):
        q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
        m = re.search(r"probability (\d+/\d+|0\.\d)", q.text)
        assert m, f"tier={tier} seed={seed}: no stated probability found in {q.text!r}"
        raw = m.group(1)
        p = Fraction(raw) if "/" in raw else Fraction(raw)
        assert 0 < p < 1, f"tier={tier} seed={seed}: stated probability {p} not strictly between 0 and 1"
        count += 1
    assert count == 200


# --- Hard/exam forced-condition sweep (Skill 1: Var != 1/4; Skill 2: q stated, not p) --


def test_model_hard_exam_var_never_one_quarter():
    for tier in ("hard", "exam"):
        n = 0
        for seed in range(300):
            q = make(MODEL_TOPIC, seed=seed, difficulty=tier)
            vm = re.search(r"Var\(X\) = (\d+)/(\d+)", q.text)
            v = Fraction(int(vm.group(1)), int(vm.group(2)))
            assert v != Fraction(1, 4), f"tier={tier} seed={seed}: Var(X) == 1/4 (double root)"
            n += 1
        assert n == 300


def test_seq_hard_exam_always_states_failure_probability():
    trap_markers = ("lands on 'loss'", "lands tails", "free of faults")
    for tier in ("hard", "exam"):
        n = 0
        for seed in range(300):
            q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
            assert any(marker in q.text for marker in trap_markers), (
                f"tier={tier} seed={seed}: hard/exam did not state the failure probability: {q.text!r}"
            )
            n += 1
        assert n == 300


# --- Never a combinatorial "x successes in n" question (the binomial trap) --


@pytest.mark.parametrize("tier", TIERS)
def test_seq_never_asks_for_a_count_of_successes(tier):
    forbidden = re.compile(r"exactly \d+ success|how many success", re.IGNORECASE)
    for seed in SEEDS:
        q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
        assert not forbidden.search(q.text), f"seed={seed}: {q.text!r} reads like a binomial-count question"
        assert "sequence of outcomes, in order" in q.text


# --- Never asks for expected value/estimation from sample data (the bernoulli-mle trap) --


@pytest.mark.parametrize("tier", TIERS)
def test_seq_never_asks_for_expected_value(tier):
    for seed in SEEDS:
        q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
        assert "E(X)" not in q.text and "expected value" not in q.text.lower()
        assert "Var(X)" not in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_seq_never_asks_to_estimate_p_from_a_sample(tier):
    for seed in SEEDS:
        q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
        assert "MLE" not in q.text and "estimate" not in q.text.lower()


# --- Independent-check tripwire -------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(Fraction(1, 3), Fraction(1, 4), "should fail")


@pytest.mark.parametrize("tier", TIERS)
def test_seq_independent_check_fires_on_a_corrupted_factor(tier):
    """Pull a rendered sequence
    question's own factors (parsed from its stated probability and outcome
    sequence, exactly as the oracle does), flip ONE factor to `1 - factor`,
    and confirm `check_equivalent` rejects the corrupted product -- proving
    the independent-check route in the generator would actually catch a
    real slip, not just that `check_equivalent` can fail in the abstract."""
    prob_re = re.compile(r"probability (\d+/\d+|0\.\d)")
    seq_re = re.compile(r"sequence of outcomes, in order, is: (.+)\.$")
    for seed in range(10):
        q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
        raw = prob_re.search(q.text).group(1)
        stated_p = Fraction(raw) if "/" in raw else Fraction(raw)
        # Determine, from the text itself, which label the stated probability
        # belongs to (the word immediately after "lands on '...'"/"lands
        # ...'/"independently ...").
        if "lands on '" in q.text:
            stated_label = re.search(r"lands on '(\w+)'", q.text).group(1)
        elif "lands heads" in q.text or "lands tails" in q.text:
            stated_label = re.search(r"lands (heads|tails)", q.text).group(1)
        elif "independently faulty" in q.text:
            stated_label = "faulty"
        else:
            stated_label = "ok"
        outcomes = seq_re.search(q.text).group(1).split(" then ")
        factors = [stated_p if o == stated_label else (1 - stated_p) for o in outcomes]
        correct_product = _as_fraction(q.answer_text)

        # Sanity: uncorrupted factors reproduce the canonical answer.
        product = Fraction(1)
        for f in factors:
            product *= f
        assert product == correct_product

        # Flip exactly one factor (skipping any factor == 1/2, where flipping
        # is a no-op -- possible when the stated p itself is 1/2 at
        # easy/medium) and confirm the check rejects it.
        flip_index = next((i for i, f in enumerate(factors) if f != Fraction(1, 2)), None)
        if flip_index is None:
            continue
        corrupted = list(factors)
        corrupted[flip_index] = 1 - corrupted[flip_index]
        corrupted_product = Fraction(1)
        for f in corrupted:
            corrupted_product *= f
        assert corrupted_product != correct_product
        with pytest.raises(VerificationError):
            check_equivalent(
                corrupted_product, correct_product, "corrupted factor list should not reproduce the answer"
            )


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Prompt variety: >= 3 distinct shapes per tier over 60 seeds -----------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_3_prompt_shapes_per_tier(topic, tier):
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(make(topic, seed=seed, difficulty=tier)) for seed in SEEDS}
    assert len(shapes) >= 3, f"{topic}/{tier}: only {len(shapes)} distinct prompt shapes over 60 seeds"


# --- Skill loading -----------------------------------------------------


def test_model_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == MODEL_SKILL), None)
    assert skill is not None, f"{MODEL_SKILL} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == [
        "simple-event-probability",
        "senior-identify-contexts-suitable-for-modelling-by-bernoulli-random",
    ]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == MODEL_TOPIC
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


def test_seq_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SEQ_SKILL), None)
    assert skill is not None, f"{SEQ_SKILL} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-event-probability"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == (SEQ_EXAM_TOPIC if source.difficulty == "exam" else SEQ_TOPIC)
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


# Answer-concentration sanity --------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_distribution_is_not_grossly_concentrated(topic, tier):
    seeds = range(400)
    answers = collections.Counter()
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        answers[q.answer_text] += 1
    total = sum(answers.values())
    top_text, top_n = answers.most_common(1)[0]
    share = top_n / total
    assert share < 0.25, f"{topic}/{tier}: top answer {top_text!r} is {share:.0%} of {total} draws"
