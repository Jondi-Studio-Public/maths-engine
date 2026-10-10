"""Approximate confidence intervals for a mean
(senior-examine-the-approximate-confidence-interval-left-overline-mathrm, ACMSM141) and a proportion
(senior-use-the-approximate-confidence-interval-left-widehat-p, ACMMM178).

The single most important test here is
`test_boundary_vocabulary_never_crosses_between_skills`: the two skills'
question text must be distinguishable from the text alone, never merely from
the skill id.
"""

import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import GRADER_TOLERANCE, VerificationError
from mathgen.generators.p8c_confidence_interval_build import _verify_interval

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]

MEAN_TOPIC = "p8c_confidence_interval_mean"
PROP_TOPIC = "p8c_confidence_interval_proportion"
MEAN_ID = "senior-examine-the-approximate-confidence-interval-left-overline-mathrm"
PROP_ID = "senior-use-the-approximate-confidence-interval-left-widehat-p"

_PAIR_RE = re.compile(r"lower = (-?[\d.]+), upper = (-?[\d.]+)")


def _pair_values(text: str) -> tuple[Fraction, Fraction]:
    m = _PAIR_RE.match(text)
    assert m, f"unparseable pair answer: {text!r}"
    return Fraction(m.group(1)), Fraction(m.group(2))


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("topic", [MEAN_TOPIC, PROP_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [MEAN_TOPIC, PROP_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_and_clear_grader_tolerance(topic, difficulty):
    """Distractors must be non-empty, distinct, never equal the answer, and --
    since this is a `pair` skill graded with a FIXED 0.051 tolerance -- every
    distractor pair must differ from the correct pair by more than
    GRADER_TOLERANCE in at least one component, checked with Fraction
    arithmetic on the values as rendered (never a float ==/subtraction)."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) == 3
        lo_a, hi_a = _pair_values(question.answer_text)
        for distractor in question.distractors:
            lo_d, hi_d = _pair_values(distractor)
            gap = abs(lo_d - lo_a) > GRADER_TOLERANCE or abs(hi_d - hi_a) > GRADER_TOLERANCE
            assert gap, (
                f"{topic} {difficulty} seed={seed}: distractor {distractor!r} "
                f"is within grader tolerance of answer {question.answer_text!r}"
            )


# --- Real grader testing -----------------------------------------------------


@pytest.mark.parametrize("topic", [MEAN_TOPIC, PROP_TOPIC])
def test_real_grader_accepts_canonical_rejects_distractors(topic):
    for difficulty in TIERS:
        for seed in SEEDS:
            question = make(topic, seed=seed, difficulty=difficulty)
            assert grade_pair(question, question.answer_text).correct, (
                f"{topic} {difficulty} seed={seed}: canonical answer rejected"
            )
            for distractor in question.distractors:
                assert not grade_pair(question, distractor).correct, (
                    f"{topic} {difficulty} seed={seed}: distractor {distractor!r} accepted"
                )


@pytest.mark.parametrize("topic", [MEAN_TOPIC, PROP_TOPIC])
def test_real_grader_rejects_deliberately_wrong_answer(topic):
    for difficulty in TIERS:
        for seed in SEEDS:
            question = make(topic, seed=seed, difficulty=difficulty)
            wrong = "lower = 999.9, upper = 1000.9"
            assert not grade_pair(question, wrong).correct, f"{topic} {difficulty} seed={seed}: wrong answer accepted"


# --- Independent check fires on a corrupted answer ---------------------------


def test_verify_interval_fires_on_corrupted_bounds():
    # Correct: center 100, margin (independent) 5 -> (95, 105).
    _verify_interval(100.0, 95.0, 105.0, 5.0)
    with pytest.raises(VerificationError):
        _verify_interval(100.0, 95.0, 105.0, 6.0)  # margin identity broken
    with pytest.raises(VerificationError):
        _verify_interval(100.0, 96.0, 105.0, 5.0)  # midpoint identity broken (perturbed bound)
    with pytest.raises(VerificationError):
        _verify_interval(100.0, 105.0, 95.0, 5.0)  # flipped sign (bounds swapped)


# --- Answer kind / precision --------------------------------------------------


@pytest.mark.parametrize("topic", [MEAN_TOPIC, PROP_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_pair_with_exactly_one_decimal_place(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, dict)
        assert list(question.answer.keys()) == ["lower", "upper"]
        for label, value in question.answer.items():
            text = f"{value:.10f}".rstrip("0").rstrip(".")
            decimals = len(text.split(".")[1]) if "." in text else 0
            assert decimals <= 1, f"{topic} {difficulty} seed={seed}: {label}={value} not 1dp"
        # Every rendered number in answer_text/distractors carries exactly 1dp,
        # so no option is identifiable by shape (midpoint_coordinate_scalar's
        # own defect).
        for text in [question.answer_text, *question.distractors]:
            for token in re.findall(r"-?\d+\.\d+", text):
                assert len(token.split(".")[1]) == 1


# --- Distinctness of the draw space ------------------------------------------


@pytest.mark.parametrize("topic", [MEAN_TOPIC, PROP_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{topic} {difficulty}: only {len(texts)} distinct questions/60 seeds"


# --- The cross-skill vocabulary boundary (THE test this file exists for) ----


def test_boundary_vocabulary_never_crosses_between_skills():
    """Skill A's text must never mention the proportion vocabulary, and skill
    B's text must never mention the mean/std-dev vocabulary, over a >=40-seed,
    every-tier sweep -- distinguishable from the QUESTION TEXT alone."""
    forbidden_in_mean = ["proportion", "phat", "p-hat", "p̂"]
    forbidden_in_prop = ["mean", "standard deviation"]
    checked_mean = checked_prop = 0
    for difficulty in TIERS:
        for seed in SEEDS:
            mean_q = make(MEAN_TOPIC, seed=seed, difficulty=difficulty)
            text_lower = mean_q.text.lower()
            for bad in forbidden_in_mean:
                assert bad not in text_lower, f"mean skill seed={seed} {difficulty}: found {bad!r} in {mean_q.text!r}"
            checked_mean += 1

            prop_q = make(PROP_TOPIC, seed=seed, difficulty=difficulty)
            text_lower = prop_q.text.lower()
            for bad in forbidden_in_prop:
                assert bad not in text_lower, (
                    f"proportion skill seed={seed} {difficulty}: found {bad!r} in {prop_q.text!r}"
                )
            checked_prop += 1
    assert checked_mean == len(TIERS) * len(SEEDS)
    assert checked_prop == len(TIERS) * len(SEEDS)


def test_boundary_required_vocabulary_present():
    """Positive half of the boundary: each skill's own required words are
    actually present, every seed, every tier."""
    for difficulty in TIERS:
        for seed in SEEDS:
            mean_q = make(MEAN_TOPIC, seed=seed, difficulty=difficulty)
            t = mean_q.text
            assert "sample mean" in t and "population mean" in t and "mu" in t
            assert ("sample standard deviation" in t) or ("sample variance" in t)

            prop_q = make(PROP_TOPIC, seed=seed, difficulty=difficulty)
            t = prop_q.text
            assert "sample proportion" in t and "population proportion" in t
            assert "phat" in t
            assert re.search(r"\bp\b", t)


# --- Skill contracts load with the expected shape ----------------------------


def test_confidence_interval_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    mean_skill = by_id[MEAN_ID]
    prop_skill = by_id[PROP_ID]

    for skill, topic in [(mean_skill, MEAN_TOPIC), (prop_skill, PROP_TOPIC)]:
        assert skill.prerequisites == ["simple-event-probability"]
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert skill.status == "active"
        assert {source.topic for source in skill.sources} == {topic}
        assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]
        assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"]]
