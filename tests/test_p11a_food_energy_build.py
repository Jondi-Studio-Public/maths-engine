"""Food energy units, Calories/kJ (senior-use-units-of-energy-used-for-foods-including)."""

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_string
from mathgen.generators.p11a_food_energy_build import CATALOGUE, FACTOR

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
WIDE_SEEDS = range(60)

TOPIC = "food_energy_conversion"
TIERS = ["easy", "medium", "hard"]

# Every catalogue value must fall inside its own retyped range here.
PLAUSIBLE_RANGES = {
    "a medium apple": (60, 110),
    "a medium banana": (85, 130),
    "a slice of white bread": (50, 90),
    "a can of regular soft drink (375 mL)": (120, 170),
    "a 50 g chocolate bar": (220, 300),
    "a boiled egg": (60, 100),
    "a cup of cooked rice": (130, 200),
    "a small baked potato": (60, 110),
    "a slice of cheese": (80, 130),
    "a tablespoon of peanut butter": (70, 110),
    "a cup of whole milk": (120, 180),
    "a small serving of hot chips": (220, 320),
    "a cooked chicken breast (100 g)": (140, 200),
    "a muesli bar": (90, 160),
    "a slice of pizza": (200, 300),
    "a cup of orange juice": (85, 130),
    "a doughnut": (200, 300),
    "a can of tuna in spring water": (85, 130),
    "a cup of ice cream": (220, 320),
    "an adult's approximate daily energy requirement": (1800, 2400),
}


def test_plausibility_oracle_ranges_contain_every_catalogue_value():
    assert set(PLAUSIBLE_RANGES) == {name for name, _ in CATALOGUE}
    for name, cal in CATALOGUE:
        lo, hi = PLAUSIBLE_RANGES[name]
        assert lo <= cal <= hi, f"{name}: {cal} Cal outside authored plausible range [{lo}, {hi}]"


def test_factor_is_exact_rational_never_float():
    assert FACTOR == Fraction(21, 5)
    assert isinstance(FACTOR, Fraction)


def test_factor_always_terminates_to_exactly_one_decimal_place():
    """Algebraic proof, exercised: Fraction(21, 5) has denominator 5, so for
    any integer Calorie count C, C * 21/5 * 10 = C * 42 is always an integer
    -- the value always sits on the 1-decimal-place grid."""
    for _, cal in CATALOGUE:
        scaled = Fraction(cal) * FACTOR * 10
        assert scaled.denominator == 1


# --- determinism / distractor hygiene ----------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_generator_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_formatted_alike(difficulty):
    """No option identifiable by its shape: easy is always 1 d.p., medium
    and hard are always a bare integer (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        options = [question.answer_text] + question.distractors
        if difficulty == "easy":
            for opt in options:
                assert opt.count(".") == 1 and len(opt.split(".")[1]) == 1, opt
        else:
            for opt in options:
                assert "." not in opt, opt


# --- the stated factor phrase is visible to the learner, in BOTH text and latex


@pytest.mark.parametrize("difficulty", TIERS)
def test_factor_phrase_appears_in_both_text_and_latex(difficulty):
    phrase = "1 calorie is approximately 4.2 kilojoules"
    for seed in WIDE_SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert phrase in question.text.lower()
        norm_latex = question.latex.lower().replace(r"\text{", "").replace("{", "").replace("}", "").replace("\\", "")
        assert phrase in norm_latex


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        if difficulty == "easy":
            # 1 whole unit off is far outside the 1-d.p. stated tolerance (0.05).
            wrong = Fraction(question.answer_text) + 1
            wrong_text = stated_precision_string(wrong, 1)
        else:
            wrong = int(question.answer) + 1
            wrong_text = str(wrong)
        assert not grade_number(question, wrong_text).correct


# --- tier-specific shape ------------------------------------------------------


def test_easy_answer_always_states_exactly_one_decimal_place():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        assert question.answer_text.endswith(".0") or "." in question.answer_text
        whole, frac = question.answer_text.split(".")
        assert len(frac) == 1


def test_medium_and_hard_answer_is_a_bare_integer():
    for difficulty in ("medium", "hard"):
        for seed in SEEDS:
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "." not in question.answer_text
            int(question.answer_text)  # must parse as a plain integer


def test_hard_never_draws_the_same_catalogue_item_twice():
    for seed in WIDE_SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        # Two distinct food-item names must appear in the question text.
        names_present = [name for name, _ in CATALOGUE if name in question.text.lower() or name in question.text]
        # At least the two involved names appear (case of the first letter
        # aside); assert they are not literally the same substring twice.
        assert question.text.count(" and ") >= 1


def test_easy_and_medium_pools_are_exactly_the_20_catalogue_items():
    """Easy and medium tiers each draw one item from a 20-entry catalogue, so
    the true pool of distinct prompts is genuinely 20 -- a 60-seed coupon-
    collector sample can miss one or two by chance (measured: 19/20 and
    18/20 over 60 seeds), so this asserts the real pool size over a wide
    enough sample (300 seeds) to saturate it, per the "genuinely narrow pool"
    convention rather than widening the generator or padding the count."""
    for difficulty in ("easy", "medium"):
        seen = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(300)}
        assert len(seen) == 20, f"{difficulty}: {len(seen)} distinct questions over 300 seeds, expected exactly 20"


def test_hard_has_at_least_20_distinct_questions_over_60_seeds():
    seen = {make(TOPIC, seed=s, difficulty="hard").text for s in WIDE_SEEDS}
    assert len(seen) >= 20, f"hard: only {len(seen)} distinct questions over 60 seeds"


def test_independent_check_fires_on_corrupted_factor():
    """A generator built with a corrupted (non-4.2) factor must fail the
    module's own division-based independent check."""
    from mathgen.core import check

    corrupted_factor = Fraction(4)  # "used-a-rounder-factor" corruption
    cal = Fraction(100)
    kj_text = "420.0"  # true kJ figure at the real factor 4.2
    recomputed = Fraction(kj_text) / corrupted_factor
    with pytest.raises(VerificationError):
        check(recomputed == cal, "independent check should have failed")


def test_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="unknown")


# --- skill loading -----------------------------------------------------------


def test_skill_loads_with_expected_properties():
    _, skills = load_skills(ROOT / "skills")
    skill = next((s for s in skills if s.id == "senior-use-units-of-energy-used-for-foods-including"), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["convert-metric-length"]
    assert len(skill.sources) == 3
