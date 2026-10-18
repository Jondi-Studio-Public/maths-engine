"""Tests `reasonableness_verdict` and `locate_calculation_error`: determinism, fixed option sets, grading and an
independent check.

reasonableness_verdict (ACMEM003): judge a stated result's plausibility, a
verdict never a recomputation, gross errors only.
locate_calculation_error (ACMEM006): find which step of a 3-step worked
calculation is wrong, a local slip a magnitude-only glance cannot catch.
"""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10b_checking_results_build import (
    OPTIONS_A,
    OPTIONS_B,
    OPTION_REASONABLE,
    OPTION_TOO_LARGE,
    OPTION_TOO_SMALL,
    OPTION_WRONG_SIGN,
    OPTION_STEP1,
    OPTION_STEP2,
    OPTION_STEP3,
    OPTION_NO_ERROR,
    _independent_check_a,
    _independent_check_b,
)

SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", ["reasonableness_verdict", "locate_calculation_error"])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == 3
        assert len(question.distractors) == len(set(question.distractors))


# --- reasonableness_verdict (ACMEM003) --------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_reasonableness_verdict_answer_is_one_of_four_fixed_options(difficulty):
    for seed in SEEDS:
        question = make("reasonableness_verdict", seed=seed, difficulty=difficulty)
        assert question.answer_text in OPTIONS_A
        assert set(question.distractors) | {question.answer_text} == set(OPTIONS_A)


@pytest.mark.parametrize("difficulty", TIERS)
def test_reasonableness_verdict_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make("reasonableness_verdict", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(question, distractor_index).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_reasonableness_verdict_true_result_is_always_a_multiple_of_ten(difficulty):
    """Construction's own invariant: true is always an exact multiple of 10,
    so 'too small' (true / 10) is always an exact integer, never a fraction."""
    pattern = re.compile(r"states: (.+?) = (-?\d+)\.")
    for seed in SEEDS:
        question = make("reasonableness_verdict", seed=seed, difficulty=difficulty)
        expr, stated = pattern.search(question.text).groups()
        true = eval(expr.replace("×", "*"), {"__builtins__": {}}, {})
        assert true % 10 == 0, f"seed {seed} {difficulty}: true {true} not a multiple of 10"


@pytest.mark.parametrize("difficulty", TIERS)
def test_reasonableness_verdict_independent_check_fires_on_a_wrong_ratio(difficulty):
    """Corrupt the stated value so its ratio to true is not one of the four
    sanctioned ratios, and confirm the independent check raises."""
    for seed in SEEDS:
        question = make("reasonableness_verdict", seed=seed, difficulty=difficulty)
        pattern = re.compile(r"states: (.+?) = (-?\d+)\.")
        expr, stated = pattern.search(question.text).groups()
        true = eval(expr.replace("×", "*"), {"__builtins__": {}}, {})
        stated = int(stated)
        # A ratio of 3 is never one of {1, 10, 1/10, -1}.
        corrupted_stated = true * 3 if true != 0 else 3
        with pytest.raises(VerificationError):
            _independent_check_a(expr, true, corrupted_stated, OPTION_REASONABLE)


@pytest.mark.parametrize("difficulty", TIERS)
def test_reasonableness_verdict_independent_check_fires_on_mismatched_label(difficulty):
    """A correct ratio (1) mapped to the WRONG option must also raise."""
    for seed in SEEDS:
        question = make("reasonableness_verdict", seed=seed, difficulty=difficulty)
        pattern = re.compile(r"states: (.+?) = (-?\d+)\.")
        expr, stated = pattern.search(question.text).groups()
        true = eval(expr.replace("×", "*"), {"__builtins__": {}}, {})
        with pytest.raises(VerificationError):
            _independent_check_a(expr, true, true, OPTION_TOO_LARGE)


def test_reasonableness_verdict_class_balance_at_400_seeds():
    """Report the class balance at >=300 seeds, never 40 -- a fair 4-way draw's minority class dips under 20%
    about as often as a coin's 40-draw majority exceeds 55%."""
    for difficulty in TIERS:
        counts = Counter(
            make("reasonableness_verdict", seed=seed, difficulty=difficulty).answer_text for seed in range(400)
        )
        assert set(counts) == set(OPTIONS_A)
        for option, n in counts.items():
            assert 0.15 <= n / 400 <= 0.35, f"{difficulty} {option!r}: {n}/400 = {n / 400:.2%}"


# --- locate_calculation_error (ACMEM006) ------------------------------------


_STEP_RE = re.compile(r"Step (\d): (\d+) (\+|-|×) (\d+) = (\d+)\.")


def _parse_steps(text):
    """[(index, a, op, b, stated), ...] from the rendered step lines."""
    return [(int(idx), int(a), op, int(b), int(stated)) for idx, a, op, b, stated in _STEP_RE.findall(text)]


@pytest.mark.parametrize("difficulty", TIERS)
def test_locate_calculation_error_answer_is_one_of_four_fixed_options(difficulty):
    for seed in SEEDS:
        question = make("locate_calculation_error", seed=seed, difficulty=difficulty)
        assert question.answer_text in OPTIONS_B
        assert set(question.distractors) | {question.answer_text} == set(OPTIONS_B)


@pytest.mark.parametrize("difficulty", TIERS)
def test_locate_calculation_error_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make("locate_calculation_error", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(question, distractor_index).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_locate_calculation_error_exactly_one_step_mismatches_its_own_recomputation(difficulty):
    """The invariant the independent check enforces: recompute every step from
    its own two stated operands; exactly one mismatches for an error class,
    zero for 'no error', and the mismatching index equals the drawn class."""
    for seed in SEEDS:
        question = make("locate_calculation_error", seed=seed, difficulty=difficulty)
        steps = _parse_steps(question.text)
        assert len(steps) == 3
        mismatches = []
        for idx, a, op, b, stated in steps:
            true = {"+": a + b, "-": a - b, "×": a * b}[op]
            if true != stated:
                mismatches.append(idx)
        if question.answer_text == OPTION_NO_ERROR:
            assert mismatches == []
        else:
            expected_index = {OPTION_STEP1: 1, OPTION_STEP2: 2, OPTION_STEP3: 3}[question.answer_text]
            assert mismatches == [expected_index]


@pytest.mark.parametrize("difficulty", TIERS)
def test_locate_calculation_error_every_step_true_result_at_least_the_floor(difficulty):
    """The floor construction keeps the relative slip comfortably under 30%."""
    floor = 300 if difficulty == "hard" else 40
    for seed in SEEDS:
        question = make("locate_calculation_error", seed=seed, difficulty=difficulty)
        for idx, a, op, b, stated in _parse_steps(question.text):
            true = {"+": a + b, "-": a - b, "×": a * b}[op]
            assert true >= floor, f"seed {seed} {difficulty} step {idx}: true {true} < floor {floor}"


def test_locate_calculation_error_medium_and_hard_always_mix_operations():
    """Medium/hard: at least one x step among the three -- genuine escalation
    over easy's +/- only."""
    for difficulty in ("medium", "hard"):
        for seed in SEEDS:
            question = make("locate_calculation_error", seed=seed, difficulty=difficulty)
            ops = {op for _, _, op, _, _ in _parse_steps(question.text)}
            assert "×" in ops, f"seed {seed} {difficulty}: no x step in {ops}"


def test_locate_calculation_error_easy_is_additive_only():
    for seed in SEEDS:
        question = make("locate_calculation_error", seed=seed, difficulty="easy")
        ops = {op for _, _, op, _, _ in _parse_steps(question.text)}
        assert ops <= {"+", "-"}, f"seed {seed} easy: unexpected op in {ops}"


def test_locate_calculation_error_independent_check_fires_on_a_corrupted_step():
    """Feed a step list with a mismatch at a step other than the drawn class
    (or a mismatch when none was drawn) and confirm the check raises."""
    steps = [
        dict(index=1, first=100, second=30, op="+", true=130, stated=130),
        dict(index=2, first=130, second=20, op="-", true=110, stated=110),
        dict(index=3, first=110, second=5, op="+", true=115, stated=115),
    ]
    # Correct as-is, drawn class None (no error): should not raise.
    _independent_check_b(steps, None)

    # Corrupt step 2's stated value without telling the check: now it must raise.
    corrupted = [dict(s) for s in steps]
    corrupted[1] = dict(corrupted[1], stated=corrupted[1]["true"] + 10)
    with pytest.raises(VerificationError):
        _independent_check_b(corrupted, None)

    # Declaring the WRONG drawn class for a genuinely-corrupted step 2 also raises.
    with pytest.raises(VerificationError):
        _independent_check_b(corrupted, 3)

    # Declaring the correct drawn class for the same corruption does not raise.
    _independent_check_b(corrupted, 2)


def test_locate_calculation_error_class_balance_at_400_seeds():
    for difficulty in TIERS:
        counts = Counter(
            make("locate_calculation_error", seed=seed, difficulty=difficulty).answer_text for seed in range(400)
        )
        assert set(counts) == set(OPTIONS_B)
        for option, n in counts.items():
            assert 0.15 <= n / 400 <= 0.35, f"{difficulty} {option!r}: {n}/400 = {n / 400:.2%}"


# --- the ACMEM003-vs-ACMEM006 cross-check -----------------------------------


def test_acmem006_local_slip_never_lands_in_acmem003_gross_error_ratio_set():
    """The cross-check this pairing exists to own: for every seed where
    locate_calculation_error draws an erroneous step, the ratio of that one
    step's stated to true result must NEVER be one of reasonableness_verdict's
    own gross-error ratios {10, 1/10, -1} -- proving ACMEM006's local slip is
    never accidentally a reasonableness-detectable gross error, i.e. the two
    skills genuinely test different things on the same kind of object."""
    gross_error_ratios = {10, Fraction(1, 10), -1}
    checked = 0
    for difficulty in TIERS:
        for seed in range(200):
            question = make("locate_calculation_error", seed=seed, difficulty=difficulty)
            for idx, a, op, b, stated in _parse_steps(question.text):
                true = {"+": a + b, "-": a - b, "×": a * b}[op]
                if true == stated:
                    continue
                checked += 1
                ratio = Fraction(stated, true)
                assert ratio not in gross_error_ratios, (
                    f"seed {seed} {difficulty} step {idx}: ratio {ratio} is one of "
                    f"ACMEM003's own gross-error ratios -- the local slip is not local"
                )
    assert checked >= 200, f"only {checked} erroneous steps found over the sweep; widen it"


# --- skills loader -----------------------------------------------------------


def test_both_skills_load_via_api_skills():
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}

    skill_a = skills_dict.get("senior-ascertain-the-reasonableness-of-answers-to-arithmetic-calculations")
    assert skill_a is not None
    assert skill_a.answer == "mcq"
    assert "order-of-operations" in skill_a.prerequisites
    assert skill_a.placement_eligible is False

    skill_b = skills_dict.get("senior-check-results-of-calculations-for-accuracy")
    assert skill_b is not None
    assert skill_b.answer == "mcq"
    assert "order-of-operations" in skill_b.prerequisites
    assert skill_b.placement_eligible is False
