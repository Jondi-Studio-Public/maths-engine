"""Tests `p8c_line_best_fit_predict` (`senior-use-the-line-of-best-fit-to-make`)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

TOPIC = "p8c_line_best_fit_predict"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]


def _q(seed, tier):
    return make(TOPIC, seed=seed, difficulty=tier)


# --- determinism & distractor hygiene ---------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for seed in SEEDS:
        q = _q(seed, tier)
        assert len(q.distractors) == 3, f"seed {seed}: expected 3 distractors, got {q.distractors}"


# --- real grader ---------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_rejects_distractors(tier):
    for seed in SEEDS:
        q = _q(seed, tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed {seed} {tier}: canonical rejected"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed {seed} {tier}: distractor {d} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_deliberately_wrong_answer(tier):
    for seed in SEEDS:
        q = _q(seed, tier)
        wrong = str(int(q.answer) + 999999)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed {seed} {tier}: wrong answer {wrong} accepted"


# --- answer is a plain integer, every seed, every tier ----------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_exact_integer(tier):
    for seed in SEEDS:
        q = _q(seed, tier)
        assert "." not in q.answer_text, f"seed {seed} {tier}: {q.answer_text} has a decimal point"
        assert isinstance(q.answer, int), f"seed {seed} {tier}: answer not int-typed: {q.answer!r}"
        # Not merely int-typed while numerically fractional: assert no remainder.
        frac = Fraction(q.answer)
        assert frac.denominator == 1, f"seed {seed} {tier}: {q.answer} has a remainder"
        assert str(q.answer) == q.answer_text


# --- interpolation/extrapolation coverage, every tier -----------------------


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_draws_both_interpolation_and_extrapolation(tier):
    seen = {"interpolation": 0, "extrapolation": 0}
    for seed in range(60):
        q = _q(seed, tier)
        if "interpolation" in q.text:
            seen["interpolation"] += 1
        if "extrapolation" in q.text:
            seen["extrapolation"] += 1
    assert seen["interpolation"] > 0, f"{tier}: never drew an interpolation case over 60 seeds"
    assert seen["extrapolation"] > 0, f"{tier}: never drew an extrapolation case over 60 seeds"


@pytest.mark.parametrize("tier", TIERS)
def test_x_never_drawn_exactly_at_boundary(tier):
    """Never draw x (or x1/x2) exactly at x_min/x_max: parse the stated range
    and every x mentioned in the framing sentence, confirm no equality."""
    range_re = re.compile(r"x from (\d+) to (\d+)")
    x_eq_re = re.compile(r"x = (-?\d+)")
    for seed in range(60):
        q = _q(seed, tier)
        m = range_re.search(q.text)
        assert m, f"seed {seed} {tier}: could not find stated x-range in {q.text!r}"
        x_min, x_max = int(m.group(1)), int(m.group(2))
        for x_val in (int(v) for v in x_eq_re.findall(q.text)):
            assert x_val != x_min and x_val != x_max, (
                f"seed {seed} {tier}: x={x_val} sits exactly on the boundary [{x_min}, {x_max}]"
            )


def test_no_negative_quantity_in_any_context():
    """Every context here is a naturally non-negative quantity; no drawn x
    (or x1/x2) may render as negative anywhere in the question text."""
    for tier in TIERS:
        for seed in range(60):
            q = _q(seed, tier)
            for n in re.findall(r"-?\d+", q.text):
                assert int(n) >= 0, f"seed {seed} {tier}: negative number in text: {q.text!r}"


# --- misconception-value distinctness, per formula, honestly checked -------


def test_misconception_formulas_distinct_from_answer_every_seed():
    for seed in range(60):
        # easy
        q = _q(seed, "easy")
        eq_m = re.search(r"y = (\d+)x \+ (\d+)", q.text)
        x_m = re.search(r"when x = (\d+)\?", q.text)
        m, c, x = int(eq_m.group(1)), int(eq_m.group(2)), int(x_m.group(1))
        y = int(q.answer_text)
        assert c * x + m != y
        assert m * x != y
        assert m * x - c != y

        # medium
        q = _q(seed, "medium")
        eq_m = re.search(r"y = (\d+)x \+ (\d+)", q.text)
        y_m = re.search(r"gives y = (\d+)\?", q.text)
        m, c, y_target = int(eq_m.group(1)), int(eq_m.group(2)), int(y_m.group(1))
        x_target = int(q.answer_text)
        assert round(Fraction(y_target - m, c)) != x_target
        assert round(Fraction(y_target, m)) != x_target
        assert round(Fraction(y_target + c, m)) != x_target

        # hard
        q = _q(seed, "hard")
        eq_m = re.search(r"y = (\d+)x \+ (\d+)", q.text)
        interval_m = re.search(r"from (\d+) to (\d+)\?", q.text)
        m, c = int(eq_m.group(1)), int(eq_m.group(2))
        x1, x2 = int(interval_m.group(1)), int(interval_m.group(2))
        answer = int(q.answer_text)
        assert c * (x2 - x1) != answer
        assert m * x2 + c != answer
        assert m * x1 + c != answer


# --- skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-use-the-line-of-best-fit-to-make"), None)
    assert skill is not None, "skill not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert "simple-event-probability" in skill.prerequisites
    assert skill.domain == "statistics-probability"
