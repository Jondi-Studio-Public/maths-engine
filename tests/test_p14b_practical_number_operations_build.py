"""See the generator module's docstring for the skill's scope and the duplicate-separation argument against
`f10-additive-multiplicative-money-model` and `percentage-discount`."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, stated_precision_tolerance
from mathgen.generators.p14b_practical_number_operations_build import (
    _deconflict_cents,
    _draw_price_cents,
    _wrong_direction_value,
)

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p14b_practical_number_operations"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
WIDE_SEEDS = range(60)

# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# distractor hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3, q.distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_clear_the_stated_precision_gap(difficulty):
    """Every distractor differs from the answer by more than
    `stated_precision_tolerance(2)` (0.005) -- the 2-d.p. `answer: number`
    discipline."""
    tol = stated_precision_tolerance(2)
    for seed in WIDE_SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        answer_value = Fraction(q.answer_text)
        for d in q.distractors:
            gap = abs(Fraction(d) - answer_value)
            assert gap > tol, f"{q.text}: distractor {d!r} only {gap} from answer {q.answer_text!r}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_formatted_alike(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        options = [q.answer_text] + q.distractors
        assert all("." in opt for opt in options), f"{q.text}: not every option states cents precision: {options}"
        assert all(len(opt.split(".")[1]) == 2 for opt in options), (
            f"{q.text}: not every option states 2 d.p.: {options}"
        )


# ---------------------------------------------------------------------------
# real grader accept/reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = str(Fraction(q.answer_text) + 1000)
        assert not grade_number(q, wrong).correct, q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_a_dollar_sign_prefixed_typed_answer(difficulty):
    """`grade_number`'s tokenizer drops a leading `$` (neither digit nor
    letter), so "$4.35" is accepted for a canonical "4.35"."""
    for seed in WIDE_SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(q, f"${q.answer_text}").correct, q.text


# ---------------------------------------------------------------------------
# the two in-generator checks
# ---------------------------------------------------------------------------


def test_wrong_direction_value_independent_formula_matches_direct_fraction_route():
    """`_wrong_direction_value`'s integer-division route must agree with a
    plain `Fraction` computation of the same "apply the opposite direction"
    value, over a sweep of (price, pct, op) combinations."""
    for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        for op in ["add", "sub"]:
            price_cents, _ = _draw_price_cents(__import__("random").Random(f"probe:{pct}:{op}"), pct, op, 1000, 50000)
            other_op = "sub" if op == "add" else "add"
            factor = 100 - pct if other_op == "sub" else 100 + pct
            expected = Fraction(price_cents) * factor / 100
            check_equivalent(
                _wrong_direction_value(price_cents, pct, op), expected, "wrong-direction independent recompute mismatch"
            )


def test_check_equivalent_fires_on_a_corrupted_wrong_direction_value():
    price_cents, _ = _draw_price_cents(__import__("random").Random("probe"), 10, "sub", 1000, 50000)
    true_value = _wrong_direction_value(price_cents, 10, "sub")
    with pytest.raises(VerificationError):
        check_equivalent(true_value + 1, true_value, "should have failed")


def test_deconflict_cents_guarantees_distinctness_and_preserves_order_when_already_distinct():
    assert _deconflict_cents(100, 200, 300, 400) == [200, 300, 400]
    # a coincidental collision (candidate 2 == candidate 3 == the answer) is
    # nudged forward by whole cents, deterministically, never randomly:
    assert _deconflict_cents(100, 100, 100, 100) == [101, 102, 103]


# ---------------------------------------------------------------------------
# answer satisfies its own declared kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_always_states_exactly_two_decimal_places(difficulty):
    for seed in WIDE_SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "." in q.answer_text, q.answer_text
        whole, frac = q.answer_text.split(".")
        assert len(frac) == 2, q.answer_text


# ---------------------------------------------------------------------------
# tier-specific shape
# ---------------------------------------------------------------------------


def test_easy_always_exactly_one_percentage_and_one_flat_step():
    for seed in WIDE_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert len(q.working) == 2, q.working
        assert "Percentage step" in q.working[0]
        assert "Flat step" in q.working[1]


def test_medium_is_either_split_or_multiply_and_both_are_live():
    seen_split, seen_multiply = 0, 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if "split equally" in q.text:
            seen_split += 1
            assert "Split step" in q.working[-1]
        elif "buying" in q.text:
            seen_multiply += 1
            assert "Multiply step" in q.working[-1]
        else:
            pytest.fail(f"medium question matched neither shape: {q.text!r}")
    assert seen_split > 0 and seen_multiply > 0, (seen_split, seen_multiply)


def test_hard_is_always_a_three_step_chain():
    for seed in WIDE_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert len(q.working) == 3, q.working
        assert "Percentage step" in q.working[0]
        assert "Flat step" in q.working[1]
        assert "Split step" in q.working[2]


def test_exam_is_the_hard_chain_plus_a_stated_red_herring():
    for seed in WIDE_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert len(q.working) == 4, q.working
        assert "irrelevant quantity" in q.working[0]
        assert "(" in q.text and ")" in q.text, q.text


def test_no_tier_ever_produces_a_two_percentage_only_chain():
    """A two-percentage chain would read as
    `percentage-discount`'s own chained-discount hard tier. Every tier's
    `working` names exactly one "Percentage step" line."""
    for difficulty in TIERS:
        for seed in WIDE_SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            pct_lines = [w for w in q.working if w.startswith("Percentage step")]
            assert len(pct_lines) == 1, f"{difficulty} seed {seed}: {q.working}"


def test_every_amount_is_cents_precision_never_whole_dollars_only():
    """The other named separation from `f10-additive-multiplicative-money-
    model`: every dollar figure appearing in the prompt text carries 2 d.p.,
    never a bare whole-dollar `$N`."""
    import re

    bare_whole_dollar_re = re.compile(r"\$(\d+)(?!\.\d)\b")
    for difficulty in TIERS:
        for seed in WIDE_SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert not bare_whole_dollar_re.search(q.text), f"{difficulty} seed {seed}: {q.text}"


# ---------------------------------------------------------------------------
# variety
# ---------------------------------------------------------------------------


def test_at_least_three_prompt_shapes_per_tier():
    from mathgen.text_quality import prompt_shape

    for difficulty in TIERS:
        shapes = {prompt_shape(make(TOPIC, seed=s, difficulty=difficulty)) for s in WIDE_SEEDS}
        assert len(shapes) >= 3, f"{difficulty}: only {len(shapes)} distinct prompt shapes"


def test_hard_and_exam_each_add_a_prompt_shape_absent_from_easy_and_medium():
    from mathgen.text_quality import prompt_shape

    lower = set()
    for difficulty in ("easy", "medium"):
        lower |= {prompt_shape(make(TOPIC, seed=s, difficulty=difficulty)) for s in WIDE_SEEDS}
    hard = {prompt_shape(make(TOPIC, seed=s, difficulty="hard")) for s in WIDE_SEEDS}
    exam = {prompt_shape(make(TOPIC, seed=s, difficulty="exam")) for s in WIDE_SEEDS}
    assert hard - lower, "hard tier adds no new prompt shape over easy+medium"
    assert exam - (lower | hard), "exam tier adds no new prompt shape over easy+medium+hard"


# ---------------------------------------------------------------------------
# skill loading
# ---------------------------------------------------------------------------


# GST rate and service-charge/surcharge rate realism
# ---------------------------------------------------------------------------

SWEEP_SEEDS = range(200)
SERVICE_CHARGE_CAP = 15  # the stated cap this fix narrows service/surcharge rates to


def test_every_gst_prompt_states_exactly_10_percent():
    """Australian GST is always 10% -- every rendered prompt containing the
    literal word "GST" must state "10%" immediately in that clause, over
    every tier and >= 200 seeds."""
    found_gst = 0
    for difficulty in TIERS:
        for seed in SWEEP_SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if "GST" in q.text:
                found_gst += 1
                assert "10% GST" in q.text, f"{difficulty} seed {seed}: {q.text!r}"
    assert found_gst > 0, "no seed/tier combination ever rendered a GST clause"


def test_no_service_charge_or_surcharge_rate_exceeds_the_cap():
    """A service charge or surcharge rate must never exceed
    `SERVICE_CHARGE_CAP` (15%) -- no more implausible "40% service charge on
    a hardware order" or "50% service charge on a café bill"."""
    import re

    pct_re = re.compile(r"(\d+)% (?:service charge|import surcharge)")
    found = 0
    for difficulty in TIERS:
        for seed in SWEEP_SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            m = pct_re.search(q.text)
            if m:
                found += 1
                pct = int(m.group(1))
                assert pct <= SERVICE_CHARGE_CAP, f"{difficulty} seed {seed}: {q.text!r}"
    assert found > 0, "no seed/tier combination ever rendered a service-charge/surcharge clause"


def test_skill_loads_with_expected_properties():
    _, skills = load_skills(ROOT / "skills")
    skill = next(
        (s for s in skills if s.id == "senior-solve-practical-problems-requiring-basic-number-operations"), None
    )
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["order-of-operations"]
    assert len(skill.sources) == 4
    assert any(src.difficulty == "exam" and src.stages == ["exam"] for src in skill.sources)
