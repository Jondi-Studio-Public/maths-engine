"""Money representation (f10-money-dollars-cents-representation, AC9M3M06) and operation-choice modelling
(f10-additive-multiplicative-money-model, AC9M2N06)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, check_equivalent, stated_precision_tolerance
from mathgen.generators.p12a_money_build import _reparse_from_text

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
WIDE_SEEDS = range(60)

REP_TOPIC = "money_dollars_cents_representation"
MODEL_TOPIC = "additive_multiplicative_money_model"
TIERS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generator_is_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert q.to_json() == repeat.to_json()


# ---------------------------------------------------------------------------
# distractor hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) == 3, q.distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_representation_distractors_clear_the_stated_precision_gap(difficulty):
    """Every distractor for a 2-d.p.-displayed canonical ("X in dollars")
    differs from the answer by more than `stated_precision_tolerance(2)`
    (0.005) -- the grader-tolerance discipline required
    for any 2-d.p. `answer: number` skill."""
    tol = stated_precision_tolerance(2)
    for seed in WIDE_SEEDS:
        q = make(REP_TOPIC, seed=seed, difficulty=difficulty)
        if "." not in q.answer_text:
            continue  # dollars -> cents direction: bare integer, exact grading, not this check
        answer_value = Fraction(q.answer_text)
        for d in q.distractors:
            gap = abs(Fraction(d) - answer_value)
            assert gap > tol, f"{q.text}: distractor {d!r} only {gap} from answer {q.answer_text!r}"


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_formatted_alike(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        options = [q.answer_text] + q.distractors
        has_dot = ["." in opt for opt in options]
        assert all(has_dot) or not any(has_dot), f"{q.text}: mixed decimal/integer options {options}"


# ---------------------------------------------------------------------------
# real grader accept/reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct, q.text
        for d in q.distractors:
            assert not grade_number(q, d).correct, f"{q.text}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_a_dollar_sign_prefixed_typed_answer(difficulty):
    """`grade_number`'s tokenizer drops a leading `$` (neither digit nor
    letter), so "$4.35" is accepted for a canonical "4.35" -- confirmed
    directly against the real grader, not assumed."""
    for seed in WIDE_SEEDS:
        q = make(REP_TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(q, f"${q.answer_text}").correct, q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_the_answer_typed_for_the_wrong_unit(difficulty):
    """A cents-target answer typed as though it were the dollars value (or
    vice versa) is a genuinely different number and must be rejected --
    confirms the "wrong unit" failure mode."""
    for seed in WIDE_SEEDS:
        q = make(REP_TOPIC, seed=seed, difficulty=difficulty)
        if "." in q.answer_text:
            # cents -> dollars: the bare cents digits (no decimal point) is
            # a different number from the dollar value, unless the dollar
            # value happens to be a single whole dollar with no cents.
            bare = q.answer_text.replace(".", "").lstrip("0") or "0"
            if bare == q.answer_text.split(".")[0]:
                continue
            assert not grade_number(q, q.answer_text.replace(".", "")).correct, q.text


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(q.answer, int):
            wrong = str(q.answer + 1000)
        else:
            wrong = str(Fraction(q.answer_text) + 1000)
        assert not grade_number(q, wrong).correct, q.text


# ---------------------------------------------------------------------------
# independent (in-generator) check fires on corruption
# ---------------------------------------------------------------------------


def test_reparse_independent_check_fires_on_a_corrupted_answer():
    """`_reparse_from_text` re-derives the value from the RENDERED text
    alone; corrupting `answer` without touching `text` must make
    `check_equivalent` raise."""
    q = make(MODEL_TOPIC, seed=0, difficulty="easy")
    true_value = _reparse_from_text(q.text)
    corrupted = true_value + 1
    with pytest.raises(VerificationError):
        check_equivalent(corrupted, true_value, "should have failed")


def test_reparse_raises_on_unrecognised_text():
    with pytest.raises(VerificationError):
        _reparse_from_text("This sentence matches no known template.")


# ---------------------------------------------------------------------------
# answer satisfies its own declared kind
# ---------------------------------------------------------------------------


def test_representation_d2c_answer_is_a_bare_integer_cents_value():
    for difficulty in TIERS:
        for seed in WIDE_SEEDS:
            q = make(REP_TOPIC, seed=seed, difficulty=difficulty)
            if "." in q.answer_text:
                continue  # this seed drew the c2d direction
            assert isinstance(q.answer, int)
            int(q.answer_text)  # parses as a plain integer, no decimal point


def test_representation_c2d_answer_states_exactly_two_decimal_places():
    for difficulty in TIERS:
        for seed in WIDE_SEEDS:
            q = make(REP_TOPIC, seed=seed, difficulty=difficulty)
            if "." not in q.answer_text:
                continue  # this seed drew the d2c direction
            whole, frac = q.answer_text.split(".")
            assert len(frac) == 2, q.answer_text


def test_model_answer_is_always_a_plain_integer():
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(MODEL_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert "." not in q.answer_text


# ---------------------------------------------------------------------------
# tier-specific shape and trap-forcing
# ---------------------------------------------------------------------------


def test_representation_easy_is_always_whole_dollar_both_directions():
    for seed in WIDE_SEEDS:
        q = make(REP_TOPIC, seed=seed, difficulty="easy")
        if "." in q.answer_text:
            assert q.answer_text.endswith(".00"), q.answer_text
        else:
            assert int(q.answer_text) % 100 == 0, q.answer_text


def test_representation_medium_never_forces_a_trap():
    for seed in WIDE_SEEDS:
        q = make(REP_TOPIC, seed=seed, difficulty="medium")
        if "." in q.answer_text:
            # c2d: leading-zero trap would show as < $1.00
            assert Fraction(q.answer_text) >= 1, f"medium c2d seed {seed} forced the leading-zero trap: {q.text}"
        else:
            # d2c: regrouping trap would show as a single-digit fractional cents part
            cents = int(q.answer_text)
            assert cents % 100 >= 10, f"medium d2c seed {seed} forced the regrouping trap: {q.text}"


def test_representation_hard_always_forces_one_of_the_two_named_traps():
    fired_leading_zero = 0
    fired_regrouping = 0
    for seed in WIDE_SEEDS:
        q = make(REP_TOPIC, seed=seed, difficulty="hard")
        if "." in q.answer_text:
            assert Fraction(q.answer_text) < 1, f"hard c2d seed {seed} did not force the leading-zero trap: {q.text}"
            fired_leading_zero += 1
        else:
            cents = int(q.answer_text)
            assert cents % 100 < 10, f"hard d2c seed {seed} did not force the regrouping trap: {q.text}"
            fired_regrouping += 1
    assert fired_leading_zero > 0 and fired_regrouping > 0, (
        f"both trap directions must actually be exercised over a wide sweep: "
        f"leading_zero={fired_leading_zero}, regrouping={fired_regrouping}"
    )


# One distinguishing substring per (tier, operation), unique to that
# builder's own template -- used to confirm all four operations are live at
# every tier over a wide seed sweep, since "tier == operation" would
# defeat the skill's purpose.
_OPERATION_MARKERS = {
    "easy": {"add": "earns another", "subtract": "have left", "multiply": "of the same", "divide": "shares $"},
    "medium": {"add": "garage sale", "subtract": "chores", "multiply": "also buys", "divide": "prize money"},
    "hard": {
        "add": "washing cars",
        "subtract": "saving twice as fast",
        "multiply": "raffle tickets",
        "divide": "envelopes",
    },
}


@pytest.mark.parametrize("difficulty", TIERS)
def test_model_all_four_operations_are_live_at_every_tier(difficulty):
    markers = _OPERATION_MARKERS[difficulty]
    seen = {op: 0 for op in markers}
    for seed in range(400):
        q = make(MODEL_TOPIC, seed=seed, difficulty=difficulty)
        matched = [op for op, marker in markers.items() if marker in q.text]
        assert len(matched) == 1, f"{q.text}: matched {matched}, expected exactly one operation marker"
        seen[matched[0]] += 1
    for op, count in seen.items():
        assert count > 0, f"{difficulty}: operation {op!r} never drawn over 400 seeds"


def test_model_hard_tier_states_the_trap_in_working():
    for seed in WIDE_SEEDS:
        q = make(MODEL_TOPIC, seed=seed, difficulty="hard")
        assert any("Trap:" in w for w in q.working), q.text


# ---------------------------------------------------------------------------
# variety: >= 20 distinct rendered questions per tier over 60 seeds,
# >= 3 distinct prompt shapes per tier, hard adds a shape absent from every
# lower tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic):
    for difficulty in TIERS:
        seen = {make(topic, seed=s, difficulty=difficulty).text for s in WIDE_SEEDS}
        assert len(seen) >= 20, f"{topic} {difficulty}: only {len(seen)} distinct questions over 60 seeds"


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
def test_at_least_three_prompt_shapes_per_tier(topic):
    from mathgen.text_quality import prompt_shape

    for difficulty in TIERS:
        shapes = {prompt_shape(make(topic, seed=s, difficulty=difficulty)) for s in WIDE_SEEDS}
        assert len(shapes) >= 3, f"{topic} {difficulty}: only {len(shapes)} distinct prompt shapes"


@pytest.mark.parametrize("topic", [REP_TOPIC, MODEL_TOPIC])
def test_hard_adds_a_prompt_shape_absent_from_every_lower_tier(topic):
    from mathgen.text_quality import prompt_shape

    lower = set()
    for difficulty in ("easy", "medium"):
        lower |= {prompt_shape(make(topic, seed=s, difficulty=difficulty)) for s in WIDE_SEEDS}
    hard = {prompt_shape(make(topic, seed=s, difficulty="hard")) for s in WIDE_SEEDS}
    assert hard - lower, f"{topic}: hard tier adds no new prompt shape over easy+medium"


# ---------------------------------------------------------------------------
# skill loading
# ---------------------------------------------------------------------------


def test_representation_skill_loads_with_expected_properties():
    _, skills = load_skills(ROOT / "skills")
    skill = next((s for s in skills if s.id == "f10-money-dollars-cents-representation"), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["australian-coin-values", "money-change"]
    assert len(skill.sources) == 3


def test_model_skill_loads_with_expected_properties():
    _, skills = load_skills(ROOT / "skills")
    skill = next((s for s in skills if s.id == "f10-additive-multiplicative-money-model"), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["money-change", "f10-multiply-divide-one-digit-arrays"]
    assert len(skill.sources) == 3
