"""Significant-figures, engineering-notation.

Expected, not an error.
"""

import re
from decimal import Decimal
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, grade_number, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8a_notation_build import (
    _digit_string_round,
    _is_boundary,
    _round_significant,
    _verify_engineering_pair,
    _verify_sig_fig_rounding,
)

SIG_FIGS_TOPIC = "significant_figures"
ENG_NOTATION_TOPIC = "engineering_notation"
SIG_FIGS_VARIANTS = ["plain", "small_value", "large_trailing_zero"]
ENG_NOTATION_VARIANTS = ["large_positive", "small_value", "wide_exponent"]
SEEDS = range(60)


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_deterministic(variant):
    for seed in SEEDS:
        q1 = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        q2 = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_engineering_notation_deterministic(variant):
    for seed in SEEDS:
        q1 = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        q2 = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        assert q1.to_json() == q2.to_json()


# --- distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_distractors_distinct_nonempty_never_equal_answer(variant):
    at_least_two = 0
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1
        if len(q.distractors) >= 2:
            at_least_two += 1
    # measured, not assumed: plain is the tightest tier.
    assert at_least_two / len(SEEDS) >= 0.85, f"{variant}: only {at_least_two}/{len(SEEDS)} had >=2"


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_engineering_notation_distractors_distinct_nonempty_never_equal_answer(variant):
    for seed in SEEDS:
        q = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 2


def test_default_variant_clears_the_shared_85_percent_bar():
    # The bare default -- make(topic, seed=s) with NO variant -- must still offer >= 2 distractors
    # on at least 85% of seeds.
    for topic, n in [(SIG_FIGS_TOPIC, 100), (ENG_NOTATION_TOPIC, 100)]:
        counts = [len(make(topic, seed=s).distractors) for s in range(n)]
        assert sum(1 for c in counts if c >= 2) / n >= 0.85


# --- real grader accept/reject ------------------------------------------------


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_real_grader_accepts_canonical_and_rejects_every_distractor(variant):
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_real_grader_rejects_a_deliberately_wrong_answer(variant):
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        wrong = Fraction(q.answer_text) + 1
        assert not grade_number(q, str(wrong)).correct


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_engineering_notation_real_grader_accepts_canonical_and_rejects_every_distractor(variant):
    for seed in SEEDS:
        q = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


# --- independent in-generator check fires on corrupted input -----------------


def test_sig_figs_independent_check_fires_on_corrupted():
    value = Decimal("0.004813")
    correct, _ = _round_significant(value, 2)
    assert correct == Decimal("0.0048")
    _verify_sig_fig_rounding(value, 2, correct)  # must not raise
    with pytest.raises(VerificationError):
        _verify_sig_fig_rounding(value, 2, Decimal("0.0049"))


def test_sig_figs_digit_string_route_matches_decimal_route_directly():
    cases = [
        (Decimal("0.004813"), 2, Decimal("0.0048")),
        (Decimal("48130"), 2, Decimal("48000")),
        (Decimal("4.7985"), 3, Decimal("4.80")),
        (Decimal("9.95"), 2, Decimal("10.0")),
    ]
    for value, sf, expected in cases:
        assert _digit_string_round(value, sf) == expected


def test_engineering_notation_independent_check_fires_on_corrupted():
    value = Decimal("47500")  # 4.75e4 -> engineering 47.5e3
    _verify_engineering_pair(value, 3, Decimal("47.5"))  # must not raise
    with pytest.raises(VerificationError):
        _verify_engineering_pair(value, 3, Decimal("4.75"))  # wrong mantissa
    with pytest.raises(VerificationError):
        _verify_engineering_pair(value, 6, Decimal("0.0475"))  # wrong exponent


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        make(SIG_FIGS_TOPIC, seed=1, variant="unknown")
    with pytest.raises(ValueError):
        make(ENG_NOTATION_TOPIC, seed=1, variant="unknown")


# --- structural traps, forced every seed --------------------------------------


_VALUE_RE = re.compile(r"Round ([\d.]+) to")


def test_plain_tier_never_has_a_leading_or_trailing_zero_complication():
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant="plain")
        value = Decimal(_VALUE_RE.match(q.text).group(1))
        assert value.adjusted() >= 0, f"seed {seed}: unexpected leading zero, {q.text!r}"


def test_small_value_tier_always_forces_a_leading_zero():
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant="small_value")
        value = Decimal(_VALUE_RE.match(q.text).group(1))
        assert value.adjusted() <= -2, f"seed {seed}: no leading zero forced, {q.text!r}"


def test_large_trailing_zero_tier_always_forces_magnitude_preservation():
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant="large_trailing_zero")
        m = re.match(r"Round (\d+) to (\d+)", q.text)
        digits, sf = m.group(1), int(m.group(2))
        assert sf < len(digits), f"seed {seed}: sf not fewer than integer digits, {q.text!r}"
        assert "." not in q.text.split(" to ")[0]


_SCI_RE = re.compile(r"Write ([\d.]+) . 10\^(-?\d+)")


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_scientific_exponent_never_already_a_multiple_of_three(variant):
    for seed in SEEDS:
        q = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        sci_exp = int(_SCI_RE.match(q.text).group(2))
        assert sci_exp % 3 != 0, f"seed {seed}: {q.text!r}"


# --- variety -------------------------------------------------------------


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_at_least_20_distinct_questions_per_tier(variant):
    texts = {make(SIG_FIGS_TOPIC, seed=s, variant=variant).text for s in range(60)}
    assert len(texts) >= 20, f"{variant}: only {len(texts)} distinct questions over 60 seeds"


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_engineering_notation_at_least_20_distinct_questions_per_tier(variant):
    texts = {make(ENG_NOTATION_TOPIC, seed=s, variant=variant).text for s in range(60)}
    assert len(texts) >= 20, f"{variant}: only {len(texts)} distinct questions over 60 seeds"


# --- answer satisfies its own declared kind / not on a rounding boundary -----


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_answer_not_on_rounding_boundary(variant):
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        value = Decimal(_VALUE_RE.match(q.text).group(1))
        sf = int(re.search(r"to (\d+) significant", q.text).group(1))
        assert not _is_boundary(value, sf)


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_engineering_notation_mantissa_in_range(variant):
    for seed in SEEDS:
        q = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        m = re.match(r"([\d.]+) . 10\^(-?\d+)", q.answer_text)
        mantissa = Decimal(m.group(1))
        assert Decimal(1) <= mantissa < Decimal(1000)
        exp = int(m.group(2))
        assert exp % 3 == 0


# --- answer and distractors formatted alike -----------------------------------


_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@pytest.mark.parametrize("variant", SIG_FIGS_VARIANTS)
def test_sig_figs_answer_and_distractors_formatted_alike(variant):
    for seed in SEEDS:
        q = make(SIG_FIGS_TOPIC, seed=seed, variant=variant)
        assert _NUMERIC_RE.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _NUMERIC_RE.match(d), d


_ENG_OPTION_RE = re.compile(r"^[\d.]+ . 10\^-?\d+( \([\d.]+ \w+\))?$")


@pytest.mark.parametrize("variant", ENG_NOTATION_VARIANTS)
def test_engineering_notation_answer_and_distractors_formatted_alike(variant):
    for seed in SEEDS:
        q = make(ENG_NOTATION_TOPIC, seed=seed, variant=variant)
        assert _ENG_OPTION_RE.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _ENG_OPTION_RE.match(d), d


# --- skills load correctly ----------------------------------------------------


def test_skills_load_with_correct_properties():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}

    sig = by_id.get("significant-figures")
    assert sig is not None
    assert sig.prerequisites == ["round-decimals"]
    assert sig.answer == "number"
    assert sig.placement_eligible is False

    eng = by_id.get("engineering-notation")
    assert eng is not None
    assert eng.prerequisites == ["scientific-notation"]
    assert eng.answer == "mcq"
    assert eng.placement_eligible is False
