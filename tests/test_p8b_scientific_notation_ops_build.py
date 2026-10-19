"""Tests for the new `scientific_notation_operate` topic, which extends the LIVE `scientific-notation` skill
(`skills/secondary-number/scientific-notation.yaml`, `answer: mcq`) with the four arithmetic operations on two
scientific-notation numbers.

This skill has no `difficulty`/`stages` tiering, so there is exactly one draw space to sweep, not one per
tier.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct
from mathgen.generators.p8b_scientific_notation_ops_build import (
    _display_in_range,
    _fmt,
    _pairwise_ok,
    _renormalise,
    _value,
    _verify,
)

TOPIC = "scientific_notation_operate"
SEEDS = range(60)
OPS = ("add", "subtract", "multiply", "divide")
_OPTION_RE = re.compile(r"^-?\d+(?:\.\d+)? × 10\^-?\d+$")


@functools.lru_cache(maxsize=1)
def _skills():
    # load_skills() rescans every YAML file (~563ms); cache it at module
    # scope so no per-seed loop pays that cost.
    return load_skills()


def _skill(skill_id):
    _domains, skills = _skills()
    for s in skills:
        if s.id == skill_id:
            return s
    raise AssertionError(f"skill {skill_id!r} not found")


# --- determinism -------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
def test_deterministic_across_seeds(op):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, ops=(op,))
        q2 = make(TOPIC, seed=seed, ops=(op,))
        assert q1.to_json() == q2.to_json()


def test_default_ops_covers_all_four_over_240_seeds():
    """No `ops=` kwarg -> all four operations occur (rng.choice among all
    four, same shape as `foundations_algebra`'s `fractions` `ops=` kwarg)."""
    seen_ops = set()
    for seed in range(240):
        q = make(TOPIC, seed=seed)
        seen_ops.add(_op_symbol(q.text))
    assert seen_ops == {"+", "-", "×", "÷"}


# --- distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("op", OPS)
def test_distractors_distinct_nonempty_and_never_equal_answer(op):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        assert len(q.distractors) == 3
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))


@pytest.mark.parametrize("op", OPS)
def test_answer_and_every_distractor_formatted_alike(op):
    """Every option matches the one option-format string
    `f"{coeff:g} × 10^{exp}"` -- no option is identifiable by shape."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        for opt in [q.answer_text, *q.distractors]:
            assert _OPTION_RE.match(opt), opt


@pytest.mark.parametrize("op", OPS)
def test_answer_coefficient_is_in_one_to_ten(op):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        coeff, _exp = _parse_option(q.answer_text)
        assert 1 <= coeff < 10, q.answer_text


# --- pairwise value-distinctness (explicit disclosure requirement) --


@pytest.mark.parametrize("op", OPS)
def test_un_renormalised_distractor_is_value_equal_to_the_answer(op):
    """The un-renormalised form is *supposed* to be the same real number as
    the answer (renormalisation is value-preserving) -- see the generator
    module's docstring, deviation 1. This is the one pair the value-distinct
    check below deliberately excludes."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        answer_value = _value(*_parse_option(q.answer_text))
        distractor_values = [_value(*_parse_option(d)) for d in q.distractors]
        assert answer_value in distractor_values, (q.answer_text, q.distractors)


@pytest.mark.parametrize("op", OPS)
def test_every_other_option_pair_is_value_distinct(op):
    """All 4 options pairwise distinct by VALUE except (answer, un-renormalised
    distractor) -- computed from each option's own (coefficient, exponent),
    never by string comparison."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        options = [q.answer_text, *q.distractors]
        values = [_value(*_parse_option(o)) for o in options]
        equal_pairs = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if values[i] == values[j]:
                    equal_pairs += 1
        assert equal_pairs == 1, (q.text, options)  # only (answer, D1)


@pytest.mark.parametrize("op", OPS)
def test_renormalisation_was_genuinely_needed_every_seed(op):
    """Every emitted question is one where renormalisation changed the
    (coefficient, exponent) pair -- see generator module docstring,
    deviation 2: a no-op renormalisation would make the un-renormalised
    distractor pair-identical to the answer, an unfair duplicate option."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        answer_pair = _parse_option(q.answer_text)
        distractor_pairs = [_parse_option(d) for d in q.distractors]
        assert answer_pair not in distractor_pairs, (q.answer_text, q.distractors)


# --- real grader: accept canonical, reject distractors and a wrong answer --


@pytest.mark.parametrize("op", OPS)
def test_real_mcq_grader_accepts_answer_rejects_distractors(op):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, ops=(op,))
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, i).correct


@pytest.mark.parametrize("op", OPS)
def test_deliberately_wrong_answer_rejected(op):
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, ops=(op,))
        options = mcq_options(q)
        bogus_idx = len(options) + 5
        assert not grade_mcq(q, bogus_idx).correct
        assert not grade_mcq(q, "not-a-number").correct


# --- independent in-generator checks fire on a corrupted answer -----------


def test_check_fires_when_un_renormalised_value_does_not_match_the_answer():
    answer_pair = (Fraction(2), 5)
    d1 = (Fraction(19, 1), 3)  # 19 x 10^3 = 19000, NOT 2 x 10^5 = 200000
    d2 = (Fraction(3), 8)
    d3 = (Fraction(4), 9)
    with pytest.raises(VerificationError):
        _verify(answer_pair, d1, d2, d3, true_value_float=200000.0)


def test_check_distinct_fires_when_a_distractor_equals_the_answer():
    answer_value = _value(Fraction(2), 5)
    other_value = _value(Fraction(2), 5)  # deliberately the same value
    with pytest.raises(VerificationError):
        check_distinct(answer_value, other_value)


def test_pairwise_ok_rejects_a_no_op_renormalisation():
    # raw already in [1, 10): d1's pair would be identical to the answer's.
    answer_pair = (Fraction(2), 5)
    d1 = (Fraction(2), 5)
    d2 = (Fraction(3), 8)
    d3 = (Fraction(4), 9)
    assert _pairwise_ok(answer_pair, d1, d2, d3) is False


def test_pairwise_ok_rejects_a_value_collision_among_the_other_options():
    answer_pair = (Fraction(2), 5)
    d1 = (Fraction(20), 4)  # value-equal to the answer, by design (20e4 = 2e5)
    d2 = (Fraction(3), 6)
    d3 = (Fraction(30), 5)  # value-equal to d2 (3e6 == 30e5) -- must be rejected
    assert _pairwise_ok(answer_pair, d1, d2, d3) is False


def test_pairwise_ok_accepts_a_healthy_draw():
    answer_pair = (Fraction(2), 5)
    d1 = (Fraction(20), 4)
    d2 = (Fraction(2), 8)
    d3 = (Fraction(2), 2)
    assert _pairwise_ok(answer_pair, d1, d2, d3) is True


def test_renormalise_shifts_out_of_range_coefficients():
    assert _renormalise(Fraction("19.8"), 3) == (Fraction("1.98"), 4)
    assert _renormalise(Fraction("0.55"), -5) == (Fraction("5.5"), -6)
    assert _renormalise(Fraction("4.5"), 7) == (Fraction("4.5"), 7)  # already in range


def test_display_in_range_catches_the_rounds_up_to_ten_edge_case():
    # 9.9999996 rounds to "10" under :g -- an invalid scientific-notation
    # coefficient even though the exact Fraction value is in [1, 10).
    almost_ten = Fraction(99999996, 10000000)
    assert almost_ten < 10
    assert _display_in_range(almost_ten) is False
    assert _display_in_range(Fraction("9.9")) is True
    assert _display_in_range(Fraction(10)) is False
    assert _display_in_range(Fraction(1)) is True


# --- at least 20 distinct rendered questions per op over 60 seeds ---------


@pytest.mark.parametrize("op", OPS)
def test_at_least_20_distinct_questions_over_60_seeds(op):
    texts = {make(TOPIC, seed=s, ops=(op,)).text for s in SEEDS}
    assert len(texts) >= 20


# --- skill loads correctly -------------------------------------------------


def test_skill_loads_with_new_source_and_unchanged_declared_fields():
    skill = _skill("scientific-notation")
    assert skill.answer == "mcq"
    # `placement_eligible` is a declared field this test must not
    # touch -- the existing LIVE skill YAML never sets
    # it, so it loads at whatever `mathgen.skills`'s own default is (True);
    # asserting the true current value, not assuming a new-skill
    # convention of False, which only applies to skills built from scratch.
    assert skill.placement_eligible is True
    assert skill.prerequisites == ["decimal-place-value", "evaluate-powers"]
    assert skill.domain == "secondary-number"
    topics = {src.topic for src in skill.sources}
    assert "scientific_notation_operate" in topics
    assert "scientific_notation" in topics  # existing source untouched
    new_sources = [src for src in skill.sources if src.topic == "scientific_notation_operate"]
    assert len(new_sources) == 1
    # Every active source now carries a tier label (skills/skill.schema.yaml); operating on two
    # values and renormalising is one step up from the `easy` conversion source. No stages.
    assert new_sources[0].difficulty == "medium"
    assert new_sources[0].stages is None


# --- helpers used by this file only ----------------------------------------


_OPT_RE = re.compile(r"^(?P<coeff>-?\d+(?:\.\d+)?) × 10\^(?P<exp>-?\d+)$")


def _parse_option(option_text):
    m = _OPT_RE.match(option_text)
    assert m, option_text
    return Fraction(m["coeff"]), int(m["exp"])


_TEXT_OP_RE = re.compile(r"\) (?P<op>[+\-×÷]) \(")


def _op_symbol(text):
    m = _TEXT_OP_RE.search(text)
    assert m, text
    return m["op"]
