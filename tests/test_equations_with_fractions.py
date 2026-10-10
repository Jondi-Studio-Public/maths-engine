"""Tests the `linear_equation_fraction` topic and the `linear-equations-with-fractions` skill: determinism, exact
integer answers, distractors and grading.

Linear-equations-with-fractions.

Covers mathgen/generators/equations_with_fractions.py's ``linear_equation_fraction``
topic (four ``kind``s: single, two_terms, binomial, both_sides) and the skill
``linear-equations-with-fractions`` that wires them up."""

from pathlib import Path

import pytest
from sympy import Rational

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent
from mathgen.text_quality import find_text_defects

ROOT = Path(__file__).parent.parent
SEEDS = range(60)
TOPIC = "linear_equation_fraction"

# Mirrors the params each skill source passes in skills/equations/linear-equations-with-fractions.yaml.
KIND_PARAMS = {
    "single": dict(kind="single", sol_hi=8),
    "two_terms": dict(kind="two_terms", k_hi=2),
    "binomial": dict(kind="binomial", h_hi=9, x0_hi=9),
    "both_sides": dict(kind="both_sides", h_hi=6, sol_hi=9),
}
KINDS = list(KIND_PARAMS)


@pytest.mark.parametrize("kind", KINDS)
def test_determinism_across_seeds(kind):
    for seed in range(20):
        q1 = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        q2 = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors


@pytest.mark.parametrize("kind", KINDS)
def test_answer_is_exact_integer(kind):
    """Prefer draws whose solutions are exact -- every kind is constructed backwards from an integer x0, so
    answer is a plain int."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        assert isinstance(q.answer, int)
        assert q.answer_text == f"x = {q.answer}"


@pytest.mark.parametrize("kind", KINDS)
def test_distractors_distinct_nonempty_never_equal_answer(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        assert len(q.distractors) > 0, f"seed {seed}: no distractors"
        for d in q.distractors:
            assert d != q.answer_text, f"seed {seed}: distractor equals answer"
        assert len(q.distractors) == len(set(q.distractors)), f"seed {seed}: duplicate distractors"


@pytest.mark.parametrize("kind", KINDS)
def test_at_least_85pct_of_seeds_keep_two_distractors(kind):
    kept = sum(1 for seed in SEEDS if len(make(TOPIC, seed=seed, **KIND_PARAMS[kind]).distractors) >= 2)
    assert kept / len(SEEDS) >= 0.85, f"{kind}: only {kept}/{len(SEEDS)} seeds kept >=2 distractors"


@pytest.mark.parametrize("kind", KINDS)
def test_real_grader_accepts_canonical_rejects_every_distractor(kind):
    """hazard: every distractor must clear the grader by construction, checked on every candidate unconditionally
    -- not just the ones that 'look' rejected."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        result = GRADERS["number"](q, q.answer_text)
        assert result.correct, f"seed {seed}: grader rejected canonical {q.answer_text}"
        for d in q.distractors:
            result = GRADERS["number"](q, d)
            assert not result.correct, f"{kind} seed {seed}: grader ACCEPTED distractor {d!r}"


@pytest.mark.parametrize("kind", KINDS)
def test_deliberately_wrong_answer_rejected(kind):
    for seed in range(10):
        q = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        wrong_text = f"x = {q.answer + 1}"
        result = GRADERS["number"](q, wrong_text)
        assert not result.correct, f"seed {seed}: grader accepted deliberately wrong {wrong_text}"


@pytest.mark.parametrize("kind", KINDS)
def test_at_least_20_distinct_questions_per_kind(kind):
    seen = {make(TOPIC, seed=seed, **KIND_PARAMS[kind]).text for seed in SEEDS}
    assert len(seen) >= 20, f"{kind}: only {len(seen)} distinct questions over 60 seeds"


@pytest.mark.parametrize("kind", KINDS)
def test_no_bad_text_patterns(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS[kind])
        defects = find_text_defects(q, include_working=True)
        assert not defects, f"{kind} seed {seed}: {defects} in {q.text!r}"


def test_two_terms_uses_both_denominators_and_can_subtract():
    """The 'second denominator' escalation: both p and q must appear, and the
    minus-sign variant (x/p - x/q = c) must actually occur across seeds."""
    saw_plus = saw_minus = False
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS["two_terms"])
        assert q.text.count("x/") == 2, f"seed {seed}: expected two fraction terms in {q.text!r}"
        if " + " in q.text:
            saw_plus = True
        if " - " in q.text:
            saw_minus = True
    assert saw_plus and saw_minus, "expected both + and - variants across 60 seeds"


def test_binomial_numerator_is_a_binomial():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS["binomial"])
        assert q.text.startswith("Solve ("), f"seed {seed}: expected a bracketed numerator in {q.text!r}"


def test_both_sides_has_a_fraction_on_each_side():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, **KIND_PARAMS["both_sides"])
        assert q.text.count("/") == 2, f"seed {seed}: expected two fractions in {q.text!r}"
        assert q.text.count("x") == 2, f"seed {seed}: expected x on both sides in {q.text!r}"


def test_independent_check_fires_on_corrupted_answer():
    """check_equivalent/check_distinct actually fail the build on a wrong value."""
    with pytest.raises(VerificationError):
        check_equivalent(Rational(1, 2), Rational(1, 3), "must differ -- proves the check fires")
    with pytest.raises(VerificationError):
        check_distinct(Rational(1, 2), Rational(1, 2), msg="must differ -- proves the check fires")


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, kind="nonsense")


def test_skill_loads_with_correct_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id["linear-equations-with-fractions"]
    assert skill.domain == "equations"
    assert skill.status == "active"
    assert skill.answer == "number"
    assert skill.prerequisites == ["two-step-equations", "equations-with-brackets", "fraction-arithmetic"]
    sources_by_stage_group = {tuple(s.stages): s for s in skill.sources}
    assert sources_by_stage_group[(0, 1, 2)].params["kind"] == "single"
    assert sources_by_stage_group[(3,)].params["kind"] == "two_terms"
    assert sources_by_stage_group[(4,)].params["kind"] == "binomial"
    assert sources_by_stage_group[(5, 6, "maintenance")].params["kind"] == "both_sides"
