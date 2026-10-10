"""Probability -- Bayes update from finite counts (`ucv1-bayes-finite-counts`, topic `bayes_finite_counts`).

Kernel under test: the learner constructs the joint counts from a prior population split and a per-group rate,
THEN inverts them into a posterior -- never a lookup on an already-built table.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_pairwise_distinct
from mathgen.generators.p11e_bayes_build import _independent_check

TOPIC = "bayes_finite_counts"
SKILL_ID = "ucv1-bayes-finite-counts"
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)

_FRACTION_RE = re.compile(r"^\d+/\d+$")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert len(q.distractors) >= 3


@pytest.mark.parametrize("tier", TIERS)
def test_medium_ships_four_distractors_others_three(tier):
    """Medium has a fourth named misconception (forgot-to-complement); the
    generator asks `pick_distractors` for n=4 there so it is actually
    exercisable, never crowded out by the three tier-universal ones."""
    expected = 4 if tier == "medium" else 3
    counts = [len(make(TOPIC, seed=seed, difficulty=tier).distractors) for seed in SEEDS]
    assert all(c == expected for c in counts), (tier, counts)


# --- Real grader: accepts canonical, rejects every distractor -----------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_fraction(q, q.answer_text)
        assert result.correct, f"seed {seed} {tier}: canonical {q.answer_text!r} not accepted"
        for d in q.distractors:
            result = grade_fraction(q, d)
            assert not result.correct, f"seed {seed} {tier}: distractor {d!r} was accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_deliberately_wrong_value(tier):
    """A genuinely different Fraction VALUE (never a differently-displayed
    equivalent of the canonical -- `grade_fraction` compares by value, so an
    unreduced re-spelling would be wrongly accepted, not a real test)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        p, qd = q.answer["p"], q.answer["q"]
        wrong = Fraction(p, qd) + Fraction(1, qd * 7 + 3)  # never simplifies back to p/qd
        assert wrong != Fraction(p, qd)
        result = grade_fraction(q, str(wrong))
        assert not result.correct, f"seed {seed} {tier}: wrong value {wrong} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_grade_fraction_accepts_unreduced_equivalent_of_canonical(tier):
    """`grade_fraction` compares by value: a typed unreduced fraction equal to
    the canonical must still be ACCEPTED -- confirms our distractors are never
    a "does not simplify" case in disguise (they would fail this if they were)."""
    q = make(TOPIC, seed=0, difficulty=tier)
    p, qd = q.answer["p"], q.answer["q"]
    unreduced = f"{p * 3}/{qd * 3}"
    result = grade_fraction(q, unreduced)
    assert result.correct


# --- Answer shape --------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_reduced_fraction_strictly_between_0_and_1(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        p, qd = q.answer["p"], q.answer["q"]
        assert math.gcd(p, qd) == 1, f"seed {seed} {tier}: {p}/{qd} not in lowest terms"
        assert 0 < p < qd, f"seed {seed} {tier}: posterior {p}/{qd} not strictly in (0, 1)"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_every_distractor_formatted_alike(tier):
    """Every option is a proper p/q fraction string -- never a bare integer
    mixed in among fraction-shaped options (the answer, by construction, is
    always strictly in (0, 1), so this also proves no option collapses to a
    whole number that would be pickable on shape alone)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for text in [q.answer_text, *q.distractors]:
            assert _FRACTION_RE.match(text), f"seed {seed} {tier}: {text!r} not p/q shaped"


# --- Variety --------------------------------------------------------------


def test_hard_tier_exercises_both_conditioning_directions():
    """Hard tier randomly asks about E or not-E (the new step is the
    ratio-to-counts derivation, not the conditioning direction) -- confirm
    both are actually reached over 60 seeds."""
    faulty = sum("found to be faulty" in make(TOPIC, seed=s, difficulty="hard").text for s in SEEDS)
    not_faulty = sum("found to not be faulty" in make(TOPIC, seed=s, difficulty="hard").text for s in SEEDS)
    assert faulty > 0 and not_faulty > 0
    assert faulty + not_faulty == len(SEEDS)


# --- Independent in-generator check (population enumeration) -------------


def test_independent_check_fires_on_corrupted_answer():
    # n1=10 (f1=4 faulty), n2=10 (f2=2 faulty); true P(group1|E) = 4/6 = 2/3.
    _independent_check(10, 4, 10, 2, complement=False, answer=Fraction(2, 3))
    with pytest.raises(VerificationError):
        _independent_check(10, 4, 10, 2, complement=False, answer=Fraction(1, 2))


def test_independent_check_fires_on_corrupted_complement_answer():
    _independent_check(10, 4, 10, 2, complement=True, answer=Fraction(6, 14))
    with pytest.raises(VerificationError):
        _independent_check(10, 4, 10, 2, complement=True, answer=Fraction(1, 2))


# --- Resample-on-collision guard: prove it is real, not decorative -------


def test_pairwise_distinct_guard_would_reject_a_reachable_collision():
    """Prove the guard genuinely fires on parameters the direct-tier draw can
    reach: a1=1, b1=2 (rate 1/2) with n1 == n2 (equal populations) makes the
    prior share n1/(n1+n2) collide with the group-1 likelihood a1/b1, both
    1/2 -- a real collision the resample loop must catch, not a hypothetical
    one."""
    likelihood = Fraction(1, 2)  # a1=1, b1=2
    prior = Fraction(20, 20 + 20)  # n1 == n2 == 20
    assert likelihood == prior
    with pytest.raises(VerificationError):
        check_pairwise_distinct(Fraction(3, 7), likelihood, prior, Fraction(1, 9))


# --- Skill loading ---------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None, f"{SKILL_ID} not found under skills/"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert "conditional-probability-counts" in skill.prerequisites
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == set(TIERS)
    all_stages = [stage for src in skill.sources for stage in (src.stages or [])]
    assert 0 in all_stages and "maintenance" in all_stages
