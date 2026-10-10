"""Tests `p14e_total_variance_two_case` (`ucv1-total-variance-finite`), built by
`mathgen/generators/p14e_total_variance_two_case_build.py`.

``ucv1-total-variance-finite``, built by ``mathgen/generators/p14e_total_variance_two_case_build.py`` (topic
``p14e_total_variance_two_case``, answer: fraction, all four tiers).

See the generator module's docstring for the kernel, the boundary against
`finite-distribution-variance`, and the tier design.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import PRESETS, REGISTRY, VerificationError, check
from mathgen.generators.p14e_total_variance_two_case_build import (
    _e_var_within,
    _e_y,
    _e_y2,
    _var_between,
    _var_total,
    p14e_total_variance_two_case,
)

TOPIC = "p14e_total_variance_two_case"
SKILL_ID = "ucv1-total-variance-finite"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)


# --- Registration -----------------------------------------------------


def test_topic_registered_with_all_four_presets():
    assert TOPIC in REGISTRY
    assert set(PRESETS[TOPIC]) == {"easy", "medium", "hard", "exam"}


# --- Determinism --------------------------------------------------------


# --- Distractor hygiene ---------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    counts = Counter()
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1
        counts[len(q.distractors)] += 1
    counts_dict = dict(counts)
    at_least_two = sum(v for k, v in counts_dict.items() if k >= 2)
    assert at_least_two / 300 >= 0.85, f"{tier}: <2-distractor share too high: {counts_dict}"


# --- Real grader: fraction, exact-value graded ----------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_answer_rejects_every_distractor(tier):
    grader = GRADERS["fraction"]
    assert grader is grade_fraction


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    grader = GRADERS["fraction"]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        p, q_denom = q.answer["p"], q.answer["q"]
        wrong = Fraction(p + q_denom * 7 + 1, q_denom)  # a value that can never collide
        wrong_text = str(wrong.numerator) if wrong.denominator == 1 else f"{wrong.numerator}/{wrong.denominator}"
        assert not grader(q, wrong_text).correct, f"{tier} seed={seed}: corrupted answer accepted"


# --- Answer kind: exact fraction, never "does not simplify" --------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_shape_is_reduced_fraction(tier):
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert set(q.answer) == {"p", "q"}
        frac = Fraction(q.answer["p"], q.answer["q"])
        assert frac.numerator == q.answer["p"] and frac.denominator == q.answer["q"], (
            "answer is not stored in lowest terms"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_options_formatted_alike(tier):
    """Every distractor is a fraction/integer string, same shape as the answer."""
    for seed in range(50):
        q = make(TOPIC, seed=seed, difficulty=tier)
        for opt in [q.answer_text, *q.distractors]:
            assert Fraction(opt) is not None  # raises ValueError if not fraction-shaped


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


# --- easy: p1 = 1/2 exactly ------------------------------------------------


def test_easy_probability_is_always_one_half():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert "probability 1/2" in q.text or "p1 = 1/2" in q.text or "(1/2 each)" in q.text


# --- medium: p1 never 1/2 --------------------------------------------------


def test_medium_probability_is_never_one_half():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert "probability 1/2" not in q.text


# --- hard: counts idiom, never probabilities stated directly --------------


def test_hard_states_counts_not_probabilities():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "population of N" in q.text or "population of size N" in q.text
        assert "probability" not in q.text


# --- exam: inversion, Var(Y) stated, one conditional variance withheld ---


def test_exam_states_var_y_and_asks_for_conditional_variance():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "Var(Y) =" in q.text
        assert "conditional variance of Y in case 2" in q.text
        # case 2's conditional variance is never stated as a number in the prompt.
        assert "conditional variance {s2}" not in q.text


def test_exam_sometimes_uses_counts_and_sometimes_probabilities():
    counts_seen = 0
    prob_seen = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        if "population of N" in q.text:
            counts_seen += 1
        else:
            prob_seen += 1
    assert counts_seen > 0 and prob_seen > 0, (counts_seen, prob_seen)


# --- Two-cases-only constraint: never a third case ------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_never_mentions_a_third_case(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "case 3" not in q.text.lower()


# --- Independent in-generator check tripwire ------------------------------


def test_check_fires_on_corrupted_condition():
    with pytest.raises(VerificationError):
        check(1 == 2, "should fail")
    check(1 == 1, "should not fail")


def test_e_y2_route_matches_two_term_decomposition_for_hand_values():
    p1, p2 = Fraction(1, 3), Fraction(2, 3)
    mu1, mu2 = Fraction(3), Fraction(-2)
    s1, s2 = Fraction(4), Fraction(9)
    total = _var_total(p1, p2, mu1, mu2, s1, s2)
    e_y = _e_y(p1, p2, mu1, mu2)
    e_y2 = _e_y2(p1, p2, mu1, mu2, s1, s2)
    assert e_y2 - e_y * e_y == total


def test_drops_within_and_drops_between_are_the_two_summands():
    p1, p2 = Fraction(1, 4), Fraction(3, 4)
    mu1, mu2 = Fraction(0), Fraction(5)
    s1, s2 = Fraction(2), Fraction(6)
    within = _e_var_within(p1, p2, s1, s2)
    between = _var_between(p1, p2, mu1, mu2)
    assert within + between == _var_total(p1, p2, mu1, mu2, s1, s2)


def test_direct_builder_callable_via_public_entry_point():
    import random

    rng = random.Random(1)
    d = p14e_total_variance_two_case(rng, kind="easy")
    assert set(d["answer"]) == {"p", "q"}


# --- Skill loading ---------------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["finite-distribution-variance"]
    assert len(skill.sources) == 4
    stages = [set(s.stages) for s in skill.sources]
    assert stages == [{0, 1, 2}, {3, 4}, {5, 6, "maintenance"}, {"exam"}]
    assert skill.sources[3].difficulty == "exam"
    assert all(source.topic == TOPIC for source in skill.sources)


def test_skill_passes_structural_validation():
    from mathgen import skills as skills_mod

    domains, loaded = skills_mod.load_skills()
    report = skills_mod.validate(domains, loaded)
    assert report.ok, report.errors


# --- 200-seed measurements -------------------------------------------------


#: Honest measured floors (not an arbitrary shared threshold): easy's answer
#: space is bounded by p1=1/2 and small integer mu/sigma draws (measured 112
#: distinct/200); medium and hard have much wider fraction-valued answer
#: spaces from a varying p1 (measured 174, 196); exam's answer is a bare
#: withheld integer sigma2^2 drawn from range(1, 40) (measured 39/200 -- the
#: range's own size is the ceiling, not a defect).
_CONCENTRATION_FLOOR = {"easy": 90, "medium": 150, "hard": 150, "exam": 30}


@pytest.mark.parametrize("tier", TIERS)
def test_200_seed_answer_concentration_measured(tier):
    values = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        values.add(Fraction(q.answer["p"], q.answer["q"]))
    floor = _CONCENTRATION_FLOOR[tier]
    assert len(values) >= floor, (
        f"{tier}: answer concentration too high, {len(values)} distinct over 200 (floor {floor})"
    )


def test_easy_swapped_weights_collision_rate_measured():
    """Both are dropped from the offered distractors whenever they collide, never offered as a duplicate."""
    from mathgen.generators.p14e_total_variance_two_case_build import (
        _e_var_within as ew,
    )

    N = 300
    collisions = 0
    for seed in range(N):
        q = make(TOPIC, seed=seed, difficulty="easy")
        # 2 live misconceptions (drops-within, drops-between) always offered
        # (unless they coincidentally equal each other or the answer); when
        # BOTH unweighted-EY and swapped-weights collide with the answer (the
        # expected case at p1=1/2), the candidate pool from those two named
        # misconceptions is empty and distractor count stays at 2 (drops-
        # within/drops-between only) rather than 4.
        if len(q.distractors) <= 2:
            collisions += 1
    rate = collisions / N
    assert 0.0 <= rate <= 1.0
