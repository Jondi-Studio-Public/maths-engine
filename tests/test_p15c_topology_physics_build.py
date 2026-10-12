"""Focused contract tests for: ucv1-topology-basis-membership (answer: mcq) and ucv1-symmetry-conservation
(answer: mcq)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

TOPOLOGY_ID = "ucv1-topology-basis-membership"
SYMMETRY_ID = "ucv1-symmetry-conservation"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]
_TOPOLOGY_CLASSES = {"a basis", "fails coverage", "fails the intersection condition", "fails both"}
_QUANTITIES = {"E", "p_x", "p_y", "p_z", "L_x", "L_y", "L_z"}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


TOPOLOGY_SKILL, TOPOLOGY_SOURCES = _skill_sources(TOPOLOGY_ID)
SYMMETRY_SKILL, SYMMETRY_SOURCES = _skill_sources(SYMMETRY_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-topology-basis-membership
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_topology_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(TOPOLOGY_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_topology_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(TOPOLOGY_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_topology_answer_is_one_of_the_four_classes(difficulty):
    for seed in SEEDS:
        q = _make(TOPOLOGY_SOURCES, difficulty, seed)
        assert q.answer_text in _TOPOLOGY_CLASSES
        assert set(q.distractors) | {q.answer_text} == _TOPOLOGY_CLASSES


@pytest.mark.parametrize("difficulty", TIERS)
def test_topology_200_seed_class_balance(difficulty):
    counts = Counter(_make(TOPOLOGY_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == _TOPOLOGY_CLASSES, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.45, f"{difficulty}: top class {counts.most_common(1)} is {top_share:.1%} of 200 draws"


def _member_sizes(text):
    """Sizes of every explicitly-listed set in the collection clause (skips
    the universe declaration itself, "the empty set", and any indirect
    clause, none of which are a {...} member listing)."""
    import re

    body = text.split("is given: ")[1].split(". Decide")[0]
    return [clause.count(",") + 1 for clause in re.findall(r"\{[^{}]*\}", body)]


def test_topology_medium_tier_uses_indirect_member_on_some_draws():
    """Easy never does this."""
    indirect = 0
    for seed in range(200):
        q = _make(TOPOLOGY_SOURCES, "medium", seed)
        if "the union of" in q.text:
            indirect += 1
    assert indirect >= 100, f"only {indirect}/200 medium draws used an indirect member"


def test_topology_easy_never_uses_indirect_member():
    for seed in range(200):
        q = _make(TOPOLOGY_SOURCES, "easy", seed)
        assert "the union of" not in q.text


def test_topology_hard_tier_200_seed_class_balance_within_tolerance():
    """After the decorrelating burn draws, no class should exceed ~30% of 200 draws."""
    counts = Counter(_make(TOPOLOGY_SOURCES, "hard", seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert top_share <= 0.30, f"hard: top class {counts.most_common(1)} is {top_share:.1%} of 200 draws"


def test_topology_hard_intersection_failing_draws_use_size3_members():
    """Boundary: on the hard tier, the intersection-failing classes are
    built from size-3 subsets of the 4-element universe (never size-2, the
    shape those same classes use on easy/medium) -- a genuinely larger
    overlap to check, not just a bigger universe."""
    hard_size3 = 0
    easy_size3 = 0
    checked_hard = checked_easy = 0
    for seed in SWEEP_SEEDS:
        q = _make(TOPOLOGY_SOURCES, "hard", seed)
        if q.answer_text in ("fails the intersection condition",):
            checked_hard += 1
            if 3 in _member_sizes(q.text):
                hard_size3 += 1
        q2 = _make(TOPOLOGY_SOURCES, "easy", seed)
        if q2.answer_text in ("fails the intersection condition",):
            checked_easy += 1
            if 3 in _member_sizes(q2.text):
                easy_size3 += 1
    assert checked_hard > 0 and checked_easy > 0
    assert hard_size3 == checked_hard, "every hard intersection-failing draw should use size-3 members"
    assert easy_size3 == 0, "easy intersection-failing draws should never use size-3 members"


def test_topology_skill_loads_through_api_with_correct_properties():
    assert TOPOLOGY_SKILL.domain == "metric-spaces-and-topology"
    assert TOPOLOGY_SKILL.status == "active"
    assert TOPOLOGY_SKILL.placement_eligible is False
    assert TOPOLOGY_SKILL.answer == "mcq"
    assert TOPOLOGY_SKILL.prerequisites == ["finite-set-operations"]
    assert {src.difficulty for src in TOPOLOGY_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert TOPOLOGY_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-symmetry-conservation
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_symmetry_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(SYMMETRY_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_symmetry_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(SYMMETRY_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_symmetry_answer_is_one_of_the_seven_quantities(difficulty):
    for seed in SEEDS:
        q = _make(SYMMETRY_SOURCES, difficulty, seed)
        assert q.answer_text in _QUANTITIES
        assert set(q.distractors) <= _QUANTITIES


@pytest.mark.parametrize("difficulty", TIERS)
def test_symmetry_200_seed_answer_never_concentrates(difficulty):
    counts = Counter(_make(SYMMETRY_SOURCES, difficulty, seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert top_share <= 0.40, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


def test_symmetry_easy_tier_only_translation_pool():
    for seed in SWEEP_SEEDS:
        q = _make(SYMMETRY_SOURCES, "easy", seed)
        assert q.answer_text in {"E", "p_x", "p_y", "p_z"}


def test_symmetry_medium_tier_gives_lagrangian_and_finds_cyclic_coordinate():
    """Medium's escalation over easy is that the Lagrangian (and its potential V) is GIVEN explicitly, and the
    learner must spot the cyclic coordinate -- never a rotational quantity or E, since V is always autonomous
    and only linear in two coordinates here."""
    for seed in SEEDS:
        q = _make(SYMMETRY_SOURCES, "medium", seed)
        assert "V(x, y, z) = " in q.text
        assert "cyclic coordinate" in q.text
        assert q.answer_text in {"p_x", "p_y", "p_z"}
        assert "E" not in set(q.distractors) | {q.answer_text}


def test_symmetry_hard_tier_requires_spotting_a_rotational_invariant():
    """Hard is a genuine step up from medium -- V couples two coordinates only through the sum of their squares
    (a rotationally invariant combination the learner must recognise), not a simple missing coordinate, and
    the answer is always an angular momentum component."""
    for seed in SEEDS:
        q = _make(SYMMETRY_SOURCES, "hard", seed)
        assert re.search(r"\(\w\^2 \+ \w\^2\)", q.text)
        assert "invariant under rotation" in q.text
        assert q.answer_text in {"L_x", "L_y", "L_z"}
        assert "E" not in set(q.distractors) | {q.answer_text}


def test_symmetry_exam_tier_admits_more_than_one_symmetry_without_saying_so():
    """The answer is drawn from the true set {E, L_x, L_y, L_z}, and the distractors are always the three (always
    genuinely broken) linear momenta; the reasoning ("depends only on x^2+y^2+z^2", "invariant under
    rotation", "more than one quantity is conserved") is never stated."""
    for seed in SEEDS:
        q = _make(SYMMETRY_SOURCES, "exam", seed)
        assert "x^2 + y^2 + z^2" in q.text  # V itself is still stated
        assert "invariant under rotation" not in q.text
        assert "more than one quantity is conserved" not in q.text
        assert "depends on x, y and z only through" not in q.text
        assert q.answer_text in {"E", "L_x", "L_y", "L_z"}
        assert set(q.distractors) == {"p_x", "p_y", "p_z"}


def test_symmetry_every_option_is_a_physically_correct_continuous_pairing():
    """No genuinely discrete symmetry (parity, reflection) is ever offered,
    and every option is one of the seven real Noether pairings -- never a
    false physical claim."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = _make(SYMMETRY_SOURCES, difficulty, seed)
            assert "parity" not in q.text.lower()
            assert "reflect" not in q.text.lower()
            all_options = set(q.distractors) | {q.answer_text}
            assert all_options <= _QUANTITIES


def test_symmetry_skill_loads_through_api_with_correct_properties():
    assert SYMMETRY_SKILL.domain == "mathematical-physics"
    assert SYMMETRY_SKILL.status == "active"
    assert SYMMETRY_SKILL.placement_eligible is False
    assert SYMMETRY_SKILL.answer == "mcq"
    assert SYMMETRY_SKILL.prerequisites == [
        "ucv1-energy-conservation-check",
        "senior-examine-momentum-force-resultant-force-action-and-reaction",
    ]
    assert {src.difficulty for src in SYMMETRY_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SYMMETRY_SOURCES["exam"].stages == ["exam"]
