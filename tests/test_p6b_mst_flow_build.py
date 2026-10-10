"""Minimal spanning trees and network flow (senior-use-minimal-spanning-trees-to-solve-minimal-connector,
senior-solve-small-scale-network-flow-problems-including-the)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p6b_mst_flow_build import (
    _verify_mst,
    _verify_flow,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

MST_TOPIC = "use_minimal_spanning_trees_to_solve_minimal_connector"
FLOW_TOPIC = "solve_small_scale_network_flow_problems_including_the"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [MST_TOPIC, FLOW_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [MST_TOPIC, FLOW_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


@pytest.mark.parametrize("topic", [MST_TOPIC, FLOW_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare make(topic, seed=s) with no difficulty should have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [MST_TOPIC, FLOW_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [MST_TOPIC, FLOW_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, int):
            wrong = question.answer + 1
        else:
            wrong = question.answer + 1.0
        assert not grade_number(question, str(wrong)).correct


# --- use_minimal_spanning_trees_to_solve_minimal_connector -------------------


def test_mst_always_integer_dollars():
    """MST answers are always integers (dollars)."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(MST_TOPIC, seed=seed, difficulty=difficulty)
            assert "." not in question.answer_text
            assert question.answer_text.isdigit()


def test_mst_independent_check_fires_on_corrupted_answer():
    """Independent check for MST should fire on mismatch."""
    # Test that the check fires on a known mismatch
    towns = ["A", "B", "C"]
    edges = [
        ("A", "B", 1),
        ("B", "C", 2),
        ("A", "C", 3),
    ]
    # The correct MST should have weight 1 + 2 = 3
    _verify_mst(towns, edges, 3, [("A", "B", 1), ("B", "C", 2)])
    # Should fail if mst_weight is wrong
    with pytest.raises(VerificationError):
        _verify_mst(towns, edges, 4, [("A", "B", 1), ("B", "C", 2)])


def test_mst_raises_on_unknown_tier():
    with pytest.raises(ValueError):
        make(MST_TOPIC, seed=1, tier="unknown")


# --- solve_small_scale_network_flow_problems_including_the -------------------


def test_flow_always_integer_capacity():
    """Flow answers are always integers (kL per hour)."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(FLOW_TOPIC, seed=seed, difficulty=difficulty)
            assert "." not in question.answer_text
            assert question.answer_text.isdigit()


def test_flow_independent_check_fires_on_corrupted_answer():
    """Independent check for flow should fire on mismatch."""
    # Easy flow: S -> A -> T
    nodes_easy = ["S", "A", "T"]
    edges_easy = [("S", "A", 10), ("A", "T", 8)]
    # The correct max flow is min(10, 8) = 8
    _verify_flow(nodes_easy, edges_easy, "S", "T", 8)
    # Should fail if flow is wrong
    with pytest.raises(VerificationError):
        _verify_flow(nodes_easy, edges_easy, "S", "T", 9)


def test_flow_raises_on_unknown_tier():
    with pytest.raises(ValueError):
        make(FLOW_TOPIC, seed=1, tier="unknown")


# --- Skills loading ---


def test_mst_skill_loads_via_load_skills():
    """MST skill should load with expected properties."""
    _, skills = load_skills()
    skill_dict = {s.id: s for s in skills}
    skill = skill_dict.get("senior-use-minimal-spanning-trees-to-solve-minimal-connector")
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert "simple-interest" in skill.prerequisites
    assert len(skill.sources) == 3  # easy, medium, hard


def test_flow_skill_loads_via_load_skills():
    """Flow skill should load with expected properties."""
    _, skills = load_skills()
    skill_dict = {s.id: s for s in skills}
    skill = skill_dict.get("senior-solve-small-scale-network-flow-problems-including-the")
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert "simple-interest" in skill.prerequisites
    assert len(skill.sources) == 3  # easy, medium, hard


# --- Naive-smallest distractor exercisability ---


def test_mst_naive_smallest_differs_from_answer_at_least_once():
    """The naive-smallest-without-cycle-check distractor should differ from true answer on at least one seed per tier."""
    from mathgen.generators.p6b_mst_flow_build import _build_graph, _kruskal_mst_weight
    import string

    for difficulty in ["easy", "medium", "hard"]:
        found_difference = False
        for seed in SEEDS[:10]:  # Sample first 10 seeds per tier
            if difficulty == "easy":
                n_towns, n_extra_edges = 4, 3
                weight_lo, weight_hi = 1, 20
            elif difficulty == "medium":
                n_towns, n_extra_edges = 5, 3
                weight_lo, weight_hi = 1, 25
            else:  # hard
                n_towns, n_extra_edges = 6, 4
                weight_lo, weight_hi = 1, 30

            import random

            rng = random.Random(seed)
            towns = list(string.ascii_uppercase[:n_towns])
            edges = _build_graph(rng, n_towns, n_extra_edges, weight_lo, weight_hi)
            mst_weight, _ = _kruskal_mst_weight(towns, edges)
            naive_smallest = sum(w for _, _, w in sorted(edges, key=lambda e: e[2])[: n_towns - 1])
            if naive_smallest != mst_weight:
                found_difference = True
                break
        assert found_difference, (
            f"naive-smallest should differ from MST weight on at least one seed for {difficulty} tier"
        )
