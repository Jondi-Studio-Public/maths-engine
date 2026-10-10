"""Focused contract tests for: ucv1-tree-spanning-edge-count (answer: number)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

TREE_ID = "ucv1-tree-spanning-edge-count"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
CONCENTRATION_SEEDS = range(200)
TIERS = ["easy", "medium", "hard", "exam"]

_NUMBER_RE = re.compile(r"-?\d+")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


TREE_SKILL, TREE_SOURCES = _skill_sources(TREE_ID)


def _make(difficulty, seed):
    src = TREE_SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = str(int(q.answer) + 1000)
        assert not grade_number(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_an_exact_non_negative_integer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert _NUMBER_RE.fullmatch(q.answer_text), q.answer_text
        assert isinstance(q.answer, int)
        assert q.answer >= 0


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _NUMBER_RE.fullmatch(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_200_seed_answer_concentration_under_25_percent(difficulty):
    counts = Counter(_make(difficulty, seed).answer_text for seed in CONCENTRATION_SEEDS)
    share = max(counts.values()) / len(CONCENTRATION_SEEDS)
    assert share <= 0.25, f"{difficulty}: top answer share {share:.3f} over 200 seeds"


def test_medium_degree_sequence_is_a_valid_tree_sequence():
    """Every listed degree is >= 1 and the sequence sums to 2(n-1) -- a real,
    realizable tree degree sequence, never an impossible one."""
    pat = re.compile(r"degrees(?: is| are)?:?\s*([\d, ]+)\.")
    for seed in SEEDS:
        q = _make("medium", seed)
        m = re.search(r": ([\d, ]+)\.", q.text)
        assert m, q.text
        degrees = [int(x) for x in m.group(1).split(",")]
        assert all(d >= 1 for d in degrees), degrees
        n = len(degrees)
        assert sum(degrees) == 2 * (n - 1), (n, degrees)
        assert int(q.answer_text) == n - 1


def test_hard_never_introduces_a_cycle_and_uses_binary_tree_vertex_formula():
    """Hard tier is now a full binary tree given its LEAF count L (n = 2L - 1), not a depth-based complete binary
    tree -- rewritten to match the redesigned structural family, not to mask a defect."""
    pat = re.compile(r"(\d+) leaves")
    for seed in SEEDS:
        q = _make("hard", seed)
        m = pat.search(q.text)
        assert m, q.text
        L = int(m.group(1))
        n = 2 * L - 1
        assert int(q.answer_text) == n - 1


def test_exam_extra_edges_is_given_total_minus_spanning_tree_baseline():
    """The oracle route here derives n from L, then n - 1, then subtracts from the stated total to recover
    `extra`, exactly as the redesigned prompt requires the learner to do. Rewritten to match the redesigned
    prompt, not to mask a defect."""
    lpat = re.compile(r"(\d+) leaves")
    mpat = re.compile(r"(\d+) edges in total|(\d+) edges altogether|for (\d+) edges in total")
    for seed in SEEDS:
        q = _make("exam", seed)
        L = int(lpat.search(q.text).group(1))
        n = 2 * L - 1
        m_match = mpat.search(q.text)
        assert m_match, q.text
        m_total = int(next(g for g in m_match.groups() if g is not None))
        extra = m_total - (n - 1)
        assert int(q.answer_text) == extra
        assert m_total == (n - 1) + extra


def test_never_a_weighted_graph_optimisation_shape():
    """This skill never has edge weights and never asks the learner to
    choose which edges to keep -- the MST neighbour's whole shape."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = _make(difficulty, seed)
            assert "cost" not in q.text.lower()
            assert "cable" not in q.text.lower()
            assert "minimum" not in q.text.lower()


def test_skill_loads_through_api_with_correct_properties():
    assert TREE_SKILL.domain == "discrete-mathematics"
    assert TREE_SKILL.status == "active"
    assert TREE_SKILL.placement_eligible is False
    assert TREE_SKILL.answer == "number"
    assert TREE_SKILL.prerequisites == []
    assert {src.difficulty for src in TREE_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = TREE_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


def test_independent_check_fires_on_corrupted_answer():
    """Exercise the generator's own check() calls directly: a stated_n
    answer that does not satisfy edges + 1 == n must raise."""
    from mathgen.core import VerificationError, check

    n, edges = 10, 8  # deliberately wrong: should be 9
    with pytest.raises(VerificationError):
        check(edges + 1 == n, "stated_n: edges + 1 does not recover n")


_TOKEN_RE_TEMPLATE = r"(?<!\d){value}(?!\d)"


def test_exam_prompt_never_states_the_answer_as_a_number_token():
    """Defect 2: the exam prompt must never contain answer_text as a standalone number token anywhere in
    question.text -- not as a substring match (so "38" failing to also rule out "138" is not a loophole), a
    whole-token regex with non-digit boundaries either side. Swept over >= 200 seeds."""
    failures = []
    for seed in range(200):
        q = _make("exam", seed)
        pattern = re.compile(_TOKEN_RE_TEMPLATE.format(value=re.escape(q.answer_text)))
        if pattern.search(q.text):
            failures.append((seed, q.answer_text, q.text))
    assert not failures, (
        f"{len(failures)}/200 exam seeds leak the answer as a literal token "
        f"in question.text; first failure: {failures[0]}"
    )


def test_hard_tier_pool_has_at_least_20_distinct_answers_at_200_seeds():
    """Defect 3: the hard tier's original single structural family (complete binary trees, depth 2-9) yields only
    8 distinct answers (6, 14, 30, 62, 126, 254, 510, 1022) over 200 seeds -- memorisable. The widened family
    must clear >= 20 distinct answers."""
    answers = {_make("hard", seed).answer_text for seed in range(200)}
    assert len(answers) >= 20, (
        f"hard tier: only {len(answers)} distinct answers over 200 seeds: {sorted(int(a) for a in answers)}"
    )
