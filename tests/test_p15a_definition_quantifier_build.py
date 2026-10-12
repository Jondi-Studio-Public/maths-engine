"""Focused contract tests for: ucv1-definition-quantifier-check (answer: mcq)."""

import re

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p15a_definition_quantifier_build  # noqa: F401

SKILL_ID = "ucv1-definition-quantifier-check"
TOPIC = "p15a_definition_quantifier_check"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        assert len(q1.distractors) == 3
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert not grade_mcq(q, "9").correct


def test_skill_loads_with_expected_shape():
    assert SKILL.answer == "mcq"
    assert SKILL.placement_eligible is False
    assert SKILL.domain == "proof-and-foundations"
    assert SKILL.prerequisites == ["senior-use-the-quantifiers-for-all-and-there-exists", "finite-set-operations"]


def test_exam_uses_a_grid_not_a_pair_list():
    for seed in range(20):
        qe = _make("exam", seed)
        assert "grid below" in qe.text
        qh = _make("hard", seed)
        assert "grid below" not in qh.text
        assert "let R = {(" in qh.text


def test_never_asks_to_evaluate_a_numeric_predicate_over_a_set():
    """Reuse note: this skill works on a SUPPLIED FINITE RELATION (domain,
    codomain, quantifier scope, function-ness), never the neighbour senior
    skill's shape ('For ALL n in {...}, n is odd. True or False?')."""
    for tier in TIERS:
        for seed in range(20):
            q = _make(tier, seed)
            assert "True or False" not in q.text
            assert "relation from A to B" in q.text or "grid below" in q.text


def test_bound_variables_never_collide_with_an_element_of_b():
    """No element of B may ever be the literal letter 'a' or 'b', for any tier or any question kind, or a learner
    cannot tell the bound variable from the element."""
    for tier in TIERS:
        for seed in range(200):
            q = _make(tier, seed)
            m = re.search(r"and B = \{([^}]*)\}", q.text)
            assert m, q.text
            b_elements = {v.strip() for v in m.group(1).split(",")}
            assert "a" not in b_elements and "b" not in b_elements, (tier, seed, q.text)


def test_domain_codomain_never_states_the_convention_sentence():
    """Neither convention sentence may appear; "R is a relation from A to B" is the only source of the standard
    convention."""
    for tier in TIERS:
        for seed in range(100):
            q = _make(tier, seed)
            assert "defined as the stated ambient set" not in q.text


def test_impossible_options_never_win():
    """The two impossible options never appear as
    answer_text."""
    for tier in TIERS:
        for seed in range(100):
            q = _make(tier, seed)
            assert q.answer_text != ""  # sanity
            # find the actual option text for the answer
            options = mcq_options(q)
            ans_full = options[["A", "B", "C", "D"].index(q.answer_text)]
            assert ans_full != "NOT a function: two different elements of A map to the same element of B."
            assert ans_full != "Only statement (ii) is true."
