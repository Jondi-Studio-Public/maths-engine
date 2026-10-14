"""Tests `apply_the_product_quotient_and_chain_rule_to`: distractor counts and distinctness checked at sympy
level.

Product, quotient and chain rule."""

from pathlib import Path

import re

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from sympy import symbols, simplify, sympify

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

TOPIC = "apply_the_product_quotient_and_chain_rule_to"


# --- Determinism & distractor hygiene ----------------------------------------


def test_bare_call_default_has_enough_distractors():
    """Bare `make(topic, seed=s)` calls must have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(TOPIC, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_from_answer_sympy_level(difficulty):
    """Distractors must be symbolically distinct from the answer."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        answer_expr = sympify(question.answer)
        for distractor in question.distractors:
            # Convert ^ back to ** for sympy parsing
            distractor_expr = sympify(distractor.replace("^", "**"))
            diff = simplify(answer_expr - distractor_expr)
            assert diff != 0, f"Seed {seed}/{difficulty}: distractor {distractor} equals answer symbolically"


# --- Medium and hard tier kind coverage -----


def test_medium_tier_covers_both_kinds():
    """Medium tier must exercise both quotient_trig and quotient_power across 40 seeds.

    Every question must classify as exactly one kind. The earlier version tested
    for the literal "1/x^", which stopped matching once the question pool was
    widened to forms like 3/(2x + 1)^4 -- and an unclassifiable shape simply
    vanished from the set instead of failing. Matching the family and requiring
    total classification means a new shape fails loudly.
    """
    quotient_trig = re.compile(r"sin\(.+?\)/cos\(.+?\)")
    quotient_power = re.compile(r"\d+/\(.+?\)\^\d+")
    kinds_seen = set()
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="medium")
        if quotient_trig.search(question.text):
            kinds_seen.add("quotient_trig")
        elif quotient_power.search(question.text):
            kinds_seen.add("quotient_power")
        else:
            raise AssertionError(f"Seed {seed}/medium: question matches neither medium-tier kind: {question.text!r}")
    assert kinds_seen == {"quotient_trig", "quotient_power"}, f"Medium tier missing kinds: {kinds_seen}"


def test_hard_tier_covers_all_kinds():
    """Hard tier must exercise all three kinds (chain_sin, chain_exp, product_chain) across 40 seeds."""
    kinds_seen = set()
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        if "sin(" in question.text and "/cos(" not in question.text and "·sin(" not in question.text:
            # This is a sin() but not part of product_chain (no dot before sin)
            if "exp(" not in question.text:
                kinds_seen.add("chain_sin")
        if "exp(" in question.text and "·sin(" not in question.text:
            kinds_seen.add("chain_exp")
        if "·sin(" in question.text:
            kinds_seen.add("product_chain")
    assert kinds_seen == {"chain_sin", "chain_exp", "product_chain"}, f"Hard tier missing kinds: {kinds_seen}"


# --- Real grader testing -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(difficulty):
    """Real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"Seed {seed}/{difficulty}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"Seed {seed}/{difficulty}: distractor {distractor} was accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        # For expression answers, use "x" as a deliberately wrong answer
        wrong = "x"
        result = grade_expression(question, wrong)
        assert not result.correct, f"Seed {seed}/{difficulty}: wrong answer '{wrong}' was accepted"


# --- No bare asterisk in prose text (but intentional in answer_text for typing) -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_no_bare_asterisk_in_prose_text(difficulty):
    """No bare '*' in question text (prose only; answer_text intentionally has *)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "*" not in question.text, f"Seed {seed}/{difficulty}: bare '*' in text"


# --- Skill loading -----


def test_skill_loads_with_correct_properties():
    """Product/quotient/chain rule skill must load with correct properties."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-apply-the-product-quotient-and-chain-rule-to"), None)
    assert skill is not None, "Skill not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible == False
    assert "differentiate-polynomial" in skill.prerequisites


def test_skill_sources_cover_correct_stages():
    """Skill must have three sources covering stages [0,1,2], [3,4], [5,6,maintenance]."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-apply-the-product-quotient-and-chain-rule-to"), None)
    assert skill is not None
    assert len(skill.sources) == 5
    assert skill.sources[3].difficulty == "exam"
    # Check that all sources have the correct topic
    for source in skill.sources[:3]:
        assert source.topic == "apply_the_product_quotient_and_chain_rule_to"
    # Check stages for each difficulty
    assert set(skill.sources[0].stages) == {0, 1, 2}, "easy stages"
    assert set(skill.sources[1].stages) == {3, 4}, "medium stages"
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}, "hard stages"
