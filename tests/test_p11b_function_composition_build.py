"""Tests the two modes of `function_composition`, built on g(x) = sqrt(x - a) and f(x) = 1/(x - k).

Both draw g(x) = sqrt(x - a), f(x) = 1/(x - k) from the SAME
``function_composition`` topic, differentiated only by a ``mode`` param --
this file's ``test_fixture_is_shared_between_the_two_modes`` proves that
sharing directly, at 5 matching (difficulty, seed) pairs across both hard
classes.
"""

import math
import re
from collections import Counter
from fractions import Fraction

import pytest
from sympy import Symbol, sympify

from mathgen.grading.typed import grade_expression, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, symbolically_equal
from mathgen.generators.p11b_function_composition_build import (
    _build_compose,
    _build_when_defined,
    _verify_compose_numeric,
)

x = Symbol("x")
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]

_FIXTURE_RE = re.compile(r"f\(x\) = 1/\(x ([+-]) (\d+)\) and g\(x\) = sqrt\(x - (\d+)\)")


def _parse_ak(text: str) -> tuple[int, int]:
    sign, kabs, a = _FIXTURE_RE.search(text).groups()
    k = -int(kabs) if sign == "+" else int(kabs)
    return int(a), k


# --- determinism / distractor hygiene, both modes ---------------------------


@pytest.mark.parametrize("mode", ["compose", "when_defined"])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_and_have_unique_distractors(mode, difficulty):
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode=mode)
        repeat = make("function_composition", seed=seed, difficulty=difficulty, mode=mode)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == 3
        assert len(question.distractors) == len(set(question.distractors))


# --- the shared-fixture proof ------------------------------------------------


def test_fixture_is_shared_between_the_two_modes():
    """The concrete proof: 5 matching (difficulty,
    seed) pairs, including both hard classes, must give the two modes
    bit-identical (a, k) draws."""
    pairs = [("easy", 0), ("medium", 5), ("hard", 3), ("hard", 7), ("hard", 42)]
    rows = []
    for difficulty, seed in pairs:
        qc = make("function_composition", seed=seed, difficulty=difficulty, mode="compose")
        qd = make("function_composition", seed=seed, difficulty=difficulty, mode="when_defined")
        ac, kc = _parse_ak(qc.text)
        ad, kd = _parse_ak(qd.text)
        is_class_b = "except" in qd.answer_text
        rows.append((difficulty, seed, ac, kc, ad, kd, is_class_b))
        assert (ac, kc) == (ad, kd), (
            f"{difficulty} seed={seed}: compose(a,k)={(ac, kc)} != when_defined(a,k)={(ad, kd)}"
        )
    # Both hard classes are actually exercised by this pair set.
    hard_classes = {is_class_b for difficulty, _, _, _, _, _, is_class_b in rows if difficulty == "hard"}
    assert hard_classes == {True, False}, f"hard-tier pairs did not cover both classes: {rows}"
    print("\nfixture-identity proof (difficulty, seed, a, k, a, k, is_class_b):")
    for row in rows:
        print(" ", row)


# --- senior-find-the-composition-of-two-functions (ACMSM093, compose) ------


@pytest.mark.parametrize("difficulty", TIERS)
def test_compose_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode="compose")
        assert grade_expression(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_expression(question, distractor).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_compose_deliberately_wrong_answer_is_rejected(difficulty):
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode="compose")
        assert not grade_expression(question, "x + 999999").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_compose_answer_is_the_true_algebraic_composition(difficulty):
    """``mode="compose"`` now draws one of four prompt shapes, so the expected answer depends on which shape
    rendered -- detected the same way the oracle and the shape-coverage test below do, by a substring in
    ``question.text``, never by importing the generator's own shape names."""
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode="compose")
        a, k = _parse_ak(question.text)
        text = question.text
        if "find g(f(x))" in text:
            # reverse: g(f(x)) = sqrt(1/(x - k) - a).
            expected = sympify(f"sqrt(1/(x - {k}) - {a})")
            got = sympify(question.answer_text.replace("^", "**"))
            assert symbolically_equal(expected, got), f"seed={seed} {difficulty} reverse: {expected} != {got}"
        elif "evaluate f(g(" in text:
            # point_eval: answer isn't a function of x -- extract c from the
            # text, recompute m from c and a, and compare Fraction values.
            c = int(re.search(r"evaluate f\(g\((\d+)\)\)", text).group(1))
            m = math.isqrt(c - a)
            assert m * m == c - a, f"seed={seed} {difficulty} point_eval: c - a = {c - a} is not a perfect square"
            assert Fraction(question.answer_text) == Fraction(1, m - k), (
                f"seed={seed} {difficulty} point_eval: answer {question.answer_text!r} != 1/({m} - {k})"
            )
        else:
            # forward_a / forward_b: same math, wording differs only.
            expected = 1 / (sympify(f"sqrt(x - {a})") - k)
            got = sympify(question.answer_text.replace("^", "**"))
            assert symbolically_equal(expected, got), f"seed={seed} {difficulty}: {expected} != {got}"


def test_compose_hard_tier_covers_all_four_shapes_over_60_seeds():
    """Hard tier's shape pool has 4 choices (forward_a, forward_b, point_eval, reverse); confirm all four
    actually render over 60 seeds, detected by the same text substrings used throughout this file."""
    shapes = set()
    for seed in range(60):
        question = make("function_composition", seed=seed, difficulty="hard", mode="compose")
        text = question.text
        if "find g(f(x))" in text:
            shapes.add("reverse")
        elif "evaluate f(g(" in text:
            shapes.add("point_eval")
        elif "as a single expression" in text:
            shapes.add("forward_b")
        else:
            shapes.add("forward_a")
    assert shapes == {"forward_a", "forward_b", "point_eval", "reverse"}, shapes


def test_compose_independent_check_fires_on_a_corrupted_answer():
    a, k = 3, 5
    with pytest.raises(VerificationError):
        _verify_compose_numeric(a, k, "1/(sqrt(x - 3) - 999)")
    # The real answer must not raise.
    _verify_compose_numeric(a, k, "1/(sqrt(x - 3) - 5)")


@pytest.mark.parametrize("difficulty", TIERS)
def test_compose_at_least_20_distinct_prompts_over_60_seeds(difficulty):
    texts = {make("function_composition", seed=seed, difficulty=difficulty, mode="compose").text for seed in range(60)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct prompts over 60 seeds"


# --- senior-determine-when-the-composition-of-two-functions-is (ACMSM092) --


@pytest.mark.parametrize("difficulty", TIERS)
def test_when_defined_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode="when_defined")
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, str(correct_index)).correct
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(question, str(distractor_index)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_when_defined_deliberately_wrong_index_is_rejected(difficulty):
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode="when_defined")
        options = mcq_options(question)
        wrong_index = next(i for i in range(len(options)) if options[i] != question.answer_text)
        assert not grade_mcq(question, str(wrong_index)).correct


def test_when_defined_easy_is_always_the_no_exclusion_case():
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty="easy", mode="when_defined")
        assert "except" not in question.answer_text
        a, k = _parse_ak(question.text)
        assert k < 0


def test_when_defined_medium_is_always_the_exclusion_case():
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty="medium", mode="when_defined")
        assert "except" in question.answer_text
        a, k = _parse_ak(question.text)
        assert k >= 0


@pytest.mark.parametrize("difficulty", TIERS)
def test_when_defined_answer_matches_independent_sign_rule(difficulty):
    """Re-derive the classification from the raw (a, k) via the sign rule
    directly, a different route from reading the construction's own branch."""
    for seed in SEEDS:
        question = make("function_composition", seed=seed, difficulty=difficulty, mode="when_defined")
        a, k = _parse_ak(question.text)
        if k < 0:
            assert question.answer_text == f"Defined for all x >= {a}"
        else:
            assert question.answer_text == f"Defined for all x >= {a}, except x = {a + k * k}"


def test_when_defined_independent_check_fires_on_wrong_class():
    with pytest.raises(VerificationError):
        _build_when_defined(a=3, k=5, is_class_b=False)  # k >= 0 but claims no-exclusion class
    with pytest.raises(VerificationError):
        _build_when_defined(a=3, k=-5, is_class_b=True)  # k < 0 but claims exclusion class
    # Correct pairings must not raise.
    _build_when_defined(a=3, k=5, is_class_b=True)
    _build_when_defined(a=3, k=-5, is_class_b=False)


def test_when_defined_class_balance_at_400_seeds_hard_tier():
    """Report the class balance at 400 seeds, never 40 -- a fair coin's 400-draw majority routinely lands away
    from 50% by a wide margin without being unfair."""
    counts = Counter(
        "exclusion"
        if "except" in make("function_composition", seed=seed, difficulty="hard", mode="when_defined").answer_text
        else "no-exclusion"
        for seed in range(400)
    )
    print(f"\nwhen_defined hard-tier class balance at 400 seeds: {dict(counts)}")
    assert set(counts) == {"exclusion", "no-exclusion"}
    for label, n in counts.items():
        assert 0.15 <= n / 400 <= 0.85, f"{label}: {n}/400 = {n / 400:.2%}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_when_defined_at_least_20_distinct_prompts_over_60_seeds(difficulty):
    texts = {
        make("function_composition", seed=seed, difficulty=difficulty, mode="when_defined").text for seed in range(60)
    }
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct prompts over 60 seeds"


# --- skills loader ------------------------------------------------------


def test_both_skills_load_via_api_skills():
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}

    skill_a = skills_dict.get("senior-find-the-composition-of-two-functions")
    assert skill_a is not None
    assert skill_a.answer == "expression"
    assert "function-notation" in skill_a.prerequisites
    assert skill_a.placement_eligible is False
    assert {source.difficulty for source in skill_a.sources} == set(TIERS)
    assert all(source.params.get("mode") == "compose" for source in skill_a.sources)

    skill_b = skills_dict.get("senior-determine-when-the-composition-of-two-functions-is")
    assert skill_b is not None
    assert skill_b.answer == "mcq"
    assert "function-notation" in skill_b.prerequisites
    assert skill_b.placement_eligible is False
    assert {source.difficulty for source in skill_b.sources} == set(TIERS)
    assert all(source.params.get("mode") == "when_defined" for source in skill_b.sources)
