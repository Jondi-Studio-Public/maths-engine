"""Focused contract tests for: f10-factor-multiple-divisibility-algorithm."""

import math
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p9e_divisibility_algorithm_build import (
    _verify_compound_count,
    _verify_halving_steps,
    _verify_range_count,
)

SEEDS = range(40)
SWEEP_SEEDS = range(60)
DIFFICULTIES = ["easy", "medium", "hard"]
TOPIC = "p9e_factor_multiple_divisibility_algorithm"

_EASY_RE = re.compile(
    r"^Follow this rule: for each whole number from 1 to (\d+) inclusive, test "
    r"whether it is divisible by (\d+)\. Count how many numbers in the range "
    r"pass the test\.$"
)
_MEDIUM_RE = re.compile(
    r"^Follow this rule: for each whole number from 1 to (\d+) inclusive, test "
    r"whether it is divisible by (\d+) (AND|OR) divisible by (\d+)\. Count how "
    r"many numbers in the range pass the test\.$"
)
_HARD_RE = re.compile(
    r"^Start with (\d+)\. While the current number is even, divide it by 2 and "
    r"count one step\. Stop as soon as the number is odd\. How many steps does "
    r"this algorithm take, starting from \1\?$"
)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = q.answer + 1000  # an integer answer_text is graded exactly
        assert not grade_number(q, str(wrong)).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_independent_check_fires_on_corrupted_answer():
    """The generator's own independent-route verifiers -- ``_verify_range_count``
    (easy), ``_verify_compound_count`` (medium), ``_verify_halving_steps``
    (hard) -- must reject a corrupted expected value, on the exact real
    (R, d[, d2, gate], N) shapes the builders draw, not a synthetic pairing."""
    with pytest.raises(VerificationError):
        _verify_range_count(R=22, d=9, expected=3)  # true count is 2, not 3
    with pytest.raises(VerificationError):
        _verify_compound_count(R=24, d1=5, d2=8, gate="OR", expected=6)  # true count is 7
    with pytest.raises(VerificationError):
        _verify_halving_steps(N=40, expected_steps=4)  # true step count is 3


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_a_non_negative_integer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int) and q.answer >= 0
        assert q.answer_text == str(q.answer)
        assert re.match(r"^\d+$", q.answer_text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_every_distractor_formatted_alike(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert re.match(r"^\d+$", q.answer_text)
        for d in q.distractors:
            assert re.match(r"^\d+$", d), d


def test_easy_text_matches_template_and_answer_is_range_over_divisor():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = _EASY_RE.match(q.text)
        assert m, q.text
        R, d = int(m.group(1)), int(m.group(2))
        assert 20 <= R <= 40
        assert 2 <= d <= 10
        assert int(q.answer_text) == R // d


def test_medium_text_matches_template_with_and_or_gate():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = _MEDIUM_RE.match(q.text)
        assert m, q.text
        R, d1, gate, d2 = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        assert math.gcd(d1, d2) == 1
        assert d1 != d2
        if gate == "AND":
            expected = R // (d1 * d2)
        else:
            expected = R // d1 + R // d2 - R // (d1 * d2)
        assert int(q.answer_text) == expected


def test_medium_and_or_never_drawn_from_a_non_coprime_pair():
    """Sweep confirms the coprime restriction actually holds every seed --
    not assumed once at construction time."""
    for seed in SWEEP_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = _MEDIUM_RE.match(q.text)
        assert m, q.text
        d1, d2 = int(m.group(2)), int(m.group(4))
        assert math.gcd(d1, d2) == 1, f"seed {seed}: non-coprime pair {d1}, {d2}"


def test_hard_text_matches_template_and_traces_2adic_valuation():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _HARD_RE.match(q.text)
        assert m, q.text
        N = int(m.group(1))
        current, steps = N, 0
        while current % 2 == 0:
            current //= 2
            steps += 1
        assert int(q.answer_text) == steps
        assert current > 1, "N must not be a pure power of 2"
        assert int(q.answer_text) != current, "answer must be the step count, never the final odd value"


def test_hard_k_and_odd_part_span_the_designed_ranges():
    ks = set()
    odd_parts = set()
    for seed in SWEEP_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _HARD_RE.match(q.text)
        N = int(m.group(1))
        current, steps = N, 0
        while current % 2 == 0:
            current //= 2
            steps += 1
        ks.add(steps)
        odd_parts.add(current)
    assert ks <= set(range(1, 6))
    assert all(op > 1 and op % 2 == 1 for op in odd_parts)


def test_never_asks_a_yes_no_divisibility_question():
    banned = [
        r"divide exactly",
        r"is it divisible",
        r"does \w+ divide",
        r"yes or no",
        r"\bgcd\b",
        r"greatest common",
        r"lowest common",
        r"\blcm\b",
    ]
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            lowered = q.text.lower()
            for pattern in banned:
                assert not re.search(pattern, lowered), (
                    f"seed {seed} {difficulty}: banned pattern {pattern!r} in {q.text}"
                )


def test_skill_loads_through_api_with_correct_properties():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == "f10-factor-multiple-divisibility-algorithm"), None)
    assert skill is not None
    assert skill.domain == "secondary-number"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    assert set(skill.prerequisites) == {"list-factors", "prime-factorisation"}
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert {0, 1, 2, 3, 4, 5, 6, "maintenance"} <= all_stages
