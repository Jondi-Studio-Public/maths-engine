"""Focused contract tests for f10-logarithmic-scales."""

import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10e_logarithmic_scales_build import _verify_log_position

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
COMBO_SEEDS = range(200)
TOPIC = "p10e_logarithmic_scales"

_FORWARD_RE = re.compile(
    r"On a logarithmic scale, A is marked at (\d+) and B is marked at (\d+), (\d+) equal "
    r"scale-steps apart \(each step means multiplying by 10\)\. A point lies "
    r"(\d+)/(\d+) of the way from A towards B(?: \(continuing past B along the "
    r"same equal-ratio pattern\))?\. What exact value sits at that point\?"
)
_INVERSE_RE = re.compile(
    r"On a logarithmic scale, A is marked at (\d+) and B is marked at (\d+), (\d+) equal "
    r"scale-steps apart \(each step means multiplying by 10\)\. The value (\d+) "
    r"sits at some position between A and B\."
)


def _parse(question):
    m = _FORWARD_RE.match(question.text)
    if m:
        A, B, k, j, k2 = (int(g) for g in m.groups())
        assert k == k2
        return "forward", A, B, k, j
    m = _INVERSE_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    A, B, k, value = (int(g) for g in m.groups())
    return "inverse", A, B, k, value


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_deterministic_and_distractors_hygienic(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.to_json() == repeat.to_json()
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) == 3


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct
        for distractor in q.distractors:
            assert not grade_number(q, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        kind, A, B, k, last = _parse(q)
        if kind == "forward":
            wrong = str(int(q.answer_text) * 7 + 3)  # never a power-of-ten multiple of A
        else:
            wrong = f"{float(q.answer_text) + 0.4:.2f}"
        assert not grade_number(q, wrong).correct


def test_easy_tier_is_always_the_fixed_midpoint_and_answer_is_ten_times_A():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        kind, A, B, k, j = _parse(q)
        assert kind == "forward"
        assert (k, j) == (2, 1)
        assert B == A * 100
        assert int(q.answer_text) == A * 10


def test_medium_tier_varies_k_and_j_and_never_hits_the_midpoint():
    seen = set()
    for seed in COMBO_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        kind, A, B, k, j = _parse(q)
        assert kind == "forward"
        assert k in (3, 4)
        assert 1 <= j <= k - 1
        assert 2 * j != k, "medium tier must never draw the swap-collision midpoint"
        seen.add((k, j))
    assert seen == {(3, 1), (3, 2), (4, 1), (4, 3)}


def test_hard_tier_both_extrapolation_and_inverse_occur_and_split_is_roughly_even():
    kinds = {"forward": 0, "inverse": 0}
    for seed in range(500):
        q = make(TOPIC, seed=seed, difficulty="hard")
        kind, *_ = _parse(q)
        kinds[kind] += 1
    assert kinds["forward"] > 0 and kinds["inverse"] > 0
    assert 0.3 < kinds["forward"] / sum(kinds.values()) < 0.7


def test_hard_tier_extrapolation_j_is_always_one_decade_beyond_k():
    for seed in COMBO_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        kind, A, B, k, j = _parse(q)
        if kind == "forward":
            assert j == k + 1
            assert "continuing past B" in q.text


def test_hard_tier_inverse_position_is_never_the_midpoint():
    for seed in COMBO_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        kind, A, B, k, value = _parse(q)
        if kind == "inverse":
            assert k == 4
            assert q.answer_text in ("0.25", "0.75")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_all_forward_answers_are_exact_powers_of_ten_times_A(difficulty):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        kind, A, B, k, j = _parse(q)
        if kind == "forward":
            value = int(q.answer_text)
            ratio = value / A
            # ratio must be an exact power of ten (positive or negative j never occurs here, so >= 1)
            power = round(math.log10(ratio)) if ratio >= 1 else 0
            assert A * (10**power) == value


@pytest.mark.parametrize("difficulty,expected_count", [("easy", 10), ("medium", 22), ("hard", 31)])
def test_exact_distinct_question_count_per_tier(difficulty, expected_count):
    """Each tier's draw pool is small and finite (see module docstring), so
    the convention is to assert the EXACT count, not merely `>= 20` -- easy's
    true pool (10) is below that bar. 500 seeds is enough to exhaust every
    tier's pool (measured directly, not assumed)."""
    seen = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(500)}
    assert len(seen) == expected_count, (
        f"{difficulty}: {len(seen)} distinct questions over 500 seeds, expected {expected_count}"
    )


def test_independent_log_check_fires_on_corrupted_answer():
    _verify_log_position(4, 400, 40, Fraction(1, 2))  # must not raise
    with pytest.raises(VerificationError):
        _verify_log_position(4, 400, 41, Fraction(1, 2))


def test_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, variant="not_a_variant")


def test_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["f10-logarithmic-scales"]

    assert skill.prerequisites == ["logarithm-laws", "scientific-notation"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC, "exd_x_n3_decibel_scale"}
    assert [source.stages for source in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in skill.misconceptions} == {
        "linear-interpolation",
        "off-by-one-decade",
        "swaps-reference-direction",
        "halves-the-ratio-not-the-exponent",
    }
