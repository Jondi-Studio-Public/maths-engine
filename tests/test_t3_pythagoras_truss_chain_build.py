"""Tests for pythagoras_truss_chain (task 540)."""

import math

import pytest

from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct
from mathgen.generators.t3_pythagoras_truss_chain_build import _verify_chain_identities

# Reuse the oracle's independent parser for the constructibility sweep rather than duplicating a second one.
from tests.test_t3_pythagoras_truss_chain_oracle import parse_chain, parse_target

TOPIC = "pythagoras_truss_chain"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]


def _sweep(seeds=SEEDS):
    for tier in TIERS:
        for seed in seeds:
            yield tier, seed, make(TOPIC, seed=seed, difficulty=tier)


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_accepts_answer(difficulty):
    grader = GRADERS["number"]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        result = grader(q, q.answer_text)
        assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_rejects_distractors(difficulty):
    grader = GRADERS["number"]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for dist in q.distractors:
            result = grader(q, dist)
            assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_rejects_wrong_answer(difficulty):
    grader = GRADERS["number"]
    for seed in range(10):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = str(int(q.answer_text) + 137)
        result = grader(q, wrong)
        assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_plain_integer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)
        int(q.answer_text)  # raises if not a plain integer string
        assert "." not in q.answer_text and "/" not in q.answer_text
        for dist in q.distractors:
            int(dist)


@pytest.mark.parametrize("difficulty", TIERS)
def test_working_text_nonempty(difficulty):
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.working, list) and len(q.working) >= 2
        for step in q.working:
            assert isinstance(step, str) and step


def test_every_stage_triangle_is_constructible():
    """For 60 seeds x each tier, parse every (leg, leg, hyp) triple stated
    or implied by question.text (via the oracle's independent parser) and
    re-assert leg1**2 + leg2**2 == hyp**2, every side > 0, hyp > each leg."""
    checked = 0
    for tier, seed, q in _sweep():
        chain = parse_chain(q.text)
        a1, c1, h1 = chain["a1"], chain["c1"], chain["h1"]
        assert a1 > 0 and h1 > 0 and c1 > 0
        assert a1**2 + h1**2 == c1**2
        assert c1 > a1 and c1 > h1
        checked += 1

        if chain["length"] == 3:
            c2, h2 = chain["c2"], chain["h2"]
            assert h1 > 0 and h2 > 0 and c2 > 0
            assert h1**2 + h2**2 == c2**2
            assert c2 > h1 and c2 > h2
            checked += 1

        a_last, c_last, shared_prev = chain["a_last"], chain["c_last"], chain["shared_prev"]
        assert a_last > 0 and shared_prev > 0 and c_last > 0
        assert a_last**2 + shared_prev**2 == c_last**2
        assert c_last > a_last and c_last > shared_prev
        checked += 1
    assert checked >= 60 * 3 * 2  # >=2 constructible triangles checked per question


@pytest.mark.parametrize("difficulty", TIERS)
def test_easy_medium_always_ask_final_hypotenuse(difficulty):
    if difficulty == "hard":
        pytest.skip("hard's target varies by design")
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert parse_target(q.text) == "cL"


def test_hard_target_varies_over_all_three_quantities():
    targets = {parse_target(make(TOPIC, seed=seed, difficulty="hard").text) for seed in range(60)}
    assert targets == {"h1", "h2", "cL"}, f"only saw {targets} over 60 seeds"


# --- both in-generator check() calls fire on a corrupted input ------------


def test_pythagorean_identity_check_fires_on_corrupted_leg():
    good = dict(a1=3, c1=5, h1=4, c2=None, h2=None, a_last=3, c_last=5)
    _verify_chain_identities(2, good)  # sanity: does not raise on a real triple
    bad = dict(good, h1=5)  # 3^2 + 5^2 != 5^2
    with pytest.raises(VerificationError):
        _verify_chain_identities(2, bad)


def test_pythagorean_identity_check_fires_on_corrupted_closing_hypotenuse():
    # Genuine length-3 chain: 3-4-5, then leg 4 with 4-3-5 again (reusing
    # the same triple is fine here -- this test only exercises the isqrt
    # re-derivation, not the chain-drawing uniqueness rules), closing with
    # leg 5 and 5-12-13.
    good = dict(a1=3, c1=5, h1=4, c2=5, h2=3, a_last=4, c_last=5)
    _verify_chain_identities(3, good)  # sanity: does not raise on a real chain
    bad = dict(good, c_last=14)  # 4^2 + 3^2 == 25 != 14^2
    with pytest.raises(VerificationError):
        _verify_chain_identities(3, bad)


def test_check_distinct_fires_on_a_distractor_equal_to_the_answer():
    """The same check_distinct(...) mechanism the generator calls on every
    distractor candidate, exercised directly on a corrupted (colliding)
    candidate."""
    with pytest.raises(VerificationError):
        check_distinct(17, 17, msg="distractor equals the answer")


class TestSkillLoads:
    def test_loads_with_expected_fields(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "pythagoras-truss-chain"), None)
        assert skill is not None
        assert skill.domain == "trigonometry"
        assert skill.answer == "number"
        assert skill.prerequisites == ["pythagorean-hypotenuse", "pythagorean-leg"]
        assert skill.placement_eligible is False
        # 4th source added by packet E3, task 831: an `exam` source on a
        # separate `symmetric_truss` topic (mathgen/generators/exd_e3_
        # symmetric_truss_build.py) -- this skill's original three
        # `pythagoras_truss_chain` sources are untouched.
        assert len(skill.sources) == 4
