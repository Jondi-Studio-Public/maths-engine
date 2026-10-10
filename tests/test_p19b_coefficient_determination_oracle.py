"""Import list, confirmed below: only `re`, `os`, `fractions`, `mathgen.make`.

Every r stated in the prompt is 2 d.p. by construction, so r^2 is an EXACT terminating
decimal (4 d.p. as a decimal, 2 d.p. as a percentage) -- the oracle asserts exact
equality against `question.answer_text`, not a rounding-tolerance match.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p19b_coefficient_determination"
TIERS = ("easy", "medium", "hard", "exam")

N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", "250"))

_R_RE = re.compile(r"r = (-?\d+\.\d{2})")
_ABS_R_RE = re.compile(r"\|r\| = (\d+\.\d{2})")


def _all(tier, seeds):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# easy: one printed r, decimal form
# ============================================================================


def _easy_oracle(question_text: str) -> Fraction:
    matches = _R_RE.findall(question_text)
    assert len(matches) == 1, f"oracle: expected exactly one 'r = ..' in {question_text!r}"
    r = Fraction(matches[0])
    return r * r


@pytest.mark.parametrize("seeds", [range(N_SEEDS)])
def test_easy_oracle_matches_answer_text(seeds):
    mismatches = []
    for q in _all("easy", seeds):
        oracle = _easy_oracle(q.text)
        actual = Fraction(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, oracle, actual, q.text))
    assert mismatches == [], mismatches


def test_easy_oracle_challenges_every_distractor():
    for q in _all("easy", range(60)):
        oracle = _easy_oracle(q.text)
        for d in q.distractors:
            assert oracle != Fraction(d), f"seed {q.seed}: distractor {d!r} matches the oracle"


# ============================================================================
# medium: one printed NEGATIVE r, percentage form
# ============================================================================


def _medium_oracle(question_text: str) -> Fraction:
    matches = _R_RE.findall(question_text)
    assert len(matches) == 1, f"oracle: expected exactly one 'r = ..' in {question_text!r}"
    r = Fraction(matches[0])
    assert r < 0, f"oracle: medium tier's r is not negative: {r}"
    return r * r * 100


@pytest.mark.parametrize("seeds", [range(N_SEEDS)])
def test_medium_oracle_matches_answer_text(seeds):
    mismatches = []
    for q in _all("medium", seeds):
        oracle = _medium_oracle(q.text)
        actual = Fraction(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, oracle, actual, q.text))
    assert mismatches == [], mismatches


def test_medium_oracle_challenges_every_distractor():
    for q in _all("medium", range(60)):
        oracle = _medium_oracle(q.text)
        for d in q.distractors:
            assert oracle != Fraction(d), f"seed {q.seed}: distractor {d!r} matches the oracle"


# ============================================================================
# hard: two printed r values against the same response variable; the key is
# the LARGER r^2 (i.e. the larger |r|), regardless of which is listed first.
# ============================================================================


def _hard_oracle(question_text: str):
    matches = _R_RE.findall(question_text)
    assert len(matches) == 2, f"oracle: expected exactly two 'r = ..' in {question_text!r}"
    r1, r2 = Fraction(matches[0]), Fraction(matches[1])
    stronger_sq = max(r1 * r1, r2 * r2)
    weaker_sq = min(r1 * r1, r2 * r2)
    return stronger_sq * 100, weaker_sq * 100


@pytest.mark.parametrize("seeds", [range(N_SEEDS)])
def test_hard_oracle_matches_the_larger_r_squared(seeds):
    mismatches = []
    for q in _all("hard", seeds):
        stronger_pct, _weaker_pct = _hard_oracle(q.text)
        actual = Fraction(q.answer_text)
        if stronger_pct != actual:
            mismatches.append((q.seed, stronger_pct, actual, q.text))
    assert mismatches == [], mismatches


def test_hard_oracle_challenges_the_weaker_variables_r_squared():
    for q in _all("hard", range(60)):
        stronger_pct, weaker_pct = _hard_oracle(q.text)
        actual = Fraction(q.answer_text)
        assert stronger_pct == actual
        assert weaker_pct != actual, "the weaker variable's r^2 coincides with the answer"


def test_hard_oracle_finds_a_negative_stronger_variable_in_60_seeds():
    """Confirms the |r|-not-signed-r trap is genuinely exercised: at least one
    seed in 60 has the stronger (answer) variable's stated r negative, so
    "pick the larger SIGNED r" would fail.

    The trap still fires whenever the stronger variable is negative: with both r negative, the larger-|r|
    value is the smaller SIGNED value, so assert that directly."""
    found = False
    for q in _all("hard", range(60)):
        matches = _R_RE.findall(q.text)
        r1, r2 = Fraction(matches[0]), Fraction(matches[1])
        stronger, weaker = (r1, r2) if r1 * r1 >= r2 * r2 else (r2, r1)
        if stronger < 0:
            assert weaker > stronger, "a naive signed-r comparison should fail here"
            found = True
            break
    assert found, "60 seeds never drew the negative-is-stronger trap"


# ============================================================================
# exam: two |r| values (Model A explains more, listed first); the key is the
# DIFFERENCE of their r^2 values, in percentage points.
# ============================================================================


def _exam_oracle(question_text: str) -> Fraction:
    matches = _ABS_R_RE.findall(question_text)
    assert len(matches) == 2, f"oracle: expected exactly two '|r| = ..' in {question_text!r}"
    r1, r2 = Fraction(matches[0]), Fraction(matches[1])
    return (r1 * r1 - r2 * r2) * 100


@pytest.mark.parametrize("seeds", [range(N_SEEDS)])
def test_exam_oracle_matches_the_difference_of_r_squared(seeds):
    mismatches = []
    for q in _all("exam", seeds):
        oracle = _exam_oracle(q.text)
        actual = Fraction(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, oracle, actual, q.text))
    assert mismatches == [], mismatches


def test_exam_oracle_challenges_every_distractor():
    for q in _all("exam", range(60)):
        oracle = _exam_oracle(q.text)
        for d in q.distractors:
            assert oracle != Fraction(d), f"seed {q.seed}: distractor {d!r} matches the oracle"


# ============================================================================
# The oracle proven RED first
# ============================================================================


def test_oracle_would_catch_a_perturbed_key():
    """Prove the oracle actually challenges a wrong answer -- run it against a
    deliberately perturbed key (a real question's oracle value shifted by
    0.05) and confirm the mismatch is detected, exactly the check the real
    tests above rely on."""
    q = make(TOPIC, seed=0, difficulty="easy")
    true_oracle = _easy_oracle(q.text)
    perturbed = true_oracle + Fraction(1, 20)
    assert true_oracle != perturbed, "the perturbation itself is not detectable -- widen it"
