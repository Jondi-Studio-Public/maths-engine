"""Independent oracle for ``mathgen.generators.p19d_phat_mean_sd_build``.

Never imports the generator module. Parses `question.text` for the stated
"p = ..." and "n = ..." literals (an exam-tier question states BOTH studies'
n values, plus which one is asked about, so the oracle picks the one named
in the final "for Study X" sentence) and independently recomputes
mean = p, sd = sqrt(p*(1-p)/n) using plain `fractions.Fraction`/`math.sqrt`,
then compares to `question.answer_text`/`question.answer`.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p19d_phat_mean_sd"
TIERS = ["easy", "medium", "hard", "exam"]

_P_RE = re.compile(r"p\s*=\s*(-?\d+(?:\.\d+)?)")
_N_RE = re.compile(r"n\s*=\s*(\d+)")
_STUDY_N_RE = re.compile(r"Study ([A-Z]) will sample n\s*=\s*(\d+)")
_TARGET_RE = re.compile(r"for Study ([A-Z])")


def _oracle_phat(text: str):
    """Independently recover (p, n) from the rendered question text."""
    target_match = _TARGET_RE.search(text)
    if target_match:
        # exam tier: two named studies, read off the one actually asked about.
        study_ns = dict(_STUDY_N_RE.findall(text))
        n = int(study_ns[target_match.group(1)])
    else:
        n_match = _N_RE.search(text)
        assert n_match, f"no n = ... found in {text!r}"
        n = int(n_match.group(1))

    p_matches = _P_RE.findall(text)
    assert p_matches, f"no p = ... found in {text!r}"
    # The last "p = ..." literal in the text is always the resolved decimal
    # (every tier restates it there, even when preceded by a fraction or a
    # chance-model derivation earlier in the sentence).
    p = Fraction(p_matches[-1])
    return p, n


def _oracle_answer(p: Fraction, n: int):
    mean = float(p)
    sd = math.sqrt(float(p) * (1 - float(p)) / n)
    return mean, sd


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_rederives_the_key_over_1000_seeds(difficulty):
    challenges = 0
    for seed in range(1000):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        p, n = _oracle_phat(q.text)
        oracle_mean, oracle_sd = _oracle_answer(p, n)
        assert abs(oracle_mean - q.answer["mean"]) < 1e-9, (seed, q.text, q.answer)
        if abs(oracle_sd - q.answer["sd"]) >= 5e-5:
            challenges += 1
    assert challenges == 0, f"{challenges} challenges over 1000 seeds ({difficulty})"


@pytest.mark.parametrize("difficulty", TIERS)
def test_red_proof_oracle_catches_a_flipped_key(difficulty):
    """If the oracle is wired correctly, swapping the answer for a random
    distractor must make it disagree essentially always."""
    disagreements = 0
    total = 0
    for seed in range(260):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        p, n = _oracle_phat(q.text)
        oracle_mean, oracle_sd = _oracle_answer(p, n)
        flipped = q.distractors[0]
        fm, fs = flipped.replace("mean = ", "").replace("sd = ", "").split(", ")
        flipped_mean, flipped_sd = float(fm), float(fs)
        total += 1
        if abs(oracle_mean - flipped_mean) >= 1e-9 or abs(oracle_sd - flipped_sd) >= 5e-5:
            disagreements += 1
    assert disagreements == total, f"only {disagreements}/{total} disagreed ({difficulty})"


def test_oracle_parses_all_four_tiers_without_exception():
    for tier in TIERS:
        for seed in range(50):
            q = make(TOPIC, seed=seed, difficulty=tier)
            p, n = _oracle_phat(q.text)
            assert 0 < p < 1
            assert n > 0
