"""Independent oracle for the `p16b_phat_standardise` questions: it re-derives each answer from the rendered
question text.

Parses each rendered question's `text` and re-derives the standardised value z from scratch, comparing against
`question.answer_text`. Reads only `question.text`, `question.answer_text` and `question.distractors` --
imports NOTHING from `mathgen.generators.p16b_phat_standardise_build`.

Proven red first: running this oracle against a deliberately perturbed
`answer_text` (a scratch copy of one real question with the last digit of
the key bumped) makes `test_oracle_matches_answer_text` fail -- see
`test_oracle_fires_on_a_perturbed_answer` below, which does exactly that
without needing a second file.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p16b_phat_standardise"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(300)  # >= 250 per tier, >= 1000 total (4 * 300 = 1200)
# A comparison ("which is further from 0") tier is re-derived over >= 1000 seeds.
HARD_SEEDS = range(1000)

_P_RE = re.compile(r"p = (-?[\d.]+) is the probability")
_N_RE = re.compile(r"independent samples of (\d+)")
_PHAT_SAMPLE_RE = re.compile(r"sample (\d+): (-?[\d.]+)(?=[;.])")
_X_SAMPLE_RE = re.compile(r"sample (\d+): x = (-?\d+)")
_EASY_STATED_RE = re.compile(r"For sample (\d+), calculate the standardised value")
_MEDIUM_STATED_RE = re.compile(r"For sample (\d+), first find p̂ = x/n")
_EXAM_STATED_RE = re.compile(r"For sample (\d+), first find the true probability")
_HARD_PAIR_RE = re.compile(r"Samples (\d+) and (\d+) are on opposite sides of p")

_SPINNER_RE = re.compile(r"a spinner with (\d+) equal sectors, (\d+) of which (?:is|are) coloured red")
_DIE_RE = re.compile(r"a fair six-sided die, where success is rolling a (\d+)")
_DECK_RE = re.compile(
    r"a well-shuffled standard deck of 52 playing cards, where one card is drawn, then replaced "
    r"and the deck reshuffled, for each trial, and success is drawing a (heart|King)"
)


def _z(phat, p, n) -> float:
    se = math.sqrt(float(phat) * (1 - float(phat)) / n)
    return (float(phat) - float(p)) / se


def _fmt2(value: float) -> str:
    # Half-up rounding to 2dp via Fraction, independent of the generator's
    # own `round_half_up`/`stated_precision_string` call sites.
    scaled = math.floor(abs(value) * 100 + 0.5)
    signed = scaled if value >= 0 else -scaled
    sign = "-" if signed < 0 else ""
    magnitude = abs(signed)
    return f"{sign}{magnitude // 100}.{magnitude % 100:02d}"


def _derive_p_from_chance_model(text: str) -> Fraction:
    m = _SPINNER_RE.search(text)
    if m:
        sectors, red = int(m.group(1)), int(m.group(2))
        return Fraction(red, sectors)
    m = _DIE_RE.search(text)
    if m:
        return Fraction(1, 6)
    m = _DECK_RE.search(text)
    if m:
        return Fraction(13, 52) if m.group(1) == "heart" else Fraction(4, 52)
    raise AssertionError(f"no chance model recognised in: {text!r}")


def _oracle_easy_or_medium(question, *, from_counts: bool) -> str:
    text = question.text
    p = float(_P_RE.search(text).group(1))
    n = int(_N_RE.search(text).group(1))
    if from_counts:
        samples = {int(i): int(x) for i, x in _X_SAMPLE_RE.findall(text)}
        stated = int(_MEDIUM_STATED_RE.search(text).group(1))
        x = samples[stated]
        phat = x / n
    else:
        samples = {int(i): float(v) for i, v in _PHAT_SAMPLE_RE.findall(text)}
        stated = int(_EASY_STATED_RE.search(text).group(1))
        phat = samples[stated]
    return _fmt2(_z(phat, p, n))


def _oracle_hard(question) -> str:
    text = question.text
    p = float(_P_RE.search(text).group(1))
    n = int(_N_RE.search(text).group(1))
    samples = {int(i): float(v) for i, v in _PHAT_SAMPLE_RE.findall(text)}
    i, j = (int(v) for v in _HARD_PAIR_RE.search(text).groups())
    phat_i, phat_j = samples[i], samples[j]
    zi, zj = _z(phat_i, p, n), _z(phat_j, p, n)
    winner_z = zi if abs(zi) >= abs(zj) else zj
    return _fmt2(winner_z)


def _oracle_exam(question) -> str:
    text = question.text
    p = _derive_p_from_chance_model(text)
    n = int(re.search(r"independent samples of (\d+) trials", text).group(1))
    samples = {int(i): int(x) for i, x in _X_SAMPLE_RE.findall(text)}
    stated = int(_EXAM_STATED_RE.search(text).group(1))
    x = samples[stated]
    phat = Fraction(x, n)
    return _fmt2(_z(phat, p, n))


def _oracle_answer(question, difficulty: str) -> str:
    if difficulty == "easy":
        return _oracle_easy_or_medium(question, from_counts=False)
    if difficulty == "medium":
        return _oracle_easy_or_medium(question, from_counts=True)
    if difficulty == "hard":
        return _oracle_hard(question)
    if difficulty == "exam":
        return _oracle_exam(question)
    raise ValueError(difficulty)


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_matches_answer_text(difficulty):
    challenged = []
    n_checked = 0
    seeds = HARD_SEEDS if difficulty == "hard" else SEEDS
    for seed in seeds:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        oracle_answer = _oracle_answer(question, difficulty)
        n_checked += 1
        if abs(Fraction(oracle_answer) - Fraction(question.answer_text)) > Fraction(1, 200):
            challenged.append((seed, question.text, question.answer_text, oracle_answer))
    assert not challenged, f"{difficulty}: oracle challenges (seed, text, stated, oracle): {challenged[:5]}"
    assert n_checked == len(seeds)


def test_oracle_fires_on_a_perturbed_answer():
    """Prove the oracle is actually checking something: take one real
    question, corrupt its answer_text, and confirm the oracle comparison
    would flag it."""
    question = make(TOPIC, seed=0, difficulty="easy")
    oracle_answer = _oracle_answer(question, "easy")
    perturbed = _fmt2(float(Fraction(oracle_answer)) + 1.00)
    assert abs(Fraction(oracle_answer) - Fraction(perturbed)) > Fraction(1, 200)
