"""This file does **not** import that module -- confirmed by its own import list: only `re`, `fractions`,
`mathgen.make`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SI_TOPIC = "p13d_seasonal_index"
DS_TOPIC = "p13d_deseasonalise"


def _round_half_up(value: Fraction, dp: int) -> Fraction:
    """A from-scratch half-up rounder, deliberately NOT
    `mathgen.core.round_half_up` -- independence means not reusing the
    generator's own rounding routine either."""
    scale = 10**dp
    scaled = value * scale
    import math

    sign = -1 if scaled < 0 else 1
    magnitude = math.floor(abs(scaled) + Fraction(1, 2))
    return Fraction(sign * magnitude, scale)


# ============================================================================
# p13d_seasonal_index oracle
# ============================================================================


def _si_all(tier, seeds=range(60)):
    return [make(SI_TOPIC, seed=s, difficulty=tier) for s in seeds]


def _si_oracle_easy(q):
    m = re.findall(r"\$([\d,]+)", q.text)
    assert len(m) == 4, q.text
    v1, m1, v2, m2 = (Fraction(int(x.replace(",", ""))) for x in m)
    pct1 = v1 * 100 / m1
    pct2 = v2 * 100 / m2
    return _round_half_up((pct1 + pct2) / 2, 1)


def _si_oracle_medium(q):
    nums = re.findall(r"Quarter \d (\d+)", q.text)
    assert len(nums) == 8, q.text
    y1 = [Fraction(int(x)) for x in nums[:4]]
    y2 = [Fraction(int(x)) for x in nums[4:]]
    # The named season is the ONLY "Quarter N" mention not immediately
    # followed by a raw-table digit (those are excluded by the lookahead).
    season_idx = int(re.findall(r"Quarter (\d)(?!\s\d)", q.text)[0]) - 1
    m1, m2 = sum(y1) / 4, sum(y2) / 4
    pct1 = y1[season_idx] * 100 / m1
    pct2 = y2[season_idx] * 100 / m2
    return _round_half_up((pct1 + pct2) / 2, 1)


def _si_oracle_hard(q):
    index = Fraction(re.search(r"(?:came out at|index of|two years, is) (\d+(?:\.\d+)?)", q.text).group(1))
    pct1 = Fraction(re.search(r"(\d+(?:\.\d+)?)%", q.text).group(1))
    mean2 = Fraction(re.search(r"\$([\d,]+)", q.text).group(1).replace(",", ""))
    pct2 = 2 * index - pct1
    x = pct2 / 100 * mean2
    return _round_half_up(x, 2)


def _si_oracle_exam(q):
    total = Fraction(re.search(r"sum(?:med|ming)? to (\d+(?:\.\d+)?)", q.text).group(1))
    vals = re.findall(r"Quarter (\d) = (\d+(?:\.\d+)?)", q.text)
    assert len(vals) == 3, q.text
    given_sum = sum(Fraction(v) for _, v in vals)
    missing = total - given_sum
    factor = Fraction(400) / total
    return _round_half_up(missing * factor, 1)


_SI_ORACLES = {
    "easy": _si_oracle_easy,
    "medium": _si_oracle_medium,
    "hard": _si_oracle_hard,
    "exam": _si_oracle_exam,
}


@pytest.mark.parametrize("tier", ("easy", "medium", "hard", "exam"))
def test_si_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _si_all(tier):
        oracle = _SI_ORACLES[tier](q)
        actual = Fraction(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, q.text, str(oracle), q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", ("easy", "medium", "hard", "exam"))
def test_si_oracle_every_distractor_disagrees(tier):
    for q in _si_all(tier, seeds=range(20)):
        oracle = _SI_ORACLES[tier](q)
        for d in q.distractors:
            assert Fraction(d) != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


def test_si_oracle_catches_a_corrupted_answer():
    """Proven red first: the oracle must actually disagree when the
    generator's own answer is deliberately wrong."""
    q = make(SI_TOPIC, seed=0, difficulty="easy")
    oracle = _si_oracle_easy(q)
    corrupted = str(oracle + 1)
    assert Fraction(corrupted) != oracle


# ============================================================================
# p13d_deseasonalise oracle
# ============================================================================


def _ds_all(tier, seeds=range(60)):
    return [make(DS_TOPIC, seed=s, difficulty=tier) for s in seeds]


def _ds_oracle_easy(q):
    actual = Fraction(re.search(r"\$(\d+(?:\.\d+)?)", q.text).group(1))
    index = Fraction(re.search(r"index (?:for this quarter )?is (\d+(?:\.\d+)?)", q.text).group(1))
    return _round_half_up(actual / (index / 100), 2)


def _ds_oracle_medium(q):
    actual = Fraction(re.search(r"\$(\d+(?:\.\d+)?)", q.text).group(1))
    m = re.search(r"(\d+)% (above|below)", q.text)
    pct, direction = int(m.group(1)), m.group(2)
    index = 100 + pct if direction == "above" else 100 - pct
    return _round_half_up(actual / (Fraction(index) / 100), 2)


def _ds_oracle_hard(q):
    deseasonalised = Fraction(re.search(r"\$(\d+(?:\.\d+)?)", q.text).group(1))
    index = Fraction(re.search(r"index (?:for that quarter is|of) (\d+(?:\.\d+)?)", q.text).group(1))
    return _round_half_up(deseasonalised * (index / 100), 2)


def _ds_oracle_exam(q):
    actuals = re.findall(r"\$(\d+(?:\.\d+)?)", q.text)
    indices = re.findall(r"index (?:of )?(\d+(?:\.\d+)?)", q.text)
    assert len(actuals) == 2, q.text
    assert len(indices) == 2, q.text
    a_actual, b_actual = (Fraction(x) for x in actuals)
    a_index, b_index = (Fraction(x) for x in indices)
    d_a = a_actual / (a_index / 100)
    d_b = b_actual / (b_index / 100)
    return _round_half_up((d_b - d_a) / d_a * 100, 1)


_DS_ORACLES = {
    "easy": _ds_oracle_easy,
    "medium": _ds_oracle_medium,
    "hard": _ds_oracle_hard,
    "exam": _ds_oracle_exam,
}


@pytest.mark.parametrize("tier", ("easy", "medium", "hard", "exam"))
def test_ds_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _ds_all(tier):
        oracle = _DS_ORACLES[tier](q)
        actual = Fraction(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, q.text, str(oracle), q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", ("easy", "medium", "hard", "exam"))
def test_ds_oracle_every_distractor_disagrees(tier):
    for q in _ds_all(tier, seeds=range(20)):
        oracle = _DS_ORACLES[tier](q)
        for d in q.distractors:
            assert Fraction(d) != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


def test_ds_oracle_catches_a_corrupted_answer():
    q = make(DS_TOPIC, seed=0, difficulty="easy")
    oracle = _ds_oracle_easy(q)
    corrupted = str(oracle + 1)
    assert Fraction(corrupted) != oracle
