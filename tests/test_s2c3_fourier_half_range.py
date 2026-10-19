"""Tests `uc2-fourier-half-range` (topic `s2c3_fourier_half_range`) and its exam chain `exd_x_s2c3_fourier_half_range`.

The oracle reads the pieces of f from the question text and recomputes b_n or a_n by exact SymPy integration with
the factor 2/L, or the series value as the average of the odd or even periodic extension either side of the point.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.s2c3_fourier_half_range_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from _s2c3_oracle import expr, half_range_coefficient, half_range_value, pieces
from mathgen import make

TOPIC = "s2c3_fourier_half_range"
EXAM = "exd_x_s2c3_fourier_half_range"
SKILL = skill_by_id("uc2-fourier-half-range")
TIERS = ("easy", "medium", "hard")
_SUB = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}
_ASK = re.compile(r"Find ([ab])([₀-₉]+)")
_POINT = re.compile(r"converge at x = (\S+?)\? Give")


def _series(text):
    return "b" if "Fourier sine series" in text else "a"


def _which(text):
    m = _ASK.search(text)
    return m.group(1), int("".join(_SUB[c] for c in m.group(2)))


def test_skill_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert "uc2-fourier-series-full-period" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_coefficient_key_matches_exact_integration(tier):
    seen = Counter()
    for seed in range(80):
        q = tier_question(TOPIC, tier, seed)
        series, n = _which(q.text)
        seen[(series, n == 0)] += 1
        assert _series(q.text) == series
        assert abs(half_range_coefficient(q.text, series, n) - float(q.answer)) < 6e-4, (seed, q.text, q.answer_text)
    assert set(seen) >= {("b", False), ("a", False)}


def test_easy_is_one_piece_and_medium_is_two():
    for seed in range(40):
        assert len(pieces(tier_question(TOPIC, "easy", seed).text)) == 1
        assert len(pieces(tier_question(TOPIC, "medium", seed).text)) == 2


def test_hard_values_match_the_extension():
    kinds = Counter()
    for seed in range(120):
        q = tier_question(TOPIC, "hard", seed)
        point = expr(_POINT.search(q.text).group(1))
        assert abs(half_range_value(q.text, _series(q.text), point) - float(q.answer)) < 6e-4, (
            seed,
            q.text,
            q.answer_text,
        )
        ps = pieces(q.text)
        lo, hi = ps[0][1], ps[-1][2]
        kinds["outside" if point < lo or point > hi else "inside"] += 1
        kinds[_series(q.text)] += 1
    assert kinds["outside"] >= 20 and kinds["inside"] >= 20 and kinds["a"] >= 20 and kinds["b"] >= 20, kinds


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=6)


def test_exam_shape_and_keys():
    assert_exam_grading(SKILL, EXAM, [3, 3, 2, 2], ["number"] * 4)
    for seed in range(50):
        q = make(EXAM, seed=seed, difficulty="exam")
        n = int("".join(_SUB[c] for c in re.search(r"find b([₀-₉]+)", q.parts[0].prompt).group(1)))
        assert abs(half_range_coefficient(q.stem, "b", n) - float(q.parts[0].answer)) < 6e-4, seed
        assert abs(half_range_coefficient(q.stem, "a", n) - float(q.parts[1].answer)) < 6e-4, seed
        p_sine = expr(re.search(r"x = (\S+?)\? Give", q.parts[2].prompt).group(1))
        p_cos = expr(re.search(r"x = (\S+?)\? Give", q.parts[3].prompt).group(1))
        assert abs(half_range_value(q.stem, "b", p_sine) - float(q.parts[2].answer)) < 6e-4, seed
        assert abs(half_range_value(q.stem, "a", p_cos) - float(q.parts[3].answer)) < 6e-4, seed
