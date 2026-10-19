"""Tests `uc2-fourier-series-full-period` (topic `s2c3_fourier_series`) and its exam chain `exd_x_s2c3_fourier_series`.

The oracle reads the pieces of f back out of the question text and recomputes the coefficient by exact SymPy
integration over [-L, L], or the series value from the one-sided limits and the period.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.s2c3_fourier_series_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_follow_through,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from _s2c3_oracle import expr, full_range_coefficient, pieces, series_value
from mathgen import make

TOPIC = "s2c3_fourier_series"
EXAM = "exd_x_s2c3_fourier_series"
SKILL = skill_by_id("uc2-fourier-series-full-period")
TIERS = ("easy", "medium", "hard")
_ASK = re.compile(r"Find ([ab])([₀-₉]+)")
_SUB = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}
_POINT = re.compile(r"converge at x = (\S+?)\? Give")


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
    assert "fourier-coefficient-simple" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)
    assert "university-year-2" in SKILL.tags


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_coefficient_key_matches_exact_integration(tier):
    kinds = Counter()
    for seed in range(60):
        q = tier_question(TOPIC, tier, seed)
        kind, n = _which(q.text)
        kinds[(kind, n == 0)] += 1
        assert abs(full_range_coefficient(q.text, kind, n) - float(q.answer)) < 6e-4, (seed, q.text, q.answer_text)
        assert abs(float(q.answer)) >= 0.05
    assert len(kinds) >= 3


def test_easy_functions_are_steps_and_medium_functions_have_a_linear_piece():
    for seed in range(40):
        assert not any(body.free_symbols for body, _, _ in pieces(tier_question(TOPIC, "easy", seed).text)), seed
    linear = 0
    for seed in range(40):
        ps = pieces(tier_question(TOPIC, "medium", seed).text)
        linear += any(body.free_symbols for body, _, _ in ps)
    assert linear == 40


def test_hard_is_a_series_value_or_an_offset_break_coefficient():
    seen = Counter()
    for seed in range(120):
        q = tier_question(TOPIC, "hard", seed)
        m = _POINT.search(q.text)
        if m:
            seen["value"] += 1
            assert abs(series_value(q.text, expr(m.group(1))) - float(q.answer)) < 6e-4, (seed, q.text, q.answer_text)
        else:
            seen["coef"] += 1
            kind, n = _which(q.text)
            assert abs(full_range_coefficient(q.text, kind, n) - float(q.answer)) < 6e-4, (seed, q.text)
    assert seen["value"] >= 50 and seen["coef"] >= 20


def test_series_values_include_jumps_endpoints_and_shifted_points():
    kinds = Counter()
    for seed in range(200):
        q = tier_question(TOPIC, "hard", seed)
        m = _POINT.search(q.text)
        if not m:
            continue
        x0 = expr(m.group(1))
        ps = pieces(q.text)
        lo, hi = ps[0][1], ps[-1][2]
        if x0 >= hi or x0 < lo:
            kinds["outside"] += 1
        elif x0 == lo:
            kinds["endpoint"] += 1
        elif any(x0 == b for _, _, b in ps[:-1]):
            kinds["jump"] += 1
        else:
            kinds["inside"] += 1
    assert set(kinds) >= {"outside", "jump"}, kinds


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=10 if tier != "easy" else 6)


def test_distractors_have_the_answers_decimal_places():
    for tier in TIERS:
        for seed in range(40):
            q = tier_question(TOPIC, tier, seed)
            for d in q.distractors:
                assert len(d.split(".")[1]) == len(q.answer_text.split(".")[1]) == 3


# -- exam ---------------------------------------------------------------------------------------------------------


def test_exam_shape():
    assert_exam_grading(SKILL, EXAM, [3, 3, 1, 2], ["number"] * 4)
    q = make(EXAM, seed=3, difficulty="exam")
    assert [p.follow_through for p in q.parts] == [None, None, "sqrt(part_a**2 + part_b**2)", None]
    assert q.parts[2].sensible_range == (0, None)


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        an, bn = float(q.parts[0].answer), float(q.parts[1].answer)
        n = int("".join(_SUB[c] for c in re.search(r"Find a([₀-₉]+)", q.parts[0].prompt).group(1)))
        assert abs(full_range_coefficient(q.stem, "a", n) - an) < 6e-4, seed
        assert abs(full_range_coefficient(q.stem, "b", n) - bn) < 6e-4, seed
        assert abs(float(q.parts[2].answer) - (an * an + bn * bn) ** 0.5) < 2e-3, seed
        m = _POINT.search(q.parts[3].prompt + "?" + " Give")
        point = re.search(r"x = (\S+?)\? Give", q.parts[3].prompt).group(1)
        assert abs(series_value(q.stem, expr(point)) - float(q.parts[3].answer)) < 6e-4, seed
        assert m


def test_exam_follow_through_for_the_amplitude():
    import math

    assert_follow_through(
        SKILL,
        EXAM,
        "c",
        {"a": "2.000", "b": "1.500"},
        lambda q, replies: f"{math.hypot(2.0, 1.5):.3f}",
    )
