"""Tests `uc2-heat-equation-initial-fit` (topic `s2c3_heat_equation`) and its exam chain `exd_x_s2c3_heat_equation`.

The oracle reads the rod problem back out of the wording and re-solves it from Simpson-integrated coefficients and,
for a sample of seeds, by an explicit finite-difference scheme that shares nothing with the Fourier machinery.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest

import mathgen.generators.s2c3_heat_equation_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_follow_through,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from _s2c3_heat_oracle import Rod
from mathgen import make

TOPIC = "s2c3_heat_equation"
EXAM = "exd_x_s2c3_heat_equation"
SKILL = skill_by_id("uc2-heat-equation-initial-fit")
TIERS = ("easy", "medium", "hard")
_FIND_U = re.compile(r"Find u\((\d+), (\d+)\)")
_COEF = re.compile(r"find the coefficient ([ab])([₀-₉])")
_SUB = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}


def test_skill_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert "ucv1-heat-equation-mode" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)


def _check_value(q, rod, tol=6e-4):
    x0, t0 = (float(v) for v in _FIND_U.search(q.text).groups())
    assert abs(rod.u(x0, t0) - float(q.answer)) < tol, (q.text, q.answer_text, rod.u(x0, t0))
    return x0, t0


def test_easy_matches_the_separated_solution():
    families = Counter()
    for seed in range(60):
        q = tier_question(TOPIC, "easy", seed)
        rod = Rod(q.text)
        assert rod.finite
        families[rod.family()] += 1
        _check_value(q, rod)
    assert families["sin"] >= 15 and families["cos"] >= 15


def test_medium_matches_the_series():
    kinds = Counter()
    for seed in range(80):
        q = tier_question(TOPIC, "medium", seed)
        rod = Rod(q.text)
        if "coefficient" in q.text:
            fam, n = _COEF.search(q.text).group(1), int(_SUB[_COEF.search(q.text).group(2)])
            assert fam == ("b" if rod.family() == "sin" else "a")
            assert abs(rod.coefficient(n) - float(q.answer)) < 6e-4, (seed, q.text)
            kinds["coefficient"] += 1
        else:
            _check_value(q, rod)
            kinds["value"] += 1
        kinds[rod.bc] += 1
    assert kinds["coefficient"] >= 10 and kinds["value"] >= 30 and kinds["neumann"] >= 10 and kinds["dirichlet"] >= 25


def test_hard_has_unequal_end_temperatures_and_matches_the_series():
    for seed in range(60):
        q = tier_question(TOPIC, "hard", seed)
        rod = Rod(q.text)
        assert rod.bc == "dirichlet" and rod.t1 != rod.t2
        _check_value(q, rod)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_finite_difference_agrees_on_a_sample(tier):
    for seed in range(0, 40, 5):
        q = tier_question(TOPIC, tier, seed)
        if "coefficient" in q.text:
            continue
        rod = Rod(q.text)
        x0, t0 = (float(v) for v in _FIND_U.search(q.text).groups())
        assert abs(rod.finite_difference(x0, t0) - float(q.answer)) < 0.5, (tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=30)


def test_exam_shape_and_marks():
    assert_exam_grading(SKILL, EXAM, [2, 3, 3, 2], ["number"] * 4)
    q = make(EXAM, seed=2, difficulty="exam")
    assert [p.follow_through for p in q.parts] == [None, None, None, "log(part_b)/part_a"]


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        rod = Rod(q.stem)
        lam = rod.k * (3 * math.pi / rod.L) ** 2
        assert abs(lam - float(q.parts[0].answer)) < 6e-6, seed
        assert abs(rod.coefficient(3) - float(q.parts[1].answer)) < 6e-4, seed
        x0, t0 = (float(v) for v in re.search(r"Find u\((\d+), (\d+)\)", q.parts[2].prompt).groups())
        assert abs(rod.u(x0, t0) - float(q.parts[2].answer)) < 6e-4, seed
        # at the answer time the amplitude of mode 3 is 1 degree
        t = float(q.parts[3].answer)
        assert abs(rod.coefficient(3) * math.exp(-lam * t) - 1.0) < 0.03, seed


def test_exam_follow_through_for_the_fade_time():
    assert_follow_through(
        SKILL,
        EXAM,
        "d",
        {"a": "0.05000", "b": "20.000"},
        lambda q, replies: f"{math.log(20.0) / 0.05:.1f}",
    )
