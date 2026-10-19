"""Tests `uc2-cylindrical-triple-integral`: the topic `s2a1_cylindrical` (easy, medium, hard) and the exam chain
`exd_x_s2a1_cylindrical`.

The keys are checked against a Cartesian Monte Carlo estimate that reads only the question text (see
`tests/s2a1_oracle.py`), and against exact values for the closed forms the families reduce to.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2a1_cylindrical_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests import s2a1_oracle as oracle

SKILL_ID = "uc2-cylindrical-triple-integral"
TOPIC = "s2a1_cylindrical"
EXAM_TOPIC = "exd_x_s2a1_cylindrical"
SK = oracle.skill(SKILL_ID)
TIERS = ("easy", "medium", "hard")


def test_source_wired_last():
    last = SK.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [s.difficulty for s in SK.sources] == ["easy", "medium", "hard", "exam"]
    assert list(SK.year_level) == ["university"] and "university-year-2" in SK.tags


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_cartesian_monte_carlo(tier):
    for seed in range(24):
        q = oracle.question(TOPIC, tier, seed)
        key = oracle.key_value(q.answer_text)
        assert oracle.mc_matches(q.text, key), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    oracle.check_key_and_distractors(SK, TOPIC, tier, range(60))


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_distractors_are_tagged_with_declared_errors(tier):
    oracle.check_tags(SK, EXAM_TOPIC if tier == "exam" else TOPIC, tier, range(30))


@pytest.mark.parametrize("tier", TIERS)
def test_answers_are_exact_multiples_of_pi_or_pi_plus_rational(tier):
    for seed in range(40):
        q = oracle.question(TOPIC, tier, seed)
        assert q.answer_kind == "expression"
        value = oracle.parse_typed(q.answer_text)
        assert value.has(sympy.pi) and value.is_real and value > 0, (seed, q.answer_text)


def _facts(q):
    text = q.text
    return dict(
        cylinder="solid E:" in text,
        bowl=re.search(r"^Find the volume of the solid (?:\d*\(?x² \+ y²\)?) ≤ z ≤ \d+\.$", text) is not None,
        cone="√" in text,
        slope="x ≥ 0" in text,
        ball="ball" in text,
        offset=re.search(r"≤ \d+[xy]\.$", text) is not None,
        paraboloids="- (x² + y²)" in text and "ball" not in text,
    )


def test_each_tier_has_the_shapes_the_design_asks_for():
    easy = [_facts(oracle.question(TOPIC, "easy", s)) for s in range(60)]
    assert sum(f["cylinder"] for f in easy) >= 15 and sum(f["bowl"] for f in easy) >= 15
    medium = [_facts(oracle.question(TOPIC, "medium", s)) for s in range(60)]
    assert sum(f["cone"] for f in medium) >= 15 and sum(f["slope"] for f in medium) >= 10
    hard = [_facts(oracle.question(TOPIC, "hard", s)) for s in range(90)]
    assert sum(f["ball"] for f in hard) >= 10 and sum(f["offset"] for f in hard) >= 10
    assert sum(f["paraboloids"] for f in hard) >= 10


def test_the_easy_tier_never_gives_the_cylindrical_limits_or_the_jacobian():
    for tier in TIERS:
        for seed in range(40):
            text = oracle.question(TOPIC, tier, seed).text
            assert "cylindrical" not in text and "dr" not in text and "θ" not in text, text


def test_medium_and_hard_need_more_than_a_full_cylinder():
    for tier in ("medium", "hard"):
        for seed in range(40):
            assert "solid E:" not in oracle.question(TOPIC, tier, seed).text


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [oracle.question(TOPIC, tier, seed).answer_text for seed in range(80)]
    assert len(set(answers)) >= (10 if tier != "hard" else 14)
    assert max(answers.count(a) for a in set(answers)) <= 14


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = oracle.question(TOPIC, tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == oracle.question(TOPIC, tier, seed).to_json()


def test_closed_forms():
    # the bowl x^2 + y^2 <= z <= h has volume pi h^2 / 2
    for seed in range(60):
        q = oracle.question(TOPIC, "easy", seed)
        found = re.match(r"^Find the volume of the solid x² \+ y² ≤ z ≤ (\d+)\.$", q.text)
        if found:
            h = int(found.group(1))
            assert oracle.parse_typed(q.answer_text) == sympy.pi * h * h / 2
    # a ball of radius R cut by a cylinder of radius a has volume 4 pi (R^3 - (R^2 - a^2)^(3/2)) / 3
    for seed in range(60):
        q = oracle.question(TOPIC, "hard", seed)
        found = re.match(
            r"^Find the volume of the solid that lies inside both the ball x² \+ y² \+ z² ≤ (\d+) and the cylinder x² \+ y² ≤ (\d+)\.$",
            q.text,
        )
        if found:
            big, small = int(found.group(1)), int(found.group(2))
            assert oracle.parse_typed(q.answer_text) == sympy.Rational(4, 3) * sympy.pi * (
                big ** sympy.Rational(3, 2) - (big - small) ** sympy.Rational(3, 2)
            )


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def test_exam_shape_and_marks():
    oracle.check_exam_shape(
        SK, EXAM_TOPIC, range(40), [3, 3, 3, 1], [{"expression"}, {"expression"}, {"fraction"}, {"fraction"}]
    )
    for seed in range(20):
        assert find_text_defects(oracle.question(EXAM_TOPIC, "exam", seed)) == []


def test_exam_keys_match_cartesian_monte_carlo():
    for seed in range(20):
        q = oracle.question(EXAM_TOPIC, "exam", seed)
        stem = q.stem
        constraints = oracle.region_constraints(stem)
        k, m = re.search(r"ρ = (\d+) \+ (\d*)z", stem).groups()
        rho = int(k) + int(m or 1) * oracle.z
        volume, mass, moment = (oracle.mc_integral(constraints, f) for f in (1 + 0 * oracle.z, rho, rho * oracle.z))
        keys = [oracle.key_value(p.answer_text) for p in q.parts]
        for estimate, key in zip((volume, mass, moment / mass, mass / volume), keys):
            assert abs(estimate - key) <= 0.04 * abs(key), (seed, stem, estimate, key)


def test_exam_key_accepted_every_distractor_rejected():
    oracle.check_exam_distractors(SK, EXAM_TOPIC, range(40))


def test_follow_through_carries_a_wrong_answer_forward():
    oracle.check_follow_through(SK, EXAM_TOPIC, range(40), minimum=40)


def test_exam_covers_both_solids():
    stems = [oracle.question(EXAM_TOPIC, "exam", s).stem for s in range(60)]
    assert sum("silo" in t for t in stems) >= 15 and sum("bowl" in t for t in stems) >= 15


def test_exam_answer_space_and_determinism():
    answers = [oracle.question(EXAM_TOPIC, "exam", seed).answer_text for seed in range(200)]
    assert len(set(answers)) >= 50
    assert oracle.question(EXAM_TOPIC, "exam", 7).to_json() == oracle.question(EXAM_TOPIC, "exam", 7).to_json()


def test_a_wrong_reply_is_not_accepted_for_the_whole_chain():
    q = oracle.question(EXAM_TOPIC, "exam", 3)
    assert not grade(SK, q, {p.label: "0" for p in q.parts}).correct
